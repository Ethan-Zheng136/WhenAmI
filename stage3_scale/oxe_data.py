"""27-domain OXE standardization, bounded raw-shard preparation and weighted sampling.

Independent of the submitted two-domain preparer. Full source scans remain
train-role filtered. DROID random exterior-camera choice is deterministically
seeded per serialized episode for stable content partitioning.
"""

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from PIL import Image, ImageEnhance

from stage3.common import STORE, DEFAULT_PAST, REFERENCE, reference_imports, sha256, write_json, source_fingerprint
from stage3.reader_study_data import episode_key
from stage3_scale.inventory import role_for_key, load_roles, REGISTRY
from stage3_scale.raw_data import OFFSETS, RawWindows, build_index
from stage3_scale.oxe_inventory import original_mixture


def oxe_source_hashes():
    return {p: sha256(REFERENCE / p) for p in (
        "prismatic/vla/datasets/rlds/oxe/mixtures.py",
        "prismatic/vla/datasets/rlds/oxe/configs.py",
        "prismatic/vla/datasets/rlds/oxe/materialize.py",
        "prismatic/vla/datasets/rlds/oxe/transforms.py",
        "prismatic/vla/datasets/rlds/oxe/utils/droid_utils.py",
        "prismatic/vla/datasets/rlds/utils/data_utils.py",
    )}


class EpisodeDecoder:
    def __init__(self, row, statistics):
        import tensorflow as tf
        import tensorflow_datasets as tfds
        tf.config.set_visible_devices([], "GPU")
        reference_imports()
        from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
        self.tf, self.tfds = tf, tfds
        self.name = row["dataset"]
        self.builder = tfds.builder(self.name, data_dir=str(STORE / "datasets/open-x-embodiment"))
        if Path(self.builder.data_dir).name != row["version"]:
            raise ValueError(f"{self.name}: TFDS version differs from inventory")
        self.kw = make_oxe_dataset_kwargs(
            self.name, STORE / "datasets/open-x-embodiment",
            load_camera_views=("primary",), load_depth=False, load_proprio=False,
            action_proprio_normalization_type=NormalizationType.NORMAL,
        )
        self.identity = deepcopy(statistics[self.name])
        self.identity["action"]["mean"], self.identity["action"]["std"] = [0.] * 7, [1.] * 7
        self.identity = {
            key: {k: np.asarray(v) for k, v in value.items()} if isinstance(value, dict) else np.asarray(value)
            for key, value in self.identity.items()
        }
        self.primary_key = self.kw["image_obs_keys"]["primary"]
        if self.primary_key is None:
            raise ValueError(f"{self.name}: primary image missing in original configuration")

    def __call__(self, serialized, ordinal=0):
        from dlimp.dataset import _broadcast_metadata_rlds
        from prismatic.vla.datasets.rlds.utils.data_utils import normalize_action_and_proprio, NormalizationType
        tf = self.tf
        decoded = self.builder.info.features.deserialize_example(
            serialized, decoders={"steps": self.tfds.decode.SkipDecoding()})
        trajectory = _broadcast_metadata_rlds(tf.constant(ordinal, dtype=tf.int64), decoded)
        # Retain original random-view semantics but make each episode's view
        # stable; never let rank/retry alter its split identity.
        if self.name == "droid":
            value = serialized.numpy() if hasattr(serialized, "numpy") else bytes(serialized)
            tf.random.set_seed(int.from_bytes(hashlib.sha256(value).digest()[:4], "little") % (2**31 - 1))
        trajectory = self.kw["standardize_fn"](trajectory)
        action = trajectory["action"]
        if action.shape.rank != 2 or action.shape[-1] != 7:
            raise ValueError(f"{self.name}: expected standardized7D action, got {action.shape}")
        action = normalize_action_and_proprio(
            {"action": tf.cast(action, tf.float32), "observation": {}},
            metadata=self.identity, normalization_type=NormalizationType.NORMAL,
        )["action"].numpy()
        images = trajectory["observation"][self.primary_key].numpy()
        language = trajectory[self.kw["language_key"]].numpy()
        if len(images) != len(action) or len(language) != len(action):
            raise ValueError(f"{self.name}: image/language/action trajectory lengths disagree")
        if images.ndim != 1 or language.ndim != 1:
            raise ValueError(f"{self.name}: expected encoded image/string sequence")
        return action, images, language

    def iterator(self, path):
        options = self.tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        options.autotune.enabled = False
        return self.tf.data.TFRecordDataset([str(path)], buffer_size=1024 * 1024).with_options(options)


def frame_indices(centre, length):
    """Boundary repeat never crosses an episode. At least one future step exists."""
    if length < 2 or not 0 <= centre < length - 1:
        raise ValueError("centre must have at least one real subsequent observation")
    return np.clip(np.asarray(OFFSETS) + centre, 0, length - 1).tolist()


def action_window(actions, centre):
    """Missing relative actions neutral, terminal gripper repeated, masked from loss."""
    n = min(16, len(actions) - centre)
    target = np.zeros((16, 7), np.float32)
    target[:n] = actions[centre:centre + n]
    target[n:, 6] = actions[-1, 6]
    valid = np.arange(16) < n
    return target, valid


def valid_centres(actions, images, language):
    if len(actions) < 2 or not np.isfinite(actions).all():
        return []
    return [t for t in range(len(actions) - 1)
            if language[t].decode().strip() and all(images[i] for i in frame_indices(t, len(actions)))]


def prepare_source(row, output, pinned, stats, decoder=None):
    path = Path(row["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
        raise ValueError("source changed since inventory")
    decoder = decoder or EpisodeDecoder(row, stats)
    n_scanned, episodes, chunks, pointer, seen = 0, [], [], 0, set()
    excluded = {"dev": 0, "test": 0, "short_or_invalid": 0, "duplicate": 0}
    for ordinal, serialized in enumerate(decoder.iterator(path)):
        n_scanned += 1
        actions, images, language = decoder(serialized, ordinal)
        centres = valid_centres(actions, images, language)
        if not centres:
            excluded["short_or_invalid"] += 1
            continue
        key = episode_key(row["dataset"], actions, language, images)
        role = role_for_key(row["dataset"], key, pinned)
        if role != "train":
            excluded[role] += 1
            continue
        if key in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(key)
        spans = []
        for frame in images:
            spans.append((pointer, len(frame)))
            chunks.append(frame)
            pointer += len(frame)
        episodes.append({
            "dataset": row["dataset"], "episode_key": key, "raw_actions": torch.from_numpy(actions.copy()),
            "frame_spans": torch.tensor(spans, dtype=torch.int64), "instructions": [x.decode() for x in language],
            "centres": torch.tensor(centres, dtype=torch.int64), "length": len(actions),
        })
    if n_scanned != row["source_episodes"]:
        raise ValueError(f"{row['id']}: {n_scanned} records, expected {row['source_episodes']}")
    temporary = output.with_suffix(".tmp")
    torch.save({
        "schema": "whenami_raw_train_shard_v1", "source_id": row["id"], "episodes": episodes,
        "frames": torch.from_numpy(np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()),
    }, temporary)
    os.replace(temporary, output)
    meta = {
        "complete": True, "source_id": row["id"], "dataset": row["dataset"], "file": str(output.resolve()),
        "bytes": output.stat().st_size, "sha256": sha256(output), "scanned_episodes": n_scanned,
        "excluded": excluded, "windows": sum(len(e["centres"]) for e in episodes),
        "episodes": [{"episode_key": e["episode_key"], "windows": len(e["centres"]), "length": e["length"]}
                     for e in episodes],
        "boundary_counts": {
            "history_padded_windows": sum(int((e["centres"] < 15).sum()) for e in episodes),
            "future_padded_windows": sum(int((e["centres"] + 15 >= e["length"]).sum()) for e in episodes),
        },
    }
    write_json(output.with_suffix(".json"), meta)
    return meta


def finalize(root, inventory, allow_partial=False):
    index = build_index(root, inventory, allow_partial)
    required = {name for name, _ in original_mixture()}
    found = set(index["window_counts"])
    if not allow_partial and found != required:
        raise ValueError(f"required OXE domains missing eligible windows: {required - found}")
    index.update(
        data_mix=inventory["data_mix"], sampling_probabilities=inventory["sampling_probabilities"],
        sampling_rule=inventory["sampling_rule"], oxe_source_sha256=oxe_source_hashes(),
        domain_count=len(found), metadata_versions={d["dataset"]: d["version"] for d in inventory["datasets"]},
        window_policy="episode_boundary_repeat_with_valid_action_mask",
    )
    # New name: a plain two-domain index can never masquerade as the 27-domain
    # index by containing complete=true.
    write_json(root / "oxe27_train_index.json", index)
    print("OXE27_INDEX", index["complete"], "domains", len(found), "windows", sum(index["window_counts"].values()), flush=True)
    return index


class OXERawWindows(RawWindows):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.meta.get("data_mix") != "oxe_magic_soup_plus":
            raise ValueError("not the original OXE mixture")
        probs = self.meta["sampling_probabilities"]
        if self.meta["complete"] and set(probs) != set(self.domains):
            raise ValueError("full OXE index lacks required domains")
        p = np.asarray([probs[d] for d in self.domains], dtype=np.float64)
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError("invalid domain weights")
        self.probabilities = p / p.sum()

    def get(self, domain, ordinal, augmentation_seed=None):
        k = int(np.searchsorted(self.ends[domain], ordinal, side="right"))
        entry = self.episodes[domain][k]
        local = ordinal - (int(self.ends[domain][k - 1]) if k else 0)
        data, e = self.load_episode(entry)
        centre = int(e["centres"][local])
        indices = frame_indices(centre, e["length"])
        images = []
        for index in indices:
            start, n = e["frame_spans"][index].tolist()
            images.append(Image.open(io.BytesIO(data["frames"][start:start + n].numpy().tobytes())).convert("RGB").resize((224, 224)))
        if augmentation_seed is not None:
            rng = random.Random(augmentation_seed)
            side = rng.randint(202, 224)
            x, y = rng.randint(0, 224 - side), rng.randint(0, 224 - side)
            bright, contrast, color = [rng.uniform(.8, 1.2) for _ in range(3)]
            images = [
                ImageEnhance.Color(ImageEnhance.Contrast(ImageEnhance.Brightness(
                    im.crop((x, y, x + side, y + side)).resize((224, 224))
                ).enhance(bright)).enhance(contrast)).enhance(color) for im in images
            ]
        from stage3_scale.model import tokenize_instruction
        from evaluation.past_protocol import normalize_actions
        tokens = tokenize_instruction(e["instructions"][centre], self.tokenizer, self.prompt_builder)
        pixels = [self.transform(im) for im in images]
        raw_actions, mask = action_window(e["raw_actions"].numpy(), centre)
        actions = normalize_actions(raw_actions, self.statistics[domain]["action"])
        return {
            "input_ids": tokens, "current": pixels[4],
            "past": {key: torch.stack([im[key] for im in pixels[:4]]) for key in pixels[0]},
            "future": {key: torch.stack([im[key] for im in pixels[5:]]) for key in pixels[0]},
            "actions": torch.from_numpy(actions), "action_mask": torch.from_numpy(mask),
            "identity": f"{domain}/{e['episode_key']}/{centre}",
        }

    def batch(self, count, seed, step, microstep, rank, device, augment=True):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, microstep, rank]))
        samples = []
        for _ in range(count):
            d = self.domains[int(rng.choice(len(self.domains), p=self.probabilities))]
            ordinal = int(rng.integers(self.ends[d][-1]))
            samples.append(self.get(d, ordinal, int(rng.integers(2**31)) if augment else None))
        maximum = max(len(s["input_ids"]) for s in samples)
        ids = torch.full((count, maximum), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, sample in enumerate(samples):
            ids[i, :len(sample["input_ids"])] = sample["input_ids"]
            mask[i, :len(sample["input_ids"])] = True
        return {
            "input_ids": ids.to(device), "attention_mask": mask.to(device),
            **{part: {k: torch.stack([s[part][k] for s in samples]).to(device) for k in samples[0][part]}
               for part in ("current", "past", "future")},
            "actions": torch.stack([s["actions"] for s in samples]).to(device),
            "action_mask": torch.stack([s["action_mask"] for s in samples]).to(device),
            "identities": [s["identity"] for s in samples],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
        os.environ.pop(key, None)
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("data_mix") != "oxe_magic_soup_plus" or len(inventory["datasets"]) != 27:
        raise ValueError("require original27 mixture inventory")
    if sha256(REGISTRY) != inventory["prior_role_registry_sha256"]:
        raise ValueError("episode registry changed")
    stats_path = Path(inventory["statistics_path"])
    if sha256(stats_path) != inventory["statistics_sha256"]:
        raise ValueError("pinned statistics changed")
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = {
        "inventory_sha256": sha256(args.inventory), "role_registry_sha256": sha256(REGISTRY),
        "statistics_sha256": sha256(stats_path), "oxe_sources": oxe_source_hashes(),
        "preparer_source_sha256": sha256(__file__), "limit_shards_per_domain": args.limit_shards,
        "reuse_raw_module_sha256": sha256(Path(__file__).with_name("raw_data.py")),
        "role_filter": "train_only", "window_policy": "episode_boundary_repeat_with_valid_action_mask",
        "droid_view": "original random exterior choice, deterministic serialized-episode seed",
    }
    path = args.output / "preparation_recipe.json"
    if not args.finalize_only:
        try:
            with path.open("x") as f:
                json.dump(recipe, f, indent=2)
        except FileExistsError:
            for attempt in range(100):
                try:
                    existing = json.loads(path.read_text())
                    break
                except json.JSONDecodeError:
                    time.sleep(.1)
            else:
                raise RuntimeError("incomplete recipe")
            if existing != recipe:
                raise ValueError("preparation recipe changed; use new output")
    if args.finalize_only:
        existing = json.loads(path.read_text())
        for k in ("inventory_sha256", "statistics_sha256", "oxe_sources", "preparer_source_sha256"):
            if existing[k] != recipe[k]:
                raise ValueError("cannot finalize changed preparation")
        finalize(args.output, inventory, args.allow_partial)
        return
    rows = inventory["source_shards"]
    if args.limit_shards:
        rows = [r for d in inventory["datasets"]
                for r in [row for row in rows if row["dataset"] == d["dataset"]][:args.limit_shards]]
    pinned, stats = load_roles(REGISTRY), json.loads(stats_path.read_text())
    decoder, active = None, None
    for count, row in enumerate(rows[rank::world]):
        out = args.output / "shards" / row["dataset"] / (row["id"].split("/")[-1] + ".pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.with_suffix(".json").exists():
            meta = json.loads(out.with_suffix(".json").read_text())
            if not meta["complete"] or meta["source_id"] != row["id"] or sha256(out) != meta["sha256"]:
                raise ValueError("bad completed source shard")
            continue
        if active != row["dataset"]:
            decoder, active = EpisodeDecoder(row, stats), row["dataset"]
        meta = prepare_source(row, out, pinned, stats, decoder)
        print("OXE_SHARD", rank, count, row["id"], "windows", meta["windows"], flush=True)
    write_json(args.output / f"worker-{rank:05d}.json", {"complete": True, "rank": rank, "world": world})
    if world == 1:
        finalize(args.output, inventory, args.allow_partial)


if __name__ == "__main__":
    main()
