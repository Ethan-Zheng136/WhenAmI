"""Raw full-corpus preparation and random-access windows for joint policy training.

No fixed H/z/C cache: images are decoded again on every training sample.
Episode role hashing uses the exact earlier standardization and content identity.
Only training episodes are emitted. Immutable per-source-shard outputs allow
independent CPU workers and safe restart without a global in-memory dataset.
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

from stage3.common import STORE, BASE, DEFAULT_PAST, reference_imports, sha256, write_json, source_fingerprint
from stage3.reader_study_data import episode_key
from stage3_scale.inventory import role_for_key, load_roles, REGISTRY

OFFSETS = (-15, -11, -7, -3, 0, 3, 7, 11, 15)


def prepare_source(row, output, pinned, statistics):
    """CPU-only single TFRecord conversion; no full dataset per worker."""
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    reference_imports()
    from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
    from prismatic.vla.datasets.rlds.utils.data_utils import normalize_action_and_proprio, NormalizationType
    name = row["dataset"]
    path = Path(row["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
        raise ValueError("source TFRecord changed since inventory")
    builder = tfds.builder(name, data_dir=str(STORE / "datasets/open-x-embodiment"))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.autotune.enabled = False
    ds = tf.data.TFRecordDataset([str(path)], buffer_size=1024 * 1024).with_options(options)
    episodes, chunks, pointer = [], [], 0
    scanned, excluded = 0, {"dev": 0, "test": 0, "short_or_invalid": 0, "duplicate": 0}
    seen = set()
    identity = deepcopy(statistics[name])
    identity["action"]["mean"], identity["action"]["std"] = [0.] * 7, [1.] * 7
    identity = {key: {k: np.asarray(v) for k, v in value.items()} if isinstance(value, dict) else np.asarray(value)
                for key, value in identity.items()}
    for raw in ds:
        scanned += 1
        decoded = builder.info.features.deserialize_example(raw, decoders={"steps": tfds.decode.SkipDecoding()})
        traj = OXE_STANDARDIZATION_TRANSFORMS[name](decoded["steps"])
        image_key = "image_0" if name == "bridge_orig" else "image"
        images = traj["observation"][image_key].numpy()
        language = traj["language_instruction"].numpy()
        # Reproduce previous Reader Study normalization identity, including its
        # unused-action-dimension handling, before computing content key.
        standardized = normalize_action_and_proprio(
            {"action": tf.cast(traj["action"], tf.float32), "observation": {}},
            metadata=identity, normalization_type=NormalizationType.NORMAL,
        )["action"].numpy()
        n = len(standardized)
        if n < 31 or standardized.shape != (n, 7) or not np.isfinite(standardized).all():
            excluded["short_or_invalid"] += 1
            continue
        key = episode_key(name, standardized, language, images)
        role = role_for_key(name, key, pinned)
        if role != "train":
            excluded[role] += 1
            continue
        if key in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(key)
        valid = [
            t for t in range(15, n - 15)
            if language[t].decode().strip() and all(images[t + o] for o in OFFSETS)
        ]
        if not valid:
            excluded["short_or_invalid"] += 1
            continue
        spans = []
        for frame in images:
            spans.append((pointer, len(frame)))
            chunks.append(frame)
            pointer += len(frame)
        episodes.append({
            "dataset": name, "episode_key": key, "raw_actions": torch.from_numpy(standardized.copy()),
            "frame_spans": torch.tensor(spans, dtype=torch.int64),
            "instructions": [x.decode() for x in language], "centres": torch.tensor(valid),
            "length": n,
        })
    if scanned != row["source_episodes"]:
        raise ValueError(f"{path}: expected {row['source_episodes']} episodes, read {scanned}")
    blob = torch.from_numpy(np.frombuffer(b"".join(chunks), dtype=np.uint8).copy())
    payload = {"schema": "whenami_raw_train_shard_v1", "source_id": row["id"],
               "frames": blob, "episodes": episodes}
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    metadata = {
        "complete": True, "source_id": row["id"], "dataset": name,
        "file": str(output.resolve()), "bytes": output.stat().st_size, "sha256": sha256(output),
        "scanned_episodes": scanned, "excluded": excluded,
        "episodes": [{"episode_key": e["episode_key"], "windows": len(e["centres"]),
                      "length": e["length"]} for e in episodes],
        "windows": sum(len(e["centres"]) for e in episodes),
    }
    write_json(output.with_suffix(".json"), metadata)
    return metadata


def build_index(root, inventory, allow_partial=False):
    """Rank0 merges immutable shard metadata; duplicate episodes get one owner."""
    rows = inventory["source_shards"]
    seen, indexed, source_records, counts = set(), [], [], {}
    missing = []
    for row in rows:
        p = root / "shards" / row["dataset"] / (row["id"].split("/")[-1] + ".json")
        if not p.exists():
            missing.append(row["id"])
            continue
        meta = json.loads(p.read_text())
        if not meta["complete"] or meta["source_id"] != row["id"]:
            raise ValueError("invalid shard completion")
        source_records.append({k: meta[k] for k in ("source_id", "file", "bytes", "sha256")})
        for i, episode in enumerate(meta["episodes"]):
            identity = (row["dataset"], episode["episode_key"])
            if identity in seen:
                continue
            seen.add(identity)
            indexed.append({"dataset": row["dataset"], "shard": meta["file"], "episode_index": i, **episode})
            counts[row["dataset"]] = counts.get(row["dataset"], 0) + episode["windows"]
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} unprepared source shards")
    if not indexed:
        raise ValueError("no eligible train windows")
    output = {
        "schema": "whenami_raw_full_train_index_v1", "complete": not missing,
        "scope": "partial_smoke" if missing else "full_source_train_filtered",
        "source_shards": len(rows), "completed_shards": len(source_records),
        "missing_shards": missing, "window_counts": counts, "episodes": indexed,
        "shard_records": source_records, "split_seed": 20261002,
        "prior_role_registry_sha256": inventory["prior_role_registry_sha256"],
        "source_reference_sha256": source_fingerprint(),
        "preparation_recipe_sha256": sha256(root / "preparation_recipe.json"),
        "notes": ["Only train-role episodes, all valid interior centres.", "No frozen VLM/Past features.",
                  "Duplicate content episodes assigned once. Development/test excluded, never emitted."],
    }
    write_json(root / "train_index.json", output)
    return output


class RawWindows:
    """Memory mapped raw compressed frames; one active shard per dataset.

Sampling is a deterministic function of (seed, step, microstep, rank). There is
no hidden TF shuffle buffer to restore. Epoch-based coverage is not claimed.
"""
    def __init__(self, index, transform, tokenizer, prompt_builder, statistics, allow_partial=False):
        self.index_path = Path(index)
        self.meta = json.loads(self.index_path.read_text())
        if self.meta["schema"] != "whenami_raw_full_train_index_v1":
            raise ValueError("wrong raw index")
        if not self.meta["complete"] and not allow_partial:
            raise ValueError("partial data index is smoke-only")
        self.transform, self.tokenizer, self.prompt_builder = transform, tokenizer, prompt_builder
        self.statistics = statistics
        self.domains = sorted(self.meta["window_counts"])
        self.episodes = {d: [e for e in self.meta["episodes"] if e["dataset"] == d] for d in self.domains}
        self.ends = {d: np.cumsum([e["windows"] for e in self.episodes[d]]) for d in self.domains}
        self.loaded = {}
        self.file_records = {record["file"]: record for record in self.meta["shard_records"]}
        self.verified = set()

    def load_episode(self, entry):
        d, path = entry["dataset"], entry["shard"]
        if path not in self.verified:
            record = self.file_records[path]
            if Path(path).stat().st_size != record["bytes"]:
                raise ValueError("prepared raw shard size changed")
            self.verified.add(path)
        if d not in self.loaded or self.loaded[d][0] != path:
            self.loaded[d] = (path, torch.load(path, weights_only=True, mmap=True, map_location="cpu"))
        data = self.loaded[d][1]
        e = data["episodes"][entry["episode_index"]]
        if e["episode_key"] != entry["episode_key"]:
            raise ValueError("episode index mismatch")
        return data, e

    def get(self, domain, ordinal, augmentation_seed=None):
        k = int(np.searchsorted(self.ends[domain], ordinal, side="right"))
        entry = self.episodes[domain][k]
        local = ordinal - (int(self.ends[domain][k - 1]) if k else 0)
        data, e = self.load_episode(entry)
        t = int(e["centres"][local])
        images = []
        for offset in OFFSETS:
            start, n = e["frame_spans"][t + offset].tolist()
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
        tokens = tokenize_instruction(e["instructions"][t], self.tokenizer, self.prompt_builder)
        pixels = [self.transform(im) for im in images]
        from evaluation.past_protocol import normalize_actions
        actions = normalize_actions(e["raw_actions"][t:t + 16].numpy(), self.statistics[domain]["action"])
        return {
            "input_ids": tokens, "current": pixels[4],
            "past": {key: torch.stack([im[key] for im in pixels[:4]]) for key in pixels[0]},
            "future": {key: torch.stack([im[key] for im in pixels[5:]]) for key in pixels[0]},
            "actions": torch.from_numpy(actions), "identity": f"{domain}/{e['episode_key']}/{t}",
        }

    def batch(self, count, seed, step, microstep, rank, device, augment=True):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, microstep, rank]))
        samples = []
        for _ in range(count):
            domain = self.domains[int(rng.integers(len(self.domains)))]
            ordinal = int(rng.integers(self.ends[domain][-1]))
            samples.append(self.get(domain, ordinal, int(rng.integers(2**31)) if augment else None))
        maximum = max(len(s["input_ids"]) for s in samples)
        ids = torch.full((count, maximum), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, s in enumerate(samples):
            ids[i, :len(s["input_ids"])] = s["input_ids"]
            mask[i, :len(s["input_ids"])] = True
        return {
            "input_ids": ids.to(device), "attention_mask": mask.to(device),
            **{part: {key: torch.stack([s[part][key] for s in samples]).to(device)
                      for key in samples[0][part]} for part in ("current", "past", "future")},
            "actions": torch.stack([s["actions"] for s in samples]).to(device),
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
    # No DDP is needed for independent CPU preparation. Erase rank environment
    # before any reference import initializes accelerate/NCCL.
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
        os.environ.pop(k, None)
    inventory = json.loads(args.inventory.read_text())
    if sha256(REGISTRY) != inventory["prior_role_registry_sha256"]:
        raise ValueError("role registry changed")
    pinned = load_roles(REGISTRY)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.finalize_only:
        print("INDEX", build_index(args.output, inventory, args.allow_partial)["window_counts"], flush=True)
        return
    recipe = {
        "inventory_sha256": sha256(args.inventory),
        "role_registry_sha256": sha256(REGISTRY),
        "normalization_statistics_sha256": sha256(DEFAULT_PAST.parents[1] / "dataset_statistics.json"),
        "preparer_source_sha256": sha256(__file__), "reference_source": source_fingerprint(),
        "offsets": list(OFFSETS), "window_mode": "all_valid_interior_centres",
        "role": "train_only", "limit_shards_per_dataset": args.limit_shards,
    }
    recipe_path = args.output / "preparation_recipe.json"
    # Exclusive creation: concurrent independent workers must all agree exactly.
    try:
        with recipe_path.open("x") as f:
            json.dump(recipe, f, indent=2)
    except FileExistsError:
        for attempt in range(100):
            try:
                previous = json.loads(recipe_path.read_text())
                break
            except json.JSONDecodeError:
                time.sleep(.05)
        else:
            raise RuntimeError("preparation recipe was not committed by its first writer")
        if previous != recipe:
            raise ValueError("preparation recipe changed: choose a new output")
    stats = json.loads((DEFAULT_PAST.parents[1] / "dataset_statistics.json").read_text())
    rows = inventory["source_shards"]
    if args.limit_shards:
        # Each domain's first few sources for smoke only; never finalize as full.
        rows = [row for domain in sorted({r["dataset"] for r in rows})
                for row in [r for r in rows if r["dataset"] == domain][:args.limit_shards]]
    selected = rows[rank::world]
    for i, row in enumerate(selected):
        out = args.output / "shards" / row["dataset"] / (row["id"].split("/")[-1] + ".pt")
        out.parent.mkdir(parents=True, exist_ok=True)
        done = out.with_suffix(".json")
        if done.exists():
            meta = json.loads(done.read_text())
            if meta["source_id"] != row["id"] or not out.exists() or sha256(out) != meta["sha256"]:
                raise ValueError("invalid existing source shard")
            continue
        result = prepare_source(row, out, pinned, stats)
        print("RAW_SHARD", rank, i + 1, "/", len(selected), row["id"], result["windows"], flush=True)
    write_json(args.output / f"worker-{rank:05d}.json",
               {"complete": True, "rank": rank, "world": world, "sources": len(selected)})
    if world == 1:
        print("INDEX", build_index(args.output, inventory, args.allow_partial)["window_counts"], flush=True)


if __name__ == "__main__":
    main()
