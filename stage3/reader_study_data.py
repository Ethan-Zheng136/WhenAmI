"""Fresh episode-partitioned train/dev plus physically separate sealed test raw data.

No reference source changes. The train_dev directory retains the existing frozen
endpoint cache contract; test samples are never part of that cache/statistics.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from stage3.common import (
    BASE, STORE, DEFAULT_PAST, DEFAULT_FUTURE, endpoint_record,
    source_fingerprint, sha256, write_json,
)


def episode_key(dataset, actions, instructions, images):
    """Content identity independent of enumeration/rank; no learning target uses this ID."""
    h = hashlib.sha256(dataset.encode())
    h.update(np.asarray(actions, np.float32).tobytes())
    h.update(str(np.asarray(actions).shape).encode())
    for j in sorted(set((0, len(actions) // 2, len(actions) - 1))):
        for value in (instructions[j], images[j]):
            data = value if isinstance(value, bytes) else np.asarray(value).tobytes()
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)
    return h.hexdigest()


def assign_partition(key, seed):
    n = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little") % 10000
    return "train" if n < 8000 else ("dev" if n < 9000 else "test")


def worker(spec):
    # Called in spawned processes; import torch before reference/TF libraries.
    import torch
    os.environ.update(WAI_BOUNDED_INPUT="1", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="2")
    from stage3.common import reference_imports
    reference_imports()
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds
    from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
    from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
    from prismatic.vla.datasets.rlds.obs_transforms import decode_and_resize
    from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
    import sys
    sys.path.insert(0, str(BASE / "evaluation"))
    from past_protocol import interior_centres

    name, root = spec["dataset"], Path(spec["shard"])
    root.mkdir(parents=True, exist_ok=False)
    offsets, quotas = spec["offsets"], spec["quotas"]
    capacity = sum(quotas.values())
    arrays = {
        "frames": np.lib.format.open_memmap(root / "frames.npy", mode="w+", dtype=np.uint8,
                                          shape=(capacity, len(offsets), 224, 224, 3)),
        "raw_actions": np.lib.format.open_memmap(root / "raw_actions.npy", mode="w+", dtype=np.float32,
                                               shape=(capacity, 16, 7)),
    }
    stats = deepcopy(spec["statistics"])
    stats["action"]["mean"], stats["action"]["std"] = [0.] * 7, [1.] * 7
    kwargs, _ = get_oxe_dataset_kwargs_and_weights(
        STORE / "datasets/open-x-embodiment", [(name, 1.)], load_camera_views=("primary",),
        load_depth=False, load_proprio=False, load_language=True,
        action_proprio_normalization_type=NormalizationType.NORMAL,
    )
    kwargs = kwargs[0]
    kwargs.update(dataset_statistics=stats, num_parallel_reads=1, num_parallel_calls=1, shuffle=False)
    ds, _ = make_dataset_from_rlds(**kwargs, train=True, load_all_data_for_training=False)
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    # Same deterministic standardized stream as old pilot; skip well beyond all
    # old sampled episodes, without decoding their images.
    ds = ds.skip(spec["skip_episodes"])
    counts = dict.fromkeys(quotas, 0)
    records, seen = [], set()
    skipped, scanned = 0, 0
    for ordinal, traj in enumerate(ds.as_numpy_iterator(), start=spec["skip_episodes"]):
        scanned += 1
        if scanned > spec["max_episodes"]:
            break
        actions = np.asarray(traj["action"], np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 31 or not np.isfinite(actions).all():
            continue
        texts, images = traj["task"]["language_instruction"], traj["observation"]["image_primary"]
        key = episode_key(name, actions, texts, images)
        if key in seen:
            continue
        seen.add(key)
        role = assign_partition(key, spec["seed"])
        if counts[role] >= quotas[role]:
            continue
        centres = interior_centres(len(actions), spec["past_offsets"], spec["per_episode"])
        decoded = {}
        for t in centres:
            text = texts[t].decode().strip()
            indices = [t + o for o in offsets]
            if not text or any(isinstance(images[j], bytes) and not images[j] for j in indices):
                skipped += 1
                continue
            for j in indices:
                if j not in decoded:
                    decoded[j] = decode_and_resize(
                        {"image_primary": tf.convert_to_tensor(images[j])},
                        resize_size=(224, 224), depth_resize_size={},
                    )["image_primary"].numpy()
            i = len(records)
            arrays["frames"][i] = np.stack([decoded[j] for j in indices])
            arrays["raw_actions"][i] = actions[t:t + 16]
            records.append({
                "dataset": name, "episode": ordinal, "episode_key": key, "partition": role,
                "timestep": int(t), "instruction": text, "episode_length": len(actions),
            })
            counts[role] += 1
            if counts[role] >= quotas[role]:
                break
        if len(records) and len(records) % 512 < spec["per_episode"]:
            print("SAMPLES", name, counts, "scanned", scanned, flush=True)
        if counts == quotas:
            break
    if counts != quotas:
        raise RuntimeError(f"{name}: quota not reached: {counts} != {quotas}, scanned={scanned}")
    for arr in arrays.values():
        arr.flush()
    builder = tfds.builder(name, data_dir=str(STORE / "datasets/open-x-embodiment"))
    split = "train" if "val" in builder.info.splits else "train[:95%]"
    write_json(root / "records.json", records)
    write_json(root / "complete.json", {
        "complete": True, "dataset": name, "counts": counts, "episodes_scanned_after_skip": scanned,
        "skip_episodes": spec["skip_episodes"], "split": split, "skipped_windows": skipped,
    })
    return str(root)


def combine(root, shards, endpoints, past_metadata, offsets, seed):
    import sys
    sys.path.insert(0, str(BASE / "evaluation"))
    from past_protocol import normalize_actions
    all_records = []
    for shard in shards:
        rows = json.loads((shard / "records.json").read_text())
        all_records.extend((shard, i, row) for i, row in enumerate(rows))
    roles_by_key = {}
    for _, _, row in all_records:
        key = (row["dataset"], row["episode_key"])
        if key in roles_by_key and roles_by_key[key] != row["partition"]:
            raise RuntimeError("episode split leakage")
        roles_by_key[key] = row["partition"]
    for dirname, roles in [("train_dev", ("train", "dev")), ("sealed_test", ("test",))]:
        out = root / dirname
        out.mkdir(exist_ok=False)
        selected = [(s, i, r) for s, i, r in all_records if r["partition"] in roles]
        n = len(selected)
        frame_array = np.lib.format.open_memmap(
            out / "frames.npy", mode="w+", dtype=np.uint8, shape=(n, len(offsets), 224, 224, 3),
        )
        raw = np.empty((n, 16, 7), np.float32)
        data = {s: (
            np.load(s / "frames.npy", mmap_mode="r"), np.load(s / "raw_actions.npy", mmap_mode="r"),
        ) for s in shards}
        rows = []
        for dest, (shard, source, row) in enumerate(selected):
            frame_array[dest] = data[shard][0][source]
            raw[dest] = data[shard][1][source]
            rows.append(row)
        frame_array.flush()
        np.save(out / "raw_actions.npy", raw)
        for name, values in [
            ("dataset", [r["dataset"] for r in rows]), ("episode", [r["episode"] for r in rows]),
            ("timestep", [r["timestep"] for r in rows]), ("episode_key", [r["episode_key"] for r in rows]),
            ("partition", [r["partition"] for r in rows]),
        ]:
            np.save(out / f"{name}.npy", np.asarray(values), allow_pickle=False)
        train = np.asarray([i for i, r in enumerate(rows) if r["partition"] == "train"], dtype=np.int64)
        dev = np.asarray([i for i, r in enumerate(rows) if r["partition"] == "dev"], dtype=np.int64)
        np.save(out / "train_indices.npy", train)
        np.save(out / "valid_indices.npy", dev)
        write_json(out / "instructions.json", [r["instruction"] for r in rows])
        for side in ("past", "future"):
            np.save(out / f"{side}_actions.npy", np.stack([
                normalize_actions(a, endpoints[side]["statistics"][r["dataset"]]["action"])
                for a, r in zip(raw, rows)
            ]))
        manifest = {
            "protocol": "whenami_stage3_frozen_endpoints_v1", "complete": False,
            "n": n, "split_seed": seed, "train_n": len(train), "valid_n": len(dev),
            "endpoints": endpoints, "past_metadata": past_metadata, "frame_offsets": offsets,
            "datasets": {name: {"n": sum(r["dataset"] == name for r in rows)} for name in endpoints["past"]["statistics"]
                         if any(r["dataset"] == name for r in rows)},
            "reference_source_sha256": source_fingerprint(),
            "reader_study": {"role": dirname, "episode_key_type": "sha256_full_actions_sampled_raw_frames_instruction"},
            "notes": [
                "Fresh fixed subset beyond prior pilot episode range; hash partitioned by episode contents.",
                "train/dev/test are physically separate at test boundary; sealed_test is never encoded/trained here.",
                "Frozen endpoint pretraining may have seen any of these episodes.",
                "Development set may be monitored; sealed test requires separately locked protocol.",
            ],
        }
        write_json(out / "samples_manifest.json", manifest)
        if dirname == "sealed_test":
            (out / "SEALED_DO_NOT_EVALUATE").write_text("Holdout raw data. No feature extraction or model evaluation before protocol lock.\n")
    write_json(root / "split_audit.json", {
        "complete": True, "episode_roles": [
            {"dataset": k[0], "episode_key": k[1], "partition": v} for k, v in roles_by_key.items()
        ],
        "train_dev": str(root / "train_dev"), "sealed_test": str(root / "sealed_test"),
        "source_shards": [str(s) for s in shards],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-dataset", type=int, default=8192)
    parser.add_argument("--dev-per-dataset", type=int, default=1024)
    parser.add_argument("--test-per-dataset", type=int, default=1024)
    parser.add_argument("--skip-episodes", type=int, default=2048)
    parser.add_argument("--max-episodes", type=int, default=30000)
    parser.add_argument("--per-episode", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--datasets", nargs="+", default=["fractal20220817_data", "bridge_orig"])
    args = parser.parse_args()
    if min(args.train_per_dataset, args.dev_per_dataset, args.test_per_dataset, args.per_episode) < 1:
        raise ValueError("positive quotas required")
    # Skip safely beyond earlier deterministic training-subset scans.
    old = json.loads((STORE / "outputs/stage3_pilot/cache_pilot_01/samples_manifest.json").read_text())
    if args.skip_episodes <= max(x["episodes_scanned"] for x in old["datasets"].values()):
        raise ValueError("new raw subset must start beyond all old pilot episodes")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from stage3.common import reference_imports
    reference_imports()
    from vla.past import PastSpec, checkpoint_spec
    endpoints = {s: endpoint_record(p) for s, p in [("past", DEFAULT_PAST), ("future", DEFAULT_FUTURE)]}
    ck = torch.load(DEFAULT_PAST, weights_only=True, mmap=True, map_location="cpu")
    stage, past_offsets = checkpoint_spec(ck)
    if stage != 2:
        raise ValueError("expected Past checkpoint")
    del ck
    spec = PastSpec.from_offsets(past_offsets)
    offsets = sorted(set(past_offsets) | {3, 7, 11, 15})
    jobs = [{
        "dataset": name, "shard": str(args.output / "shards" / name),
        "statistics": endpoints["past"]["statistics"][name],
        "offsets": offsets, "past_offsets": past_offsets,
        "quotas": {"train": args.train_per_dataset, "dev": args.dev_per_dataset, "test": args.test_per_dataset},
        "skip_episodes": args.skip_episodes, "max_episodes": args.max_episodes,
        "per_episode": args.per_episode, "seed": args.seed,
    } for name in args.datasets]
    write_json(args.output / "preparation_config.json", {
        "jobs": jobs, "source_sha256": sha256(Path(__file__)), "old_pilot_manifest_sha256": sha256(
            STORE / "outputs/stage3_pilot/cache_pilot_01/samples_manifest.json"),
    })
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=multiprocessing.get_context("spawn")) as pool:
        shards = [Path(s) for s in pool.map(worker, jobs)]
    combine(args.output, shards, endpoints, spec.metadata(), offsets, args.seed)
    print("RAW_STUDY_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
