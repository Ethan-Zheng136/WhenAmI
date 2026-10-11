"""Plan TFRecord-level work without decoding trajectories or touching holdouts.

The output describes source coverage and work ownership, NOT extracted features.
The future extractor must standardize each episode identically, compute the
content key, and apply role_for_key before emitting any training window.
"""

import argparse
import hashlib
import json
from pathlib import Path

DATA = Path("/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment")
REGISTRY = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_reader_study/data_01/split_audit.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_roles(path):
    audit = json.loads(Path(path).read_text())
    if not audit.get("complete"):
        raise ValueError("Reader Study episode registry incomplete")
    roles = {}
    for row in audit["episode_roles"]:
        key = (row["dataset"], row["episode_key"])
        if row["partition"] not in ("train", "dev", "test"):
            raise ValueError("unknown episode role")
        previous = roles.setdefault(key, row["partition"])
        if previous != row["partition"]:
            raise ValueError("conflicting role for an episode")
    return roles


def role_for_key(dataset, key, pinned, split_seed=20261002):
    """Same hash split as ReaderStudy; existing identities are authoritative."""
    if (dataset, key) in pinned:
        return pinned[(dataset, key)]
    bucket = int.from_bytes(
        hashlib.sha256(f"{split_seed}:{key}".encode()).digest()[:8], "little"
    ) % 10000
    return "train" if bucket < 8000 else ("dev" if bucket < 9000 else "test")


def balanced_assignment(shards, workers):
    """Deterministic greedy episode-count balancing; no duplicate ownership."""
    if workers < 1:
        raise ValueError("positive extraction worker count required")
    load = [0] * workers
    out = [dict(row) for row in shards]
    for i in sorted(range(len(out)), key=lambda i: (-out[i]["source_episodes"], out[i]["id"])):
        rank = min(range(workers), key=lambda j: (load[j], j))
        out[i]["extraction_worker"] = rank
        load[rank] += out[i]["source_episodes"]
    return out, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=320)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--role-registry", type=Path, default=REGISTRY)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new planning directory")
    pinned = load_roles(args.role_registry)
    rows, summaries = [], []
    # Lock known versions: never silently select a newly installed dataset.
    for dataset, version in (("fractal20220817_data", "0.1.0"), ("bridge_orig", "1.0.0")):
        root = args.data_root / dataset / version
        info_path = root / "dataset_info.json"
        info = json.loads(info_path.read_text())
        split = next(s for s in info["splits"] if s["name"] == "train")
        lengths = [int(x) for x in split["shardLengths"]]
        files = sorted(root.glob("*-train.tfrecord-*-of-*"))
        if len(files) != len(lengths):
            raise ValueError(f"{dataset}: missing/extra TFRecord files")
        total_bytes = 0
        for i, (path, episodes) in enumerate(zip(files, lengths)):
            expected_suffix = f"-{i:05d}-of-{len(lengths):05d}"
            if not path.name.endswith(expected_suffix):
                raise ValueError(f"shard numbering mismatch: {path}")
            stat = path.stat()
            if stat.st_size <= 0:
                raise ValueError(f"empty source shard: {path}")
            total_bytes += stat.st_size
            rows.append({
                "id": f"{dataset}/{version}/train/{i:05d}",
                "dataset": dataset, "version": version, "source_split": "train",
                "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "source_episodes": episodes,
            })
        summaries.append({
            "dataset": dataset, "version": version, "train_shards": len(files),
            "source_train_episodes": sum(lengths), "source_file_bytes": total_bytes,
            "dataset_info_sha256": digest(info_path),
            "pinned_roles": {
                role: sum(d == dataset and r == role for (d, _), r in pinned.items())
                for role in ("train", "dev", "test")
            },
        })
    assigned, loads = balanced_assignment(rows, args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    # Copy role identities, not held-out images/features.
    registry = args.output / "pinned_episode_roles.json"
    registry.write_text(json.dumps([
        {"dataset": d, "episode_key": key, "partition": role}
        for (d, key), role in sorted(pinned.items())
    ], indent=2))
    plan = {
        "status": "source_inventory_only",
        "training_ready": False,
        "datasets": summaries,
        "extraction_workers_planned": args.workers,
        "source_shards": assigned,
        "source_episodes_per_worker": loads,
        "split_seed": 20261002,
        "prior_role_registry_sha256": digest(args.role_registry),
        "pinned_episode_roles_sha256": digest(registry),
        "source_sha256": digest(__file__),
        "requirements": [
            "Decode standardized episodes and compute the exact ReaderStudy content key before role filtering.",
            "Only train-role episodes may emit train features; keep prior dev/test episodes held out.",
            "Full source train scan is not permission to train on every source episode.",
            "Do not use old small-cache scaler/checkpoint contracts silently on expanded data.",
            "Extraction outputs need immutable per-shard completion/hash records and audited sample coverage.",
            "Main trainer needs true distributed minibatch sampling and per-rank resume state.",
            "No sealed-test images/features opened while building this inventory.",
        ],
    }
    (args.output / "inventory.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps({
        "status": plan["status"], "training_ready": False, "datasets": summaries,
        "source_shards": len(rows), "planned_workers": args.workers,
        "min_source_episodes_per_worker": min(loads), "max_source_episodes_per_worker": max(loads),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
