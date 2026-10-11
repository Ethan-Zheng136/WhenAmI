"""Version-pinned inventory of the project's ORIGINAL27-domain OXE mixture.

Enumerate TFDS metadata-selected shard paths, never broad glob matching multiple
physical layouts (notably dobbe). Weights reproduce balance_weights=True with
the original checkpoint's pinned transition counts, not equal domain sampling.
"""

import argparse
import ast
import json
from pathlib import Path

from stage3.common import STORE, DEFAULT_PAST, REFERENCE, write_json, sha256
from stage3_scale.inventory import load_roles, REGISTRY, balanced_assignment

MIXTURE = "oxe_magic_soup_plus"
VERSIONS = {
    "bridge_orig": "1.0.0", "fmb_dataset": "0.0.1", "dobbe": "0.0.1", "droid": "1.0.1",
}


def original_mixture():
    source = REFERENCE / "prismatic/vla/datasets/rlds/oxe/mixtures.py"
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == MIXTURE:
                    pairs = ast.literal_eval(value)
                    if len(pairs) != 27 or len({name for name, _ in pairs}) != 27:
                        raise ValueError("original27 mixture changed; review before training")
                    return pairs
    raise ValueError("cannot find original mixture")


def sampling_probabilities(pairs, statistics):
    sizes = {name: float(statistics[name]["num_transitions"]) for name, _ in pairs}
    if any(n <= 0 for n in sizes.values()) or any(w <= 0 for _, w in pairs):
        raise ValueError("positive transition counts and mixture multipliers required")
    denominator = sum(w * sizes[name] for name, w in pairs)
    return {name: w * sizes[name] / denominator for name, w in pairs}


def metadata_files(root, info, split_name="train"):
    split = next(s for s in info["splits"] if s["name"] == split_name)
    count = len(split["shardLengths"])
    return [root / f"{info['name']}-{split_name}.tfrecord-{i:05d}-of-{count:05d}" for i in range(count)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32, help="CPU preparation workers, not training cards")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("inventory output already exists")
    pairs = original_mixture()
    stat_path = DEFAULT_PAST.parents[1] / "dataset_statistics.json"
    stats = json.loads(stat_path.read_text())
    probabilities = sampling_probabilities(pairs, stats)
    pinned = load_roles(REGISTRY)
    rows, datasets = [], []
    for name, weight in pairs:
        version = VERSIONS.get(name, "0.1.0")
        root = STORE / "datasets/open-x-embodiment" / name / version
        metadata = root / "dataset_info.json"
        info = json.loads(metadata.read_text())
        split = next(s for s in info["splits"] if s["name"] == "train")
        files = metadata_files(root, info)
        for i, (path, episodes) in enumerate(zip(files, split["shardLengths"])):
            stat = path.stat()
            if stat.st_size <= 0:
                raise ValueError(f"empty source shard: {path}")
            rows.append({
                "id": f"{name}/{version}/train/{i:05d}", "dataset": name, "version": version,
                "source_split": "train", "path": str(path), "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "source_episodes": int(episodes),
                "tfds_name": info["name"],
            })
        datasets.append({
            "dataset": name, "version": version, "source_train_episodes": sum(map(int, split["shardLengths"])),
            "train_shards": len(files), "dataset_info_sha256": sha256(metadata),
            "mixture_multiplier": weight, "pinned_num_transitions": stats[name]["num_transitions"],
            "sampling_probability": probabilities[name],
            "extra_physical_train_files_ignored": len(list(root.glob("*-train.tfrecord-*-of-*"))) - len(files),
            "pinned_roles": {r: sum(d == name and v == r for (d, _), v in pinned.items()) for r in ("train", "dev", "test")},
        })
    assigned, loads = balanced_assignment(rows, args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "oxe27_source_inventory", "training_ready": False, "data_mix": MIXTURE,
        "datasets": datasets, "source_shards": assigned, "sampling_probabilities": probabilities,
        "sampling_rule": "legacy_mixture_multiplier_times_pinned_num_transitions",
        "prior_role_registry_sha256": sha256(REGISTRY), "split_seed": 20261002,
        "statistics_path": str(stat_path), "statistics_sha256": sha256(stat_path),
        "mixture_source_sha256": sha256(REFERENCE / "prismatic/vla/datasets/rlds/oxe/mixtures.py"),
        "preparation_workers_planned": args.workers, "source_episodes_per_worker": loads,
    }
    write_json(args.output / "inventory.json", result)
    print("OXE27_INVENTORY", len(datasets), "datasets", len(rows), "shards",
          sum(x["source_train_episodes"] for x in datasets), "source episodes", flush=True)


if __name__ == "__main__":
    main()
