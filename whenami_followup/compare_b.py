"""Recompute paired B10k/B20k with the common sampling seed42, no GPU."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from whenami_followup.metrics import score, aggregate, paired

STORE = Path("/mnt/pfs/zhengguantian/WoG_v2")


def load(main, step, manifest, statistics):
    directory = main / "development" / f"step-{step:06d}"
    report = json.loads((directory / "RESULT.json").read_text())
    if not report["complete"] or not report["causal_student_only"]:
        raise ValueError("not complete causal evaluation")
    seed_index = report["noise_seeds"].index(42)
    found, labels = {}, {}
    for path in sorted(directory.glob("rank-*.npz")):
        with np.load(path) as data:
            for j, index in enumerate(data["indices"]):
                sample = manifest["clips"][int(index)]
                key = sample["dataset"], sample["episode_key"]
                if key in found:
                    raise ValueError("duplicate episode")
                prediction = data["predictions"][j, seed_index:seed_index + 1]
                target, mask = data["targets"][j], data["masks"][j]
                scores = score(prediction, target, mask, statistics[key[0]]["action"])
                found[key] = {"dataset": key[0], "episode_key": key[1],
                              **{k: float(v[0]) for k, v in scores.items()}}
                labels[key] = target.copy(), mask.copy()
    if len(found) != report["episodes"]:
        raise ValueError("missing episodes")
    return found, labels, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((STORE / "outputs/whenami_timeline_a/dev27_16ep_01/MANIFEST.json").read_text())
    statistics = json.loads((STORE / "outputs/stage1/baseline_v2--image_aug/dataset_statistics.json").read_text())
    result = {"utc": datetime.now(timezone.utc).isoformat(), "common_sampling_seed": 42,
              "scope": "Same episodes, labels/masks, normalization and one common action-sampling seed; not closed-loop prediction.",
              "models": {}}
    for variant in ("action_only", "condition_align"):
        root = STORE / "outputs/whenami_timeline_b/training" / f"timeline_b_{variant}_A10k_320_20k_01"
        main_root = root / "main"
        old, old_labels, old_report = load(main_root, 10000, manifest, statistics)
        new, new_labels, new_report = load(main_root, 20000, manifest, statistics)
        keys = sorted(old)
        if keys != sorted(new):
            raise ValueError("unpaired cohort")
        for key in keys:
            if not all(np.array_equal(a, b) for a, b in zip(old_labels[key], new_labels[key])):
                raise ValueError("target/mask drift")
        comparisons = {
            k: paired([old[key][k] for key in keys], [new[key][k] for key in keys],
                      [key[0] for key in keys], larger=k == "gripper_accuracy")
            for k in ("mse6", "unclipped_mse6", "first_action_mse6", "gripper_accuracy")
        }
        # Compare a fixed 1000-update window, not a single minibatch.
        training = {10000: [], 20000: []}
        for line in (main_root / "train.jsonl").read_text().splitlines():
            row = json.loads(line)
            for end in training:
                if end - 1000 < row["step"] <= end:
                    training[end].append(row)
        means = {str(end): {k: float(np.mean([r[k] for r in rows]))
                           for k in ("action_fm", "condition_cosine", "weighted_alignment")}
                 for end, rows in training.items()}
        result["models"][variant] = {
            "comparison": comparisons, "at10k": aggregate(list(old.values())),
            "at20k_seed42": aggregate(list(new.values())),
            "train_last1000_means": means,
            "source_sampling_seeds": {"10000": old_report["noise_seeds"], "20000": new_report["noise_seeds"]},
        }
        print(variant, json.dumps(result["models"][variant]["comparison"]), flush=True)
        print("TRAIN", variant, json.dumps(means), flush=True)
        for d in ("bridge_orig", "fractal20220817_data"):
            print("DOMAIN", variant, d,
                  result["models"][variant]["at10k"]["domains"][d],
                  result["models"][variant]["at20k_seed42"]["domains"][d], flush=True)
    result["complete"] = True
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
