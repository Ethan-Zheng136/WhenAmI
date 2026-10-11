"""Compare the one prespecified action-priority setting and missing WoG-A reference."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage3.common import write_json, sha256
from stage3.compare import paired_errors, paired_episode_interval, training_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = {
        "forward_original": "s3_forward_pilot_01_eval",
        "bidir_original": "s3_bidir_pilot_01_eval",
        "forward_actionprio": "s3_forward_actionprio_01_eval",
        "bidir_actionprio": "s3_bidir_actionprio_01_eval",
        "wog_stage2": "wog_stage2_pilot_01_eval",
    }
    reports = {k: json.loads((args.root / v / "results.json").read_text()) for k, v in names.items()}
    reference = reports["forward_original"]
    for report in reports.values():
        if not report["complete"]:
            raise ValueError("incomplete evaluation")
        for key in ("cache_manifest_sha256", "noise_seed"):
            if report[key] != reference[key]:
                raise ValueError(f"unmatched {key}")
    manifest = Path(reference["cache"]) / "manifest.json"
    if sha256(manifest) != reference["cache_manifest_sha256"]:
        raise ValueError("cache changed")
    cache = json.loads(manifest.read_text())
    training = {}
    for key, report in reports.items():
        if key == "wog_stage2":
            continue
        training[key] = training_summary(report)
        config = json.loads((Path(report["checkpoint"]).parents[1] / "config.json").read_text())
        training[key]["alignment_weight"] = config["train_config"]["alignment_weight"]
        training[key]["action_weight"] = config["train_config"]["action_weight"]
        expected_alignment = .1 if key.endswith("actionprio") else 1.
        if training[key]["alignment_weight"] != expected_alignment or training[key]["action_weight"] != 1.:
            raise ValueError("unexpected objective weights")
        for field in ("steps", "global_batch", "lr", "seed", "world_size"):
            if training[key][field] != training["forward_original"][field]:
                raise ValueError(f"unmatched training {field}")
    metrics, contrasts, controls = [], [], []
    for dataset in sorted(cache["datasets"]):
        stats = cache["endpoints"]["past"]["statistics"][dataset]["action"]
        arrays = {}
        for key, report in reports.items():
            variant = "wog_stage2" if key == "wog_stage2" else "predicted_future_causal"
            with np.load(args.root / names[key] / f"{dataset}_{variant}.npz") as archive:
                arrays[key] = dict(archive)
            row = next(x for x in report["results"] if x["dataset"] == dataset and x["variant"] == variant)
            metrics.append({"run": key, **row})
        for variant in ("true_past", "true_future", "mean_future"):
            with np.load(args.root / names["forward_original"] / f"{dataset}_{variant}.npz") as archive:
                arrays[variant] = dict(archive)
            row = next(x for x in reference["results"] if x["dataset"] == dataset and x["variant"] == variant)
            metrics.append({"run": variant, **row})
            for key in ("bidir_original", "forward_actionprio", "bidir_actionprio"):
                with np.load(args.root / names[key] / f"{dataset}_{variant}.npz") as archive:
                    other = dict(archive)
                paired_errors(arrays[variant], other, stats)  # IDs and targets must match exactly.
                maximum = float(np.max(np.abs(arrays[variant]["predictions"] - other["predictions"])))
                np.testing.assert_allclose(
                    arrays[variant]["predictions"], other["predictions"], rtol=1e-5, atol=1e-7,
                )
                controls.append({"dataset": dataset, "variant": variant, "run": key, "max_abs": maximum})
        for first, second in [
            ("forward_original", "forward_actionprio"),
            ("bidir_original", "bidir_actionprio"),
            ("forward_actionprio", "bidir_actionprio"),
            ("true_past", "bidir_actionprio"),
            ("wog_stage2", "bidir_actionprio"),
        ]:
            e1, e2 = paired_errors(arrays[first], arrays[second], stats)
            contrasts.append({
                "dataset": dataset, "first": first, "second": second,
                "second_minus_first": paired_episode_interval(e2 - e1, arrays[first]["episode"]),
                "second_relative_mse_change_percent": float(100 * (e2.mean() / e1.mean() - 1)),
            })
    write_json(args.output, {
        "complete": True, "cache_manifest_sha256": reference["cache_manifest_sha256"],
        "noise_seed": reference["noise_seed"], "training": training, "metrics": metrics,
        "contrasts": contrasts, "fixed_control_checks": controls,
        "notes": [
            "Exploratory follow-up on an already inspected validation set; not a new final test.",
            "One training/noise seed; episode-bootstrap intervals do not measure seed robustness.",
            "Only one alternative alignment weight (.1), final1000 checkpoints prespecified.",
            "Bidirectional adds computation, not compute-matched with forward.",
            "WoG-A is a released checkpoint with different training budget, not a matched-training baseline.",
            "Normalized action-block MSE over first6 dimensions is primary; gripper and first-action are secondary.",
            "Action-priority improvements alone do not establish the value of reverse supervision.",
        ],
    })
    for row in contrasts:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
