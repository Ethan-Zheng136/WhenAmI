"""Matched-window diagnostics for R+K and its four prespecified controls."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage3.common import sha256, write_json
from stage3.compare import paired_errors, paired_episode_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-latent", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ["rk_adapter_01", "rk_predict_01", "rk_teacher_01", "rk_teacher_roll4_01"]
    if args.include_latent:
        names.append("rk_latent_01")
    reports = {name: json.loads((args.root / f"{name}_eval/results.json").read_text()) for name in names}
    reference = reports[names[0]]
    manifest_file = Path(reference["cache"]) / "manifest.json"
    if sha256(manifest_file) != reference["cache_manifest_sha256"]:
        raise ValueError("cache changed")
    cache = json.loads(manifest_file.read_text())
    training = {}
    for name, report in reports.items():
        if not report["complete"]:
            raise ValueError("incomplete evaluation")
        for field in ("cache_manifest_sha256", "noise_seed", "precision"):
            if report[field] != reference[field]:
                raise ValueError("unmatched evaluation")
        config = json.loads((args.root / name / "config.json").read_text())
        done = json.loads((args.root / name / "complete.json").read_text())
        rows = [json.loads(x) for x in (args.root / name / "train.jsonl").read_text().splitlines()]
        if not done["complete"] or not done["frozen_reader_unchanged"] or len(rows) != done["steps"]:
            raise ValueError("incomplete/unsafe training")
        training[name] = {
            "mode": config["mode"], "config": config["train_config"],
            "last100_median_step_seconds": float(np.median([x["seconds"] for x in rows[-100:]])),
            "gradient_audits": [json.loads(x) for x in (args.root / name / "gradient_audits.jsonl").read_text().splitlines()],
        }
        for field in ("max_steps", "batch_size", "lr", "seed", "fm_draws", "precision"):
            if training[name]["config"][field] != training[names[0]]["config"][field]:
                raise ValueError(f"unmatched {field}")
        expected_alignment = 0. if name == "rk_latent_01" else .1
        if training[name]["config"]["alignment_weight"] != expected_alignment:
            raise ValueError("unexpected alignment weight")
    metrics, contrasts, controls = [], [], []
    for dataset in sorted(cache["datasets"]):
        arrays = {}
        for name in names:
            for row in reports[name]["results"]:
                if row["dataset"] != dataset:
                    continue
                key = (name, row["variant"])
                with np.load(args.root / f"{name}_eval" / f"{dataset}_{row['variant']}.npz") as archive:
                    arrays[key] = dict(archive)
                metrics.append({"run": name, **row})
        stats = cache["endpoints"]["past"]["statistics"][dataset]["action"]
        baseline = arrays[(names[0], "true_past")]
        for name in names:
            other = arrays[(name, "true_past")]
            paired_errors(baseline, other, stats)
            error = float(np.max(np.abs(baseline["predictions"] - other["predictions"])))
            if error != 0:
                raise ValueError("fixed Past readout changed between evaluations")
            controls.append({"dataset": dataset, "run": name, "past_reference_max_abs": error})
        pairs = [
            ((names[0], "online"), (names[1], "online"), "forecast_vs_plain_adapter"),
            ((names[1], "online"), (names[2], "online"), "oracle_training_increment"),
            ((names[2], "online"), (names[3], "online"), "four_step_sampling_loss_increment"),
            ((names[0], "online"), (names[2], "oracle_future"), "oracle_vs_plain_adapter"),
        ]
        if args.include_latent:
            pairs.append((("rk_latent_01", "online"), (names[1], "online"),
                          "future_supervision_vs_equal_parameter_latent"))
        pairs += [((names[0], "true_past"), (name, "online"), "online_vs_original_past") for name in names]
        pairs += [
            ((name, "mean_future_intervention"), (name, "online"), "predicted_vs_mean_input_intervention")
            for name in names[1:]
        ]
        for left, right, meaning in pairs:
            e1, e2 = paired_errors(arrays[left], arrays[right], stats)
            contrasts.append({
                "dataset": dataset, "meaning": meaning, "left": list(left), "right": list(right),
                "right_minus_left": paired_episode_interval(e2 - e1, baseline["episode"]),
                "relative_mse_change_percent": float(100 * (e2.mean() / e1.mean() - 1)),
            })
    write_json(args.output, {
        "complete": True, "cache_manifest_sha256": reference["cache_manifest_sha256"],
        "noise_seed": reference["noise_seed"], "training": training,
        "metrics": metrics, "contrasts": contrasts, "fixed_control_checks": controls,
        "notes": [
            "One seed on previously inspected validation episodes: exploratory, not final test.",
            "Intervals resample episodes; they do not quantify training/noise seed variance.",
            "Teacher/rollout add computation; action FM in teacher averages online and oracle.",
            "Optional rk_latent_01 has equal trainable R+K capacity but no per-sample Future supervision.",
            "Oracle outputs cannot be deployed; online has only Past inputs.",
            "Frozen Past baseline must be identical across runs.",
            "Action-block first6-dimensional normalized MSE is primary; first action/gripper retained too.",
            "4-step training unroll does not change the common 9-interval evaluation sampler.",
        ],
    })
    for row in contrasts:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
