"""Paired oracle-only evidence comparisons; never rank these as online policies."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage3.common import write_json, sha256
from stage3.compare import paired_errors, paired_episode_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--refinement-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = ["condition_oracle_01", "mean_h_01", "projected_h_01", "hidden_h_01", "static_h_01"]
    reports = {name: json.loads((args.root / name / "results.json").read_text()) for name in names}
    meta = reports[names[0]]["metadata"]
    manifest_file = Path(meta["cache"]) / "manifest.json"
    if sha256(manifest_file) != meta["cache_manifest_sha256"]:
        raise ValueError("cache changed")
    cache = json.loads(manifest_file.read_text())
    for name, report in reports.items():
        if not report["complete"] or not report["frozen_reader_unchanged"]:
            raise ValueError("incomplete training/evaluation")
        for field in ("cache_manifest_sha256", "steps", "batch_size", "lr", "seed"):
            if report["metadata"][field] != meta[field]:
                raise ValueError(f"unmatched {field}")
        rows = [json.loads(x) for x in (args.root / name / "train.jsonl").read_text().splitlines()]
        if [r["step"] for r in rows] != list(range(1, meta["steps"] + 1)):
            raise ValueError("incomplete training log")
    h_counts = {reports[name]["metadata"]["trainable_parameters"] for name in names[1:]}
    if len(h_counts) != 1:
        raise ValueError("H-interface parameter counts differ")
    previous = {
        "plain_k": ("rk_adapter_01_eval", "online"),
        "mixed_oracle_k": ("rk_teacher_01_eval", "oracle_future"),
    }
    metrics, contrasts, controls = [], [], []
    for dataset in sorted(cache["datasets"]):
        stats = cache["endpoints"]["past"]["statistics"][dataset]["action"]
        arrays = {}
        baseline = None
        for name, report in reports.items():
            for variant in ("evidence", "mean_intervention", "true_past"):
                with np.load(args.root / name / f"{dataset}_{variant}.npz") as f:
                    arrays[(name, variant)] = dict(f)
            if baseline is None:
                baseline = arrays[(name, "true_past")]
            other = arrays[(name, "true_past")]
            paired_errors(baseline, other, stats)
            if not np.array_equal(baseline["predictions"], other["predictions"]):
                raise ValueError("fixed Past reference changed")
            controls.append({"dataset": dataset, "run": name, "past_reference_identical": True})
            for row in report["results"]:
                if row["dataset"] == dataset:
                    metrics.append({"run": name, **row})
        for alias, (directory, variant) in previous.items():
            report = json.loads((args.refinement_root / directory / "results.json").read_text())
            if report["cache_manifest_sha256"] != meta["cache_manifest_sha256"] or report["noise_seed"] != meta["seed"]:
                raise ValueError("old refinement comparison uses different samples/noise")
            with np.load(args.refinement_root / directory / f"{dataset}_{variant}.npz") as f:
                arrays[(alias, "evidence")] = dict(f)
            row = next(r for r in report["results"] if r["dataset"] == dataset and r["variant"] == variant)
            metrics.append({"run": alias, **row})
        pairs = [
            ("plain_k", "condition_oracle_01", "pure_true_C_vs_plain_K"),
            ("mixed_oracle_k", "condition_oracle_01", "pure_oracle_vs_mixed_student_oracle"),
            ("mean_h_01", "projected_h_01", "C_information_in_H_interface"),
            ("mean_h_01", "hidden_h_01", "full_H_vs_mean"),
            ("mean_h_01", "static_h_01", "extra_current_encoder_vs_mean"),
            ("projected_h_01", "hidden_h_01", "full_H_vs_C_lift_same_architecture"),
            ("static_h_01", "hidden_h_01", "real_future_vs_repeated_current"),
        ]
        for left, right, meaning in pairs:
            e1, e2 = paired_errors(arrays[(left, "evidence")], arrays[(right, "evidence")], stats)
            contrasts.append({
                "dataset": dataset, "left": left, "right": right, "meaning": meaning,
                "right_minus_left": paired_episode_interval(e2 - e1, baseline["episode"]),
                "relative_mse_change_percent": float(100 * (e2.mean() / e1.mean() - 1)),
            })
    write_json(args.output, {
        "complete": True, "cache_manifest_sha256": meta["cache_manifest_sha256"],
        "seed": meta["seed"], "probe_metadata": {n: r["metadata"] for n, r in reports.items()},
        "metrics": metrics, "contrasts": contrasts, "fixed_control_checks": controls,
        "notes": [
            "Oracle-only diagnostics, not R forecast evaluation or deployable policy results.",
            "H-interface probes match architecture/parameters and training statistics.",
            "C-lift preserves C information but changes rank/covariance relative to H; optimization effects remain possible.",
            "Static evidence is re-encoded from current only; a separately trained control, not merely an input swap.",
            "One seed on inspected validation; bootstrap excludes training/noise seed variance.",
        ],
    })
    for row in contrasts:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
