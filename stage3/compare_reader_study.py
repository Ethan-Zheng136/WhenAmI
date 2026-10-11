"""Across-seed reader factorial summary on DEVELOPMENT only."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage3.common import write_json
from stage3.compare import paired_errors, paired_episode_interval
from stage3.reader_study_sweep import SEEDS, VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports, configs = {}, {}
    for seed in SEEDS:
        for variant in VARIANTS:
            root = args.runs / f"{variant}_seed{seed}"
            config = json.loads((root / "config.json").read_text())
            done = json.loads((root / "complete.json").read_text())
            report = json.loads((root / "development_eval/results.json").read_text())
            if not done["complete"] or not done["original_reader_unchanged"] or done["sealed_test_accessed"]:
                raise ValueError("unsafe/incomplete experiment")
            if not report["complete"] or report["split"] != "development" or report["limit_per_dataset"] != 0:
                raise ValueError("expected complete unrestricted development evaluation")
            settings = config["settings"]
            if settings["seed"] != seed or settings["adaptive"] != variant.startswith("adaptive"):
                raise ValueError("factorial assignment mismatch")
            configs[(seed, variant)] = config
            reports[(seed, variant)] = report
    first = configs[(SEEDS[0], VARIANTS[0])]
    cache = Path(first["cache"])
    manifest = json.loads((cache / "manifest.json").read_text())
    for key, config in configs.items():
        for field in ("max_steps", "batch_size", "lr", "adapter_rank", "precision"):
            if config["settings"][field] != first["settings"][field]:
                raise ValueError(f"unmatched budget: {field}")
        if config["cache_manifest_sha256"] != first["cache_manifest_sha256"]:
            raise ValueError("different cache")
        if reports[key]["noise_seed"] != 42:
            raise ValueError("unmatched action sampling")
    summaries, paired = [], []
    for dataset in sorted(manifest["datasets"]):
        stats = manifest["endpoints"]["past"]["statistics"][dataset]["action"]
        for variant in VARIANTS:
            values = [
                next(x for x in reports[(seed, variant)]["results"]
                     if x["dataset"] == dataset and x["variant"] == "evidence")["normalized_mse_6"]
                for seed in SEEDS
            ]
            summaries.append({
                "dataset": dataset, "variant": variant, "seeds": list(SEEDS),
                "mse_per_seed": values, "mse_mean": float(np.mean(values)),
                "mse_std_across_seeds": float(np.std(values, ddof=1)),
            })
        all_effects = []
        for seed in SEEDS:
            arrays = {}
            for variant in VARIANTS:
                path = args.runs / f"{variant}_seed{seed}" / "development_eval" / f"{dataset}_evidence.npz"
                with np.load(path) as f:
                    arrays[variant] = dict(f)
            errors = {}
            baseline = arrays["frozen_mean"]
            for variant in VARIANTS:
                _, e = paired_errors(baseline, arrays[variant], stats)
                errors[variant] = e
            effects = {
                "future_effect_frozen": errors["frozen_future"] - errors["frozen_mean"],
                "future_effect_adaptive": errors["adaptive_future"] - errors["adaptive_mean"],
            }
            effects["interaction_difference_of_differences"] = effects["future_effect_adaptive"] - effects["future_effect_frozen"]
            all_effects.append(effects)
            for name, delta in effects.items():
                paired.append({
                    "dataset": dataset, "seed": seed, "effect": name,
                    "paired_episode_interval": paired_episode_interval(delta, baseline["episode"]),
                })
        for effect in all_effects[0]:
            values = [float(x[effect].mean()) for x in all_effects]
            paired.append({
                "dataset": dataset, "effect": effect, "seed_effects": values,
                "mean_across_seeds": float(np.mean(values)),
                "std_across_seeds": float(np.std(values, ddof=1)),
                "negative_seed_count": sum(x < 0 for x in values),
                "note": "negative interaction suggests adaptation strengthens the incremental effect of Future; not proof alone",
            })
    write_json(args.output, {
        "complete": True, "split": "development", "sealed_test_accessed": False,
        "summaries": summaries, "paired_effects": paired,
        "notes": [
            "Four training seeds; episode intervals per seed do not incorporate training-seed variance.",
            "Real Future is oracle evidence, not deployable.",
            "Primary question is Future increment within adaptive/frozen columns, not absolute fine-tuning gains.",
            "No test-set evaluation; final holdout protocol must be locked separately.",
        ],
    })
    print("READER_FACTORIAL_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
