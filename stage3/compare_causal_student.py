"""Budget/initialization-matched causal-student comparison, development only.

Three inference noises are averaged WITHIN each training seed before episode
bootstrap; they are not treated as three independent trained models.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from stage3.causal_student import VARIANTS, SEEDS
from stage3.common import sha256, write_json
from stage3.compare import paired_errors, paired_episode_interval

METRICS = ("normalized_mse_6", "normalized_first_action_mse_6",
           "normalized_mae_6", "r_6", "r2_6", "gripper_accuracy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    configs, reports = {}, {}
    first = None
    for seed in SEEDS:
        for variant in VARIANTS:
            root = args.runs / f"{variant}_seed{seed}"
            config = json.loads((root / "config.json").read_text())
            done = json.loads((root / "complete.json").read_text())
            if not done["complete"] or not done["causal_student"] or not done["teacher_and_original_reader_unchanged"] or done["sealed_test_accessed"]:
                raise ValueError("incomplete/unsafe run")
            settings = config["settings"]
            if settings["seed"] != seed or settings["variant"] != variant:
                raise ValueError("wrong factorial assignment")
            if first is None:
                first = config
            for name in ("max_steps", "batch_size", "fm_draws", "lr", "warmup", "lr_floor_ratio",
                         "alignment_weight", "distill_weight", "precision", "eval_noise_seeds"):
                if settings[name] != first["settings"][name]:
                    raise ValueError(f"unmatched setting {name}")
            if config["cache_manifest_sha256"] != first["cache_manifest_sha256"] or config["trainable_parameters"] != first["trainable_parameters"]:
                raise ValueError("unmatched cache/capacity")
            if variant != "latent" and config["base"] != configs[(seed, "latent")]["base"]:
                raise ValueError("student base initialization differs within seed")
            rows = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
            if [x["step"] for x in rows] != list(range(1, done["steps"] + 1)):
                raise ValueError("training log incomplete")
            if done["steps"] != settings["max_steps"]:
                raise ValueError("not final prescribed step")
            configs[(seed, variant)] = config
            cp_hash = sha256(root / "checkpoints" / f"step-{settings['max_steps']:06d}.pt")
            for noise in settings["eval_noise_seeds"]:
                report = json.loads((root / f"development_eval_noise{noise}/results.json").read_text())
                if not report["complete"] or not report["causal"] or report["split"] != "development":
                    raise ValueError("invalid eval status/split")
                if report["checkpoint_sha256"] != cp_hash or report["cache_manifest_sha256"] != config["cache_manifest_sha256"]:
                    raise ValueError("evaluation provenance mismatch")
                if report["noise_seed"] != noise or report["limit_per_dataset"] != 0:
                    raise ValueError("evaluation noise/sample subset mismatch")
                reports[(seed, variant, noise)] = report
    manifest_path = Path(first["settings"]["cache"]) / "manifest.json"
    if sha256(manifest_path) != first["cache_manifest_sha256"]:
        raise ValueError("cache changed")
    manifest = json.loads(manifest_path.read_text())
    noises = first["settings"]["eval_noise_seeds"]
    summaries, contrasts = [], []
    pairs = [
        ("latent", "forecast", "future_alignment_increment"),
        ("forecast", "distill_past", "generic_teacher_increment"),
        ("forecast", "distill_future", "future_teacher_increment"),
        ("distill_past", "distill_future", "privileged_vs_nonprivileged_teacher"),
        ("latent", "distill_future", "complete_student_vs_equal_capacity_control"),
    ]
    for dataset in sorted(manifest["datasets"]):
        stats = manifest["endpoints"]["past"]["statistics"][dataset]["action"]
        errors = {}
        episodes = None
        common_samples = None
        for variant in VARIANTS:
            per_seed = []
            for seed in SEEDS:
                measurements, noise_errors = [], []
                reference = None
                for noise in noises:
                    record = next(x for x in reports[(seed, variant, noise)]["results"] if x["dataset"] == dataset)
                    measurements.append(record)
                    path = args.runs / f"{variant}_seed{seed}" / f"development_eval_noise{noise}/{dataset}_online.npz"
                    with np.load(path) as archive:
                        arrays = dict(archive)
                    if common_samples is None:
                        common_samples = {k: arrays[k].copy() for k in ("indices", "episode", "timestep", "target")}
                    if any(not np.array_equal(arrays[k], v) for k, v in common_samples.items()):
                        raise ValueError("cross-model/noise sample indices, timesteps or labels differ")
                    if reference is None:
                        reference = arrays
                    _, err = paired_errors(reference, arrays, stats)
                    noise_errors.append(err)
                    if episodes is None:
                        episodes = arrays["episode"]
                    elif not np.array_equal(episodes, arrays["episode"]):
                        raise ValueError("different evaluation episode order")
                errors[(seed, variant)] = np.mean(noise_errors, axis=0)
                per_seed.append({
                    "seed": seed, **{k: float(np.mean([r[k] for r in measurements])) for k in METRICS},
                })
            summaries.append({
                "dataset": dataset, "variant": variant, "per_train_seed_noise_averaged": per_seed,
                "mean": {k: float(np.mean([r[k] for r in per_seed])) for k in METRICS},
                "std_across_train_seeds": {k: float(np.std([r[k] for r in per_seed], ddof=1)) for k in METRICS},
            })
        for left, right, meaning in pairs:
            seed_effects, relative = [], []
            for seed in SEEDS:
                delta = errors[(seed, right)] - errors[(seed, left)]
                seed_effects.append(float(delta.mean()))
                change = float(100 * (errors[(seed, right)].mean() / errors[(seed, left)].mean() - 1))
                relative.append(change)
                contrasts.append({
                    "dataset": dataset, "meaning": meaning, "left": left, "right": right,
                    "train_seed": seed, "noise_averaged": True,
                    "right_minus_left": paired_episode_interval(delta, episodes),
                    "relative_mse_change_percent": change,
                })
            contrasts.append({
                "dataset": dataset, "meaning": meaning, "left": left, "right": right,
                "seed_effects": seed_effects, "negative_seed_count": sum(x < 0 for x in seed_effects),
                "mean_effect": float(np.mean(seed_effects)), "std_effect": float(np.std(seed_effects, ddof=1)),
                "mean_relative_mse_change_percent": float(np.mean(relative)),
            })
    write_json(args.output, {
        "complete": True, "split": "development", "causal": True, "sealed_test_accessed": False,
        "training_seeds": list(SEEDS), "sampling_seeds": noises,
        "summaries": summaries, "contrasts": contrasts,
        "notes": [
            "Noise replicates averaged within trained seed before paired episode bootstrap.",
            "Per-seed intervals do not integrate training-seed variance; seed spread reported separately.",
            "All students same trainable capacity, warm-start, updates, observation batches and primary FM draws.",
            "Distillation adds teacher computation; not fully compute-matched.",
            "No held-out sealed-test evaluation; no oracle features enter student inference.",
        ],
    })
    print("CAUSAL_COMPARISON_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
