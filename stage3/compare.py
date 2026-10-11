"""Paired final-checkpoint comparison for the fixed Stage-3 pilot.

The primary contrast is bidirectional minus forward-only normalized action MSE,
on identical held-out episode windows, using identical sampling noise. Negative
is better for bidirectional. Episode-cluster intervals describe sampling
uncertainty conditional on this one training/noise seed, not training variance.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from stage3.common import BASE, write_json, sha256

sys.path.insert(0, str(BASE / "evaluation"))
from past_protocol import normalize_actions


def paired_episode_interval(delta, episode, draws=5000, seed=20261002):
    delta, episode = np.asarray(delta), np.asarray(episode)
    if delta.ndim != 1 or len(delta) != len(episode) or len(delta) == 0:
        raise ValueError("expected paired nonempty sample losses and episode IDs")
    if not np.isfinite(delta).all() or draws < 1:
        raise ValueError("finite losses and positive bootstrap count required")
    groups, inverse = np.unique(episode, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=delta)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(groups), size=(draws, len(groups)))
    values = sums[sample].sum(1) / counts[sample].sum(1)
    return {
        "mean": float(delta.mean()),
        "ci95_percentile": np.quantile(values, [.025, .975]).tolist(),
        "episodes": len(groups),
        "windows": len(delta),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def paired_errors(left, right, stats):
    for name in ("indices", "episode", "timestep", "target"):
        if not np.array_equal(left[name], right[name]):
            raise ValueError(f"comparison samples do not match: {name}")
    if left["predictions"].shape != right["predictions"].shape:
        raise ValueError("prediction shape mismatch")
    target = normalize_actions(left["target"], stats)[:, :, :6]
    errors = []
    for arrays in (left, right):
        pred = normalize_actions(arrays["predictions"], stats)[:, :, :6]
        errors.append(np.square(pred - target).mean((1, 2)))
    return errors


def training_summary(result):
    path = Path(result["checkpoint"]).parents[1]
    done = json.loads((path / "complete.json").read_text())
    config = json.loads((path / "config.json").read_text())
    if not done.get("complete") or not done.get("frozen_readers_unchanged"):
        raise ValueError(f"training did not complete safely: {path}")
    rows = [json.loads(line) for line in (path / "train.jsonl").open()]
    if len(rows) != done["steps"] or rows[-1]["step"] != done["steps"]:
        raise ValueError("training log does not cover the complete run")
    last = json.loads((path / "validation.jsonl").read_text().splitlines()[-1])
    return {
        "run": str(path), "steps": done["steps"], "mode": config["mode"],
        "global_batch": config["global_batch"], "world_size": config["world_size"],
        "lr": config["train_config"]["lr"], "seed": config["train_config"]["seed"],
        "last100_median_step_seconds": statistics.median(x["seconds"] for x in rows[-100:]),
        "summed_training_step_seconds": sum(x["seconds"] for x in rows),
        "final_validation": last,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", type=Path, required=True, help="forward evaluate output directory")
    ap.add_argument("--bidirectional", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    f = json.loads((args.forward / "results.json").read_text())
    b = json.loads((args.bidirectional / "results.json").read_text())
    for key in ("cache_manifest_sha256", "noise_seed", "precision"):
        if f[key] != b[key]:
            raise ValueError(f"evaluation settings differ: {key}")
    if not f.get("complete") or not b.get("complete"):
        raise ValueError("both evaluations must be complete")
    fs, bs = training_summary(f), training_summary(b)
    if fs["mode"] != "forward" or bs["mode"] != "bidirectional":
        raise ValueError("incorrect treatment/control modes")
    for key in ("steps", "global_batch", "world_size", "lr", "seed"):
        if fs[key] != bs[key]:
            raise ValueError(f"unmatched training control: {key}")
    cache_manifest = Path(f["cache"]) / "manifest.json"
    if sha256(cache_manifest) != f["cache_manifest_sha256"]:
        raise ValueError("sample cache changed")
    cache = json.loads(cache_manifest.read_text())
    rows = []
    for dataset in sorted(cache["datasets"]):
        forward_file = args.forward / f"{dataset}_predicted_future_causal.npz"
        bidir_file = args.bidirectional / f"{dataset}_predicted_future_causal.npz"
        with np.load(forward_file, allow_pickle=False) as fp, np.load(bidir_file, allow_pickle=False) as bp:
            stats = cache["endpoints"]["past"]["statistics"][dataset]["action"]
            f_error, b_error = paired_errors(fp, bp, stats)
            interval = paired_episode_interval(b_error - f_error, fp["episode"])
            row = {
                "dataset": dataset,
                "forward": next(x for x in f["results"] if x["dataset"] == dataset
                                and x["variant"] == "predicted_future_causal"),
                "bidirectional": next(x for x in b["results"] if x["dataset"] == dataset
                                      and x["variant"] == "predicted_future_causal"),
                "bidir_minus_forward_normalized_mse": interval,
                "bidir_mse_reduction_percent": float(100 * (1 - b_error.mean() / f_error.mean())),
                "references": [x for x in b["results"] if x["dataset"] == dataset
                               and x["variant"] in ("true_future", "true_past", "mean_future")],
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    write_json(args.output, {
        "complete": True, "primary_contrast": "final-checkpoint predicted_future_causal: bidir minus forward",
        "forward_evaluation": str(args.forward), "bidirectional_evaluation": str(args.bidirectional),
        "training": {"forward": fs, "bidirectional": bs},
        "cache_manifest_sha256": f["cache_manifest_sha256"],
        "noise_seed": f["noise_seed"], "precision": f["precision"], "results": rows,
        "notes": [
            "Matched forward update count/minibatches/noise; reverse compute is extra, not compute-matched.",
            "CIs resample whole episodes and retain window-weighted MSE.",
            "One training seed and one sampling seed; CIs do not quantify optimizer/seed variance.",
            "Validation was monitored during training, not an untouched final test set.",
            "Frozen endpoint pretraining may have seen these episodes.",
            "This is condition translation using a fixed Future reader, not Stage4 online integration.",
        ],
    })


if __name__ == "__main__":
    main()
