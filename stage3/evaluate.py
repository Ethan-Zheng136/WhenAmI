"""Readout-level evaluation of Stage-3 held-out episodes.

predicted_future is causal at this evaluation interface (Past H/z plus current
Future-VLM z). It is NOT yet Stage4's integrated Past policy. Backward predictions
use real future evidence and are explicitly oracle diagnostics.
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import numpy as np
import torch

from stage3.common import BASE, write_json, sha256
from stage3.cache import EndpointCache
from stage3.connector import SharedConnector, ConnectorConfig
from stage3.behavior import load_behavior

sys.path.insert(0, str(BASE / "evaluation"))
from past_protocol import sample_seed, action_metrics


def denormalize(predictions, stats):
    x = np.clip(predictions, -1, 1)
    x[:, :, 6] = (x[:, :, 6] >= 0.5).astype(x.dtype)  # Exactly the endpoint predict_action convention.
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", [True] * 6 + [False]), dtype=bool)
    return np.where(mask, 0.5 * (x + 1) * (high - low) + low, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    args = ap.parse_args()
    if args.output.exists() or args.limit < 1:
        raise ValueError("choose a fresh output directory and positive limit")
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    meta = cp["metadata"]
    if meta["schema"] != "whenami_stage3_connector_v1":
        raise ValueError("not a Stage3 connector checkpoint")
    cache = EndpointCache(meta["cache"])
    if cache.manifest_sha256 != meta["cache_manifest_sha256"]:
        raise ValueError("cache does not match connector training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    connector = SharedConnector(ConnectorConfig(**meta["connector"]))
    connector.load_state_dict(cp["connector"], strict=True)
    connector = connector.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = np.load(cache.path / "raw_actions.npy", mmap_mode="r")
    summaries = []
    for side in ("future", "past"):
        reader = load_behavior(cache.manifest["endpoints"][side], device)
        predicted_name = "predicted_future_causal" if side == "future" else "predicted_past_oracle"
        variants = [f"true_{side}", f"mean_{side}"]
        if side == "future" or meta["mode"] == "bidirectional":
            variants.append(predicted_name)
        for name in sorted(set(cache.datasets.tolist())):
            indices = cache.valid_indices[cache.datasets[cache.valid_indices] == name][:args.limit]
            if not len(indices):
                continue
            predictions = {variant: [] for variant in variants}
            for i in indices:
                batch = cache.batch([i], device)
                with torch.inference_mode():
                    context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda" and args.precision == "bf16" else nullcontext()
                    )
                    with context:
                        choices = {
                            f"true_{side}": batch[f"{side}_c"],
                            f"mean_{side}": getattr(connector, f"{side}_mean")[None],
                        }
                        if predicted_name in variants:
                            source, direction = ("past", "forward") if side == "future" else ("future", "backward")
                            choices[predicted_name] = connector.predict(
                                batch[f"{source}_h"], batch[f"{source}_z"], direction
                            )
                        seed = sample_seed(name, cache.episodes[i], cache.timesteps[i], args.seed)
                        for variant, condition in choices.items():
                            action = reader.sample(batch[f"{side}_z"], condition, seed)
                            predictions[variant].append(action.float().cpu().numpy()[0])
            for variant in variants:
                p = denormalize(
                    np.stack(predictions[variant]),
                    cache.manifest["endpoints"][side]["statistics"][name]["action"],
                )
                target = np.array(raw[indices])
                metrics = action_metrics(
                    p, target, cache.manifest["endpoints"]["past"]["statistics"][name]["action"]
                )
                np.savez_compressed(
                    args.output / f"{name}_{variant}.npz", predictions=p, target=target,
                    indices=indices, episode=cache.episodes[indices], timestep=cache.timesteps[indices],
                )
                summaries.append({"dataset": name, "variant": variant, **metrics})
                print("RESULT", name, variant, metrics, flush=True)
        del reader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(args.output / "results.json", {
        "complete": True, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "cache": str(cache.path), "cache_manifest_sha256": cache.manifest_sha256,
        "noise_seed": args.seed, "precision": args.precision, "results": summaries,
        "notes": [
            "Stage3 validation episodes only; endpoints themselves were not trained on an audited disjoint corpus.",
            "predicted_future is a causal condition-translation path, not Stage4 integration.",
            "Backward/predicted_past uses real future evidence and is not deployable.",
            "Single noise seed, fixed small pilot subset, not comparable to the previous evaluation sample set.",
        ],
    })


if __name__ == "__main__":
    main()
