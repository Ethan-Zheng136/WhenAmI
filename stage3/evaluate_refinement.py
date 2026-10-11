"""Actual 9-interval action sampling for R+K, always using the frozen Past reader."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from stage3.behavior import load_behavior
from stage3.cache import EndpointCache
from stage3.common import BASE, sha256, write_json
from stage3.evaluate import denormalize
from stage3.refinement import PastRefinement, RefinementConfig, online_inputs

sys.path.insert(0, str(BASE / "evaluation"))
from past_protocol import sample_seed, action_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    meta = cp["metadata"]
    if meta["schema"] != "whenami_stage3_refinement_v1":
        raise ValueError("not a refinement checkpoint")
    cache = EndpointCache(meta["cache"])
    if cache.manifest_sha256 != meta["cache_manifest_sha256"]:
        raise ValueError("cache changed")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PastRefinement(RefinementConfig(**meta["model_config"]), mode=meta["mode"]).to(device).eval()
    model.load_state_dict(cp["model"], strict=True)
    if not all(torch.isfinite(x).all() for x in model.state_dict().values() if x.is_floating_point()):
        raise FloatingPointError("nonfinite checkpoint")
    reader = load_behavior(cache.manifest["endpoints"]["past"], device)
    raw = np.load(cache.path / "raw_actions.npy", mmap_mode="r")
    rows = []
    for dataset in sorted(set(cache.datasets.tolist())):
        indices = cache.valid_indices[cache.datasets[cache.valid_indices] == dataset]
        names = ["true_past", "online"]
        if meta["mode"] != "adapter":
            names += ["oracle_future", "mean_future_intervention"]
        predictions = {k: [] for k in names}
        for i in indices:
            batch = cache.batch([i], device)
            seed = sample_seed(dataset, cache.episodes[i], cache.timesteps[i], args.seed)
            with torch.inference_mode():
                result = model(*online_inputs(batch))
                conditions = {"true_past": batch["past_c"], "online": result["online"]}
                if meta["mode"] != "adapter":
                    conditions["oracle_future"] = model.oracle(*online_inputs(batch), batch["future_c"])
                    conditions["mean_future_intervention"] = model.correct(
                        *online_inputs(batch), model.r.future_mean[None],
                    )
                for name, condition in conditions.items():
                    action = reader.sample(batch["past_z"], condition, seed)
                    predictions[name].append(action.cpu().numpy()[0])
        stats = cache.manifest["endpoints"]["past"]["statistics"][dataset]["action"]
        for name, values in predictions.items():
            action = denormalize(np.stack(values), stats)
            target = np.asarray(raw[indices])
            metrics = action_metrics(action, target, stats)
            np.savez_compressed(
                args.output / f"{dataset}_{name}.npz", predictions=action, target=target,
                indices=indices, episode=cache.episodes[indices], timestep=cache.timesteps[indices],
            )
            rows.append({"dataset": dataset, "variant": name, **metrics})
            print("RESULT", dataset, name, metrics, flush=True)
    write_json(args.output / "results.json", {
        "complete": True, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "cache": str(cache.path), "cache_manifest_sha256": cache.manifest_sha256,
        "mode": meta["mode"], "noise_seed": args.seed, "precision": "float32",
        "results": rows,
        "notes": [
            "Online uses only Past H/z/C; oracle_future reads real future and is NOT deployable.",
            "All variants use the SAME frozen Past DiT and its own current z/statistics.",
            "9 Euler intervals for every evaluation, including models with 4-interval training auxiliary.",
            "mean_future_intervention is an input intervention, not the separately trained adapter control.",
            "Small previously inspected validation set; not a final test, not closed-loop success.",
        ],
    })


if __name__ == "__main__":
    main()
