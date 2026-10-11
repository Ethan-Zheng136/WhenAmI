"""WoG-A current-only reference on the exact Stage3 pilot validation windows.

Uses the checkpoint's own current-image/language VLM and normalization. Does
not pass history or future frames to the model. The only inference hook makes
Euler action sampling use the same per-window noise seed as stage3.evaluate.
No training-budget equivalence with our endpoints is implied.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from stage3.behavior import FrozenBehavior
from stage3.cache import EndpointCache
from stage3.common import BASE, STORE, endpoint_record, check_endpoint, reference_imports, write_json

sys.path.insert(0, str(BASE / "evaluation"))
from past_protocol import sample_seed, action_metrics


def current_image_array(frames, offsets, index):
    if list(offsets).count(0) != 1:
        raise ValueError("unique current frame required")
    return frames[index, list(offsets).index(0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=STORE / "ckpts/WoG-A/checkpoints/50k-1229.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cache = EndpointCache(args.cache)
    record = endpoint_record(args.checkpoint)
    cp = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if record["config"].get("train_venc", True):
        raise ValueError("expected WoG Stage II, train_venc=False")
    if any("per_attn" in k for k in cp["model"]["action_model"]) or "past_qformer" in cp["model"]:
        raise ValueError("not a current-only WoG-A checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    reference_imports()
    from vla import load_vla
    from PIL import Image
    model = load_vla(
        args.checkpoint, train_venc=False, action_model_type="DiT-L", load_for_training=False,
    ).float().cuda().eval().requires_grad_(False)
    if model.whenami_stage != 0 or model.action_model.net.use_per_attn:
        raise ValueError("model was not loaded as original WoG Stage II")
    # Loader permits some historical optional keys; enforce the actual online
    # action network and VLM weights independently here.
    modules = {
        "action_model": model.action_model, "projector": model.vlm.projector,
        "llm_backbone": model.vlm.llm_backbone, "vision_backbone": model.vlm.vision_backbone,
    }
    for group, module in modules.items():
        module.load_state_dict(cp["model"][group], strict=True)
    del cp
    reader = FrozenBehavior(model.action_model.net)
    frames = np.load(cache.path / "frames.npy", mmap_mode="r", allow_pickle=False)
    target = np.load(cache.path / "raw_actions.npy", mmap_mode="r", allow_pickle=False)
    language = json.loads((cache.path / "instructions.json").read_text())
    original_inference = model.action_model.inference
    sampler_seed = 0

    def paired_sample(z, num_steps=10, per_token=None):
        if per_token is not None:
            raise ValueError("WoG-A must not receive real/predicted future conditions")
        return reader.sample(z, None, sampler_seed, num_steps)

    model.action_model.inference = paired_sample
    summaries = []
    try:
        for name in sorted(set(cache.datasets.tolist())):
            ids = cache.valid_indices[cache.datasets[cache.valid_indices] == name]
            predictions = []
            for j, i in enumerate(ids):
                sampler_seed = sample_seed(name, cache.episodes[i], cache.timesteps[i], args.seed)
                image = Image.fromarray(current_image_array(frames, cache.manifest["frame_offsets"], i))
                with torch.inference_mode():
                    actions, _ = model.predict_action(image=image, instruction=language[i], unnorm_key=name)
                predictions.append(actions)
                if (j + 1) % 20 == 0:
                    print("WOG-A", name, j + 1, "/", len(ids), flush=True)
            predictions = np.stack(predictions)
            truth = np.asarray(target[ids])
            metrics = action_metrics(predictions, truth, cache.manifest["endpoints"]["past"]["statistics"][name]["action"])
            np.savez_compressed(
                args.output / f"{name}_wog_stage2.npz", predictions=predictions, target=truth,
                indices=ids, episode=cache.episodes[ids], timestep=cache.timesteps[ids],
            )
            summaries.append({"dataset": name, "variant": "wog_stage2", **metrics})
            print("RESULT", summaries[-1], flush=True)
    finally:
        model.action_model.inference = original_inference
    check_endpoint(record)
    write_json(args.output / "results.json", {
        "complete": True, "checkpoint": record, "cache": str(cache.path),
        "cache_manifest_sha256": cache.manifest_sha256, "noise_seed": args.seed,
        "precision": "FP32 action network; original VLM inference autocast",
        "results": summaries,
        "notes": [
            "Original WoG Stage II / WoG-A; only current image and language enter policy.",
            "All online action/VLM checkpoint groups strictly loaded.",
            "Original preprocessing and 10-point Euler grid; same per-window action noise seed.",
            "Own checkpoint normalization to raw actions, then common Past statistics for reported errors.",
            "Released checkpoint has a different training budget; reference, not controlled architecture ablation.",
            "Not a held-out full-model benchmark or a closed-loop success measurement.",
        ],
    })


if __name__ == "__main__":
    main()
