"""Re-encode Future H using only the current image repeated at every frame.

Calls the actual frozen endpoint's condition path, not a synthetic feature
shuffle. Original-future spot checks must reproduce the existing cache before
the new static cache is marked complete. Does not run the VLM or action sampler.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stage3.cache import EndpointCache
from stage3.common import check_endpoint, check_sources, reference_imports, sha256, write_json


def static_frames(current, count=4):
    if count < 1:
        raise ValueError("positive frame count required")
    return [current] * count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cache = EndpointCache(args.cache)
    reference_imports()
    from vla import load_vla
    from PIL import Image
    record = cache.manifest["endpoints"]["future"]
    model = load_vla(record["path"], train_venc=True, action_model_type="DiT-L",
                     load_for_training=False).float().cuda().eval().requires_grad_(False)
    frames = np.load(cache.path / "frames.npy", mmap_mode="r")
    offsets = list(cache.manifest["frame_offsets"])
    current_index = offsets.index(0)
    future_indices = [offsets.index(o) for o in (3, 7, 11, 15)]
    out_path = args.output / "future_static_h.npy"
    output = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=cache.arrays["future_h"].shape,
    )
    output[:] = np.nan
    captured = {}

    def before_projection(module, inputs):
        captured["h"] = inputs[0].detach().float().cpu().numpy()[0].copy()

    hook = model.qformer.v_proj.register_forward_pre_hook(before_projection)
    errors = []
    check_ids = set(np.linspace(0, cache.manifest["n"] - 1, 12, dtype=int).tolist())
    try:
        for i in range(cache.manifest["n"]):
            current = Image.fromarray(frames[i, current_index])
            pixels = model.vlm.vision_backbone.image_transform(current)
            if not isinstance(pixels, dict):
                raise ValueError("expected actual DINO/SigLIP dictionary preprocessing")
            pixels = {k: v[None].cuda() for k, v in pixels.items()}
            if i in check_ids:
                original = [Image.fromarray(frames[i, j]) for j in future_indices]
                model._condition_from_frames(original, None, pixels, torch.float32)
                expected = np.asarray(cache.arrays["future_h"][i])
                error = float(np.max(np.abs(captured["h"] - expected)))
                np.testing.assert_allclose(captured["h"], expected, rtol=1e-4, atol=1e-5)
                errors.append({"index": i, "max_abs": error})
            model._condition_from_frames(static_frames(current), None, pixels, torch.float32)
            output[i] = captured["h"]
            if (i + 1) % 64 == 0:
                print("STATIC", i + 1, "/", cache.manifest["n"], flush=True)
    finally:
        hook.remove()
    if not np.isfinite(output).all():
        raise FloatingPointError("static feature extraction incomplete/nonfinite")
    output.flush()
    check_endpoint(record)
    check_sources(cache.manifest["reference_source_sha256"])
    write_json(args.output / "manifest.json", {
        "complete": True, "parent_cache_manifest_sha256": cache.manifest_sha256,
        "n": cache.manifest["n"], "array_sha256": sha256(out_path),
        "original_future_cache_checks": errors, "source_sha256": sha256(Path(__file__)),
        "notes": [
            "All four Future DINO frames replaced by current; VAE sees current repeated five times.",
            "Only current image enters the static condition encoder; no language or action labels.",
            "Original endpoint method used, with pre-v_proj hook; no VLM generation/action sampling.",
            "Real-future spot checks must match parent H cache; all original checkpoints/source unchanged.",
        ],
    })
    print("STATIC_COMPLETE", max(x["max_abs"] for x in errors), flush=True)


if __name__ == "__main__":
    main()
