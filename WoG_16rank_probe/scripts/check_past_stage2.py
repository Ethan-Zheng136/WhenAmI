"""Read-only Stage-2 preflight; no model training and no GPU weights allocation."""

import argparse
import ast
import os
from pathlib import Path

# This runs ONCE per node, BEFORE torchrun. The platform's WORLD_SIZE is a
# node count, not a process count. Importing vla would otherwise initialize
# accelerate/collectives with those launcher variables and may hang preflight.
for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
    os.environ.pop(key, None)

import torch  # Import before torchvision-dependent modules in the PPU image.
import numpy as np
import transformers
import timm

from vla.past import PastSpec, checkpoint_spec
from vla.qformer import QFormer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--past-init", choices=("checkpoint", "random"), default="checkpoint")
    ap.add_argument("--obs-offsets", default="[-15,-11,-7,-3,0]")
    ap.add_argument("--nproc", type=int, default=0, help="0: do not check GPU count")
    args = ap.parse_args()
    spec = PastSpec.from_offsets(ast.literal_eval(args.obs_offsets))
    if not transformers.__version__.startswith("4.40.") or not np.__version__.startswith("1.26."):
        raise RuntimeError("Use the WoG pinned environment: transformers 4.40.x / numpy 1.26.x")
    if timm.__version__ != "0.9.10":
        raise RuntimeError("WoG requires timm 0.9.10")
    if args.nproc and (not torch.cuda.is_available() or torch.cuda.device_count() < args.nproc):
        raise RuntimeError(f"Requested {args.nproc} processes, available GPUs={torch.cuda.device_count()}")
    ck = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    checkpoint_spec(ck, 2, spec.offsets)
    groups = ck["model"]
    if args.past_init == "checkpoint":
        key = "past_qformer" if "past_qformer" in groups else "qformer"
        if key not in groups or "action_model" not in groups:
            raise RuntimeError("Warm start requires a trained condition head and its action model")
        with torch.device("meta"):
            reference = QFormer(1024, 8, 2048, num_layers=6, num_query_tokens=16).state_dict()
        if set(reference) != set(groups[key]):
            raise RuntimeError(f"Checkpoint {key} keys do not match the unchanged WoG QFormer")
        for name, tensor in reference.items():
            if tensor.shape != groups[key][name].shape:
                raise RuntimeError(f"Shape mismatch: {key}.{name}")
        if not any("per_attn" in k for k in groups["action_model"]):
            raise RuntimeError("Checkpoint has no per-attention; use WoG Stage I, not WoG-A Stage II")
        print(f"PASS: strict QFormer shape/key match from {key} ({len(reference)} tensors)")
    else:
        if "past_qformer" in groups:
            raise RuntimeError("A trained Past checkpoint must use past-init=checkpoint")
        print("Past QFormer starts randomly; checkpoint is used for the VLM.")
    for rel in ("vision/dinov2_weights.pth", "vision/Wan2.1_VAE.pth"):
        p = Path(__file__).resolve().parents[1] / "pretrained" / rel
        if not p.is_file():
            raise FileNotFoundError(p)
    # The test also catches sparse-frame/raw-action indexing mistakes.
    actions = torch.arange(-spec.lookback, 16).view(1, -1, 1)
    target, _ = spec.action_targets(actions)
    assert target.flatten().tolist() == list(range(16))
    print("PASS:", spec.metadata(), "| target offsets=0..15")


if __name__ == "__main__":
    main()
