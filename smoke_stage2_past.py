"""Single-PPU integration smoke on one real RLDS sample.

Loads the actual 7B VLM / DINO / Wan / QFormer / DiT, but does not update weights
or save a checkpoint. --backward freezes the VLM for memory and checks gradients
through the real Past QFormer + DiT. This is wiring validation, not evaluation.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

REPO = Path(__file__).parent / "WoG"
sys.path.insert(0, str(REPO))

from vla import load_vla
from vla.past import PastSpec, checkpoint_spec
from prismatic.vla.datasets.datasets import RLDSBatchTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets.rlds import make_single_dataset
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=(
        "/mnt/pfs/zhengguantian/WoG_v2/outputs/stage1/baseline_v2--image_aug/"
        "checkpoints/step-020000-epoch-02-loss=0.1076.pt"
    ))
    ap.add_argument("--dataset", default="fractal20220817_data")
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.out and args.out.exists():
        raise FileExistsError(f"Choose a new smoke output path: {args.out}")
    start = time.time()
    torch.manual_seed(0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    _, offsets = checkpoint_spec(checkpoint, 2)
    del checkpoint
    spec = PastSpec.from_offsets(offsets)
    norm_stats = json.loads((Path(args.checkpoint).parents[1] / "dataset_statistics.json").read_text())
    print("Reading one real RLDS trajectory before allocating full GPU model...", flush=True)
    kwargs, _ = get_oxe_dataset_kwargs_and_weights(
        Path("/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment"),
        [(args.dataset, 1.0)], load_camera_views=("primary",),
        load_depth=False, load_proprio=False, load_language=True,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )
    kwargs = kwargs[0]
    kwargs.update(
        dataset_statistics=norm_stats[args.dataset],
        num_parallel_reads=1, num_parallel_calls=1, shuffle=False,
    )
    chunk_stats = {
        "action": {k: np.asarray(v) for k, v in norm_stats[args.dataset]["action"].items()}
    }
    dataset, _, _ = make_single_dataset(
        kwargs, train=False,
        traj_transform_kwargs=dict(
            window_size=spec.lookback + 1, future_action_window_size=15,
            obs_offsets=list(spec.offsets), skip_unlabeled=True,
            dataset_statistics=chunk_stats, num_parallel_calls=1,
        ),
        frame_transform_kwargs=dict(resize_size=(224, 224), num_parallel_calls=1),
    )
    trajectory = next(dataset.as_numpy_iterator())
    index = min(spec.lookback, max(0, len(trajectory["action"]) - 16))

    def at(x):
        return {k: at(v) for k, v in x.items()} if isinstance(x, dict) else x[index]
    sample = at(trajectory)
    stamps = sample["observation"]["timestep"].tolist()
    assert all(t <= stamps[-1] for t in stamps)
    print("Sample timesteps:", stamps, "actions:", sample["action"].shape, flush=True)
    print("Loading full model as Past Stage 2...", flush=True)
    model = load_vla(
        args.checkpoint, whenami_stage=2, train_venc=True, load_for_training=True,
        action_model_type="DiT-L",
    ).float().cuda().eval()
    tf = RLDSBatchTransform(
        None, model.llm_backbone.get_tokenizer(), model.vision_backbone.get_image_transform(),
        model.llm_backbone.prompt_builder_fn, obs_offsets=list(spec.offsets),
    )
    tok = model.llm_backbone.get_tokenizer()
    collate = PaddedCollatorForActionPrediction(tok.model_max_length, tok.pad_token_id)
    batch = collate([tf(sample)])
    batch.pop("dataset_names", None)

    def to_gpu(x):
        return {k: to_gpu(v) for k, v in x.items()} if isinstance(x, dict) else x.cuda()
    batch = to_gpu(batch)
    assert batch["future_pixel_values"] == {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _, act_loss, obs_loss = model(
            **batch, repeated_diffusion_steps=1, output_hidden_states=True
        )
    assert torch.isfinite(loss) and obs_loss == 0
    result = {
        "stage": 2, "checkpoint": args.checkpoint, "dataset": args.dataset,
        "timesteps": stamps, "action_loss_smoke_only": float(act_loss),
        "no_future_images": True,
    }
    images = [Image.fromarray(im) for im in sample["observation"]["image_primary"]]
    with torch.no_grad():
        pred, _ = model.predict_action(
            image=images[-1], past_frames=images[:-1],
            instruction=sample["task"]["language_instruction"].decode(),
            unnorm_key=args.dataset,
        )
    assert pred.shape == (16, 7) and np.isfinite(pred).all()
    result["inference_shape"] = list(pred.shape)
    print("Full forward and causal predict_action passed.", flush=True)
    if args.backward:
        # No optimizer and no source-checkpoint mutation. This is not a full
        # VLM fine-tuning/multinode-memory test.
        model.vlm.requires_grad_(False)
        model.train()
        model.vlm.eval()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, repeated_diffusion_steps=1, output_hidden_states=True)[0]
        loss.backward()
        for name, p in [
            ("past_v_proj", model.past_qformer.v_proj.weight),
            ("dit_per_attn", model.action_model.net.blocks[0].per_attn.in_proj_weight),
        ]:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            result[name + "_grad_norm"] = float(p.grad.norm())
        result["backward_past_dit"] = True
    result["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
    result["elapsed_seconds"] = time.time() - start
    print(json.dumps(result, indent=2), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Do not overwrite a previous diagnostic result.
        with args.out.open("x") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
