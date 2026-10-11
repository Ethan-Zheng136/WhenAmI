"""Real raw student + real fixed oracle check, no optimizer/weight updates.

Checks freshly re-encoded teacher on selected development indices, and verifies
raw student sampling never needs Future images. This is not numerical parity
with earlier FP32 cached oracle evaluation.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from stage3.common import STORE, DEFAULT_PAST, write_json
from stage3.behavior import flow_draw
from stage3_scale.model import load_joint, tokenize_instruction
from stage3_scale.evaluate_joint import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = STORE / "outputs/stage3_reader_study/data_01/train_dev"
    manifest = json.loads((root / "manifest.json").read_text())
    frames = np.load(root / "frames.npy", mmap_mode="r")
    language = json.loads((root / "instructions.json").read_text())
    valid = np.load(root / "valid_indices.npy")
    stats = json.loads((DEFAULT_PAST.parents[1] / "dataset_statistics.json").read_text())
    device = torch.device("cuda")
    student, visual, teacher, transform, tokenizer, prompt_builder = load_joint(42, teacher=True)
    # BF16 only to fit both7B models on one accelerator; no training claim.
    student.to(device=device, dtype=torch.bfloat16).eval()
    teacher.to(device=device, dtype=torch.bfloat16).eval()
    visual.to(device).eval()
    check = []
    with torch.no_grad():
        for i in [int(valid[0]), int(valid[-1])]:
            current = {k: v[None].to(device) for k, v in transform(Image.fromarray(frames[i, 4])).items()}
            past = [transform(Image.fromarray(frames[i, j])) for j in range(4)]
            future = [transform(Image.fromarray(frames[i, j])) for j in range(5, 9)]
            past = {k: torch.stack([x[k] for x in past])[None].to(device) for k in current}
            future = {k: torch.stack([x[k] for x in future])[None].to(device) for k in current}
            ids = tokenize_instruction(language[i], tokenizer, prompt_builder)[None].to(device)
            actions = torch.from_numpy(np.load(root / "past_actions.npy", mmap_mode="r")[i:i + 1].copy()).to(device)
            x, tau, _ = flow_draw(actions, 123)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = visual(current, past, future)
                t_velocity = teacher(ids, torch.ones_like(ids, dtype=torch.bool), current,
                                     features["history_features"], features["history_vae"],
                                     features["future_condition"], x, tau)
                actual_c = student(input_ids=ids, attention_mask=torch.ones_like(ids, dtype=torch.bool),
                                   current=current, history_features=features["history_features"],
                                   history_vae=features["history_vae"], condition_only=True)
                velocity = student(noisy_actions=x, tau=tau, conditioning=actual_c)["velocity"]
            assert torch.isfinite(t_velocity).all() and torch.isfinite(velocity).all()
            assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
            check.append({"index": i, "teacher_velocity_rms": float(t_velocity.float().square().mean().sqrt()),
                          "student_velocity_rms": float(velocity.float().square().mean().sqrt()),
                          "future_condition_rms": float(features["future_condition"].float().square().mean().sqrt())})
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    evaluate(student, visual, root, args.output / "development_sample", transform, tokenizer,
             prompt_builder, stats, device, limit_per_dataset=1)
    write_json(args.output / "RESULT.json", {
        "complete": True, "real_teacher_forward_finite_and_frozen": True,
        "real_student_raw_sampling_complete": True, "checks": check,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "note": "BF16 local wiring only; no optimizer or FSDP multi-node training/performance claim.",
    })


if __name__ == "__main__":
    main()
