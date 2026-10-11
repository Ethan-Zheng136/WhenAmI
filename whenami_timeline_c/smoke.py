"""Real A50k -> C, actual OXE samples, gradient routing and one SGD update.

This is NOT full FP32-Adam multi-node training admission or efficacy evidence.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import torch

from stage3.common import DEFAULT_FUTURE
from stage3_oxe_eval.prepare import write_json
from stage3_perf.runtime import recompute
from whenami_timeline_joint.model import JointTimeline
from whenami_timeline_joint.runtime import gradient_norms
from whenami_timeline_b.model import draw
from whenami_timeline_b_handoff.runtime import build as cold_build
from whenami_timeline_c.model import TimelineStudent, objective
from whenami_timeline_c.data import Inputs, batch
from whenami_timeline_c.loading import load_model, teacher_record


def bf16_parameters(module):
    for p in module.parameters():
        p.data = p.data.to(dtype=torch.bfloat16)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    config = json.loads((DEFAULT_FUTURE.parents[1] / "config.json").read_text())
    source = teacher_record()
    base, visual, transform, tokenizer, prompt = cold_build(config)
    student = TimelineStudent(base.vlm, base.local_encoder, base.reader, "future_read", future_gap=4)
    student = bf16_parameters(student).to("cuda:0")
    del base
    student_load = load_model(student)
    base, unused_visual, _, _, _ = cold_build(config)
    teacher = JointTimeline(base.vlm, base.local_encoder, base.reader)
    teacher = bf16_parameters(teacher).requires_grad_(False).eval().to("cuda:0")
    teacher.vlm.vision_backbone_requires_grad = False
    del base, unused_visual
    teacher_load = load_model(teacher)
    if student.reader.final_layer.linear.weight.data_ptr() == teacher.reader.final_layer.linear.weight.data_ptr():
        raise ValueError("student/teacher storage alias")
    recompute(student, "llm")
    gc.collect()
    data = Inputs(future_gap=4, dev_limit=1)
    cases = data.batch(1, 20261006, 1, 0)
    full, features, prefix, label = batch(cases, visual, transform, tokenizer, prompt, "cuda:0", 4)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = teacher(**full, **features, visible=torch.ones_like(full["times"], dtype=torch.bool),
                          condition_only=True)
        target_nodes = encoded["conditions"].reshape(1, 3, 16, 32)[:, 1:3].detach()
    noisy, tau, flow_target = draw(label.actions, 42)
    reports = {}
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-3, momentum=0)
    for variant in ("future_aux", "future_read"):
        student.variant = variant
        student.set_step(1000)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = student(prefix=prefix, noisy_actions=noisy, tau=tau)
            output["future_condition"].retain_grad()
            action_only, _ = objective(output, flow_target, label.valid, target_nodes, weight=0.)
        action_only.backward()
        action_to_future = float(output["future_condition"].grad.float().norm())
        if (variant == "future_aux" and action_to_future != 0) or (
            variant == "future_read" and action_to_future <= 0
        ):
            raise ValueError("wrong action-to-future gradient routing")
        del output, action_only
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = student(prefix=prefix, noisy_actions=noisy, tau=tau)
            loss, metrics = objective(output, flow_target, label.valid, target_nodes)
        loss.backward()
        grads = gradient_norms(student, torch.device("cuda:0"))
        if not all(torch.isfinite(torch.tensor(v)) and v > 0 for v in grads.values()):
            raise ValueError("missing/nonfinite C body gradient")
        reports[variant] = {"read_scale": float(student.read_scale),
                            "action_to_future_gradient_norm": action_to_future,
                            "body_gradients": grads, "metrics": {k: float(v) for k, v in metrics.items()}}
        del output, loss
    before = {n: p.detach().clone() for n, p in student.timeline.named_parameters()}
    teacher_probe = teacher.reader.final_layer.linear.weight.detach().clone()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
    optimizer.step()
    updated = [n for n, p in student.timeline.named_parameters() if not torch.equal(p, before[n])]
    if not updated:
        raise ValueError("no actual C parameter update")
    if not torch.equal(teacher_probe, teacher.reader.final_layer.linear.weight):
        raise ValueError("teacher weights changed")
    if any(p.grad is not None for p in teacher.parameters()):
        raise ValueError("teacher received gradients")
    student.zero_grad(set_to_none=True)
    student.eval()
    changed = deepcopy(cases)
    changed[0]["observations"]["images"][2] = 255 - changed[0]["observations"]["images"][2]
    changed[0]["observations"]["instructions"][2] = "hidden future instruction must not enter student"
    _, _, other, _ = batch(changed, visual, transform, tokenizer, prompt, "cuda:0", 4)
    for field in ("input_ids", "attention_mask", "patches", "vae", "times"):
        if not torch.equal(getattr(prefix, field), getattr(other, field)):
            raise ValueError("hidden future changed student prefix")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = student(prefix=prefix, condition_only=True)
        b = student(prefix=other, condition_only=True)
    if not all(torch.equal(a[k], b[k]) for k in ("conditions", "reader_z", "node_conditions")):
        raise ValueError("future changed actual C student output")
    write_json(args.output / "RESULT.json", {
        "complete": True, "utc": datetime.now(timezone.utc).isoformat(),
        "teacher": source, "student_load": student_load, "teacher_load": teacher_load,
        "eligible_domains": len(data.eligible_counts), "eligible_counts": data.eligible_counts,
        "actual_training_case": cases[0]["provenance"], "arms": reports,
        "updated_temporal_parameters": updated, "teacher_frozen": True,
        "actual_future_input_invariance": True,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope": "Real A50k/OXE computation + BF16-body SGD smoke only. Not full Adam admission or model quality.",
    })
    print("C_REAL_SMOKE_PASS", reports, flush=True)


if __name__ == "__main__":
    main()
