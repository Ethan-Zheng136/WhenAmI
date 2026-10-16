"""Real A50k tensors, strong timeline routing, teacher/student gradients and Adam.

This is short single-PPU engineering validation, never formal training or new
A+ effectiveness. Formal FP32-state multi-node training is separately admitted.
"""
import argparse
import gc
from datetime import datetime, timezone
from pathlib import Path
import time
import torch
from stage3_oxe_eval.prepare import write_json
from stage3_oxe_eval50.policy import param_bf16
from stage3_perf.runtime import recompute
from whenami_unified.data import Inputs, batch, executed
from whenami_unified.contracts import Evidence, Frame, Executed
from whenami_timeline_force.model import build_policy
from whenami_timeline_force.loading import A50, source_record, load_weights
from whenami_timeline_force.runtime import gradients, clean_name
from whenami_timeline_force.losses import (
    TEACHER_REQUESTS, STUDENT_REQUESTS, requested_labels, draw, fm, current_loss, revision_loss)


def tamper(evidence, index):
    frames = list(evidence.frames)
    f = frames[index]
    frames[index] = Frame({k: x*.5+3 for k, x in f.pixels.items()},
                          f.input_ids, f.attention_mask, f.patches+4, f.vae-2)
    return Evidence(tuple(frames), evidence.times, evidence.visible)


def assert_grads(model, device, ignore=()):
    values = gradients(model, device)
    if any(not torch.isfinite(torch.tensor(v)) or v <= 0 for k, v in values.items() if k not in ignore):
        raise ValueError("real module gradient invalid: "+str(values))
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    source = source_record(A50)
    begin = time.monotonic()
    teacher, visual, transform, tokenizer, prompt = build_policy(False, "revise", 42)
    teacher = param_bf16(teacher).to(device).requires_grad_(False).eval()
    teacher.vlm.vision_backbone_requires_grad = False
    visual = visual.to(device).requires_grad_(False).eval()
    loaded = load_weights(teacher, A50)
    print("FORCE_A50_LOADED", loaded, flush=True)
    data = Inputs(dev_limit=1)
    cases = data.batch(1, 20261012, 1, 0)
    evidence, labels = batch(cases, visual, transform, tokenizer, prompt, device)
    cmd = executed(cases, 3, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        original = teacher(evidence=evidence, condition_only=True)
        changed = teacher(evidence=tamper(evidence, 3), condition_only=True)
        prefix = teacher(evidence=evidence.prefix(3), condition_only=True)
        hidden = evidence.hide(1)
        masked = teacher(evidence=hidden, condition_only=True)
        masked_tampered = teacher(evidence=tamper(hidden, 1), condition_only=True)
        shifted = teacher(evidence=Evidence(evidence.frames, evidence.times+777, evidence.visible), condition_only=True)
    for v in range(3):
        for field in ("node_versions", "node_z_versions"):
            if not torch.equal(original[field][v], changed[field][v]):
                raise ValueError("future leaked into causal version: "+field)
    for field in ("conditions", "current_z"):
        if not torch.equal(masked[field], masked_tampered[field]):
            raise ValueError("hidden observation shortcut")
        if not torch.equal(original[field], shifted[field]):
            raise ValueError("unstable moving origin")
    if not torch.equal(original["node_z_versions"][2], prefix["node_z_versions"][2]):
        raise ValueError("physical prefix routing")
    hindsight_delta = float((original["node_z_versions"][3][:, 2].float()-
                              changed["node_z_versions"][3][:, 2].float()).abs().max())
    if hindsight_delta <= 0:
        raise ValueError("no hindsight path")
    print("FORCE_TEACHER_CAUSALITY_PASS", hindsight_delta, flush=True)
    # Both reader inputs recorded via hook must come from SAME versioned node.
    acts, valid = requested_labels(labels, TEACHER_REQUESTS)
    noisy, tau, target = draw(acts, 42, 1)
    captured = {}
    def capture(module, args, kwargs):
        captured["z"] = args[2]
        captured["c"] = kwargs["per_token"]
    hook = teacher.reader.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = teacher(evidence=evidence, requests=TEACHER_REQUESTS, noisy_actions=noisy, tau=tau)
    hook.remove()
    expected_z = torch.stack([output["node_z_versions"][v][:, i] for i, v in TEACHER_REQUESTS], 1).flatten(0, 1)
    expected_c = torch.stack([output["node_versions"][v][:, i] for i, v in TEACHER_REQUESTS], 1).flatten(0, 1)
    if not torch.equal(captured["z"], expected_z) or not torch.equal(captured["c"], expected_c):
        raise ValueError("raw cognition bypass")
    raw_vs_node = float((output["raw_z"][:, 2].float()-output["node_z_versions"][2][:, 2].float()).abs().max())
    del output, captured, expected_z, expected_c
    # A+ stage1 BOTH causal and hindsight action supervision.
    teacher.requires_grad_(True).train()
    teacher.vlm.vision_backbone_requires_grad = True
    recompute(teacher, "llm")
    audits = {}
    for name, requests in (("causal_FM", ((2, 2),)), ("hindsight_FM", ((2, 3),))):
        actions, valid = requested_labels(labels, requests)
        x, times, velocity_target = draw(actions, 42, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = teacher(evidence=evidence, requests=requests, noisy_actions=x, tau=times)
            loss = fm(out["velocity"], velocity_target, valid)
        loss.backward()
        audits[name] = {"loss": float(loss.detach()), "gradients": assert_grads(teacher, device)}
        teacher.zero_grad(set_to_none=True)
        del out, loss
    teacher.requires_grad_(False).eval()
    teacher.vlm.vision_backbone_requires_grad = False
    print("FORCE_STAGE1_GRADIENTS_PASS", audits, flush=True)
    student, unused, _, _, _ = build_policy(True, "revise", 42)
    del unused
    student = param_bf16(student).to(device)
    mismatch = student.load_state_dict(teacher.state_dict(), strict=False)
    if mismatch.unexpected_keys or any(not k.startswith(("executed_adapter.", "forecast_head.")) for k in mismatch.missing_keys):
        raise ValueError("teacher-student coverage")
    student_parameters = {clean_name(name): p0 for name, p0 in student.named_parameters()}
    for name, p0 in teacher.named_parameters():
        if p0.data_ptr() == student_parameters[clean_name(name)].data_ptr():
            raise ValueError("teacher-student storage alias")
    recompute(student, "llm")
    actions, valid = requested_labels(labels, STUDENT_REQUESTS)
    x, times, velocity_target = draw(actions, 43, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = student(evidence=evidence.prefix(3), executed=cmd, requests=STUDENT_REQUESTS,
                      noisy_actions=x, tau=times)
        old = out["revised_states"][-1]
        old.retain_grad()
        loss = fm(out["velocity"], velocity_target, valid)
    loss.backward()
    audits["student_current_FM"] = {"loss": float(loss.detach()),
                                    "gradients": assert_grads(student, device, ("forecast_head",))}
    old_gradient = float(old.grad.float().abs().sum()) if old.grad is not None else 0.
    if old_gradient <= 0:
        raise ValueError("current FM bypassed revised historical node")
    student.zero_grad(set_to_none=True)
    del out, old, loss
    gc.collect()
    # Isolated reader semantic revision must backpropagate through frozen teacher.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        targets = teacher(evidence=evidence, condition_only=True)
    old_x, old_tau, _ = draw(labels.actions[:, 1], 44, 1)
    student.forecast_head.float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = student(evidence=evidence.prefix(3), executed=cmd, condition_only=True)
        value, diag = revision_loss(output, targets, teacher, old_x, old_tau, labels.valid[:, 1])
    value.backward()
    audits["isolated_revision"] = {"loss": float(value.detach()),
                                   "gradients": assert_grads(student, device, ("dit",)),
                                   **{k: float(v) for k, v in diag.items()}}
    if any(p.grad is not None for p in teacher.parameters()):
        raise ValueError("teacher received input-gradient loss")
    student.zero_grad(set_to_none=True)
    del output, value
    # Actual full-model AdamW, BF16 body/moments only for single-device engineering.
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, foreach=False, weight_decay=0.)
    teacher_probe = teacher.node_cognition.readout.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = student(evidence=evidence.prefix(3), executed=cmd, requests=STUDENT_REQUESTS,
                         noisy_actions=x, tau=times)
        action = fm(output["velocity"], velocity_target, valid)
        current, cog = current_loss(output, targets)
        revision, _ = revision_loss(output, targets, teacher, old_x, old_tau, labels.valid[:, 1])
        total = action+.1*current+.1*cog+.1*revision
    total.backward()
    audits["student_joint"] = {"loss": float(total.detach()), "action": float(action.detach()),
                               "condition": float(current.detach()), "cognition": float(cog.detach()),
                               "revision": float(revision.detach()), "gradients": assert_grads(student, device)}
    before = student.forecast_head.weight.detach().clone()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
    optimizer.step()
    if torch.equal(before, student.forecast_head.weight):
        raise ValueError("Adam no actual update")
    adam_states = len(optimizer.state)
    adam_dtypes = sorted({str(v.dtype) for st in optimizer.state.values() for v in st.values() if isinstance(v, torch.Tensor)})
    if not torch.equal(teacher_probe, teacher.node_cognition.readout.weight):
        raise ValueError("teacher changed")
    del output, total, action, current, cog, revision, before, optimizer
    student.zero_grad(set_to_none=True)
    gc.collect()
    student.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        enc = student(evidence=evidence.prefix(3), executed=cmd, condition_only=True)
        xl, tl, _ = draw(labels.actions[:, 2], 42, 1)
        sampled = student(conditioning={"conditions": enc["conditions"], "reader_z": enc["current_z"]},
                          noisy_actions=xl, tau=tl)["velocity"]
        plain = student(evidence=evidence.prefix(3), executed=cmd, noisy_actions=xl, tau=tl)["velocity"]
        stale = student(evidence=evidence.prefix(3), executed=cmd, noisy_actions=xl, tau=tl,
                        stale_current=True)["velocity"]
        swapped_future = student(evidence=tamper(evidence, 3).prefix(3), executed=cmd, condition_only=True)
    if not torch.equal(sampled, plain) or not torch.equal(enc["conditions"], swapped_future["conditions"]):
        raise ValueError("decoder deployment or student future invariance")
    stale_delta = float((plain.float()-stale.float()).abs().max())
    if stale_delta <= 0:
        raise ValueError("revised old versions don't affect actual action")
    # Current target labels don't create commands; padding NaNs are masked.
    bad_commands = cmd.commands.clone()
    bad_commands[~cmd.mask] = torch.nan
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ignored = student(evidence=evidence.prefix(3), executed=Executed(bad_commands, cmd.mask), condition_only=True)
    if not torch.equal(enc["conditions"], ignored["conditions"]):
        raise ValueError("padding commands leaked")
    result = {"complete": True, "engineering_only": True, "hardware": torch.cuda.get_device_name(0),
        "utc": datetime.now(timezone.utc).isoformat(), "source": {k: source[k] for k in ("path", "step", "sha256")},
        "loaded": loaded, "new_teacher_training_completed": False,
        "all_reader_conditions_from_node": True, "teacher_causal_future_invariance": True,
        "hidden_frame_invariance": True, "stable_time_origin": True, "teacher_student_nonaliased": True,
        "frozen_teacher_input_gradients": True, "deployment_decoder_bitwise_parity": True,
        "student_future_and_command_padding_invariance": True,
        "raw_vs_node_z_max_abs": raw_vs_node, "hindsight_z_intervention_max_abs": hindsight_delta,
        "action_to_updated_old_gradient_l1": old_gradient, "stale_old_velocity_max_abs": stale_delta,
        "audits": audits, "adam_states": adam_states, "adam_dtypes": adam_dtypes,
        "optimizer_precision": "single-card BF16body/moments, P FP32; formalFP32-stateFSDPadmission pending",
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "elapsed_seconds": time.monotonic()-begin,
        "scope": "realA50k,newnode-routing,oneAdamengineeringupdate; no learned effect or multirank claim"}
    write_json(a.output/"RESULT.json", result)
    print("TIMELINE_FORCE_REAL_SMOKE_PASS", result, flush=True)


if __name__ == "__main__":
    main()
