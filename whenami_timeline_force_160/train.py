"""[160-PPU RESUME COPY of whenami_timeline_force.train] Two-stage OXE27 action-shaped timeline with explicit50k and restart contracts."""
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from stage3_oxe_eval.prepare import write_json, sha
from stage3_oxe_eval50.policy import param_bf16
from whenami_unified.data import Inputs, batch, executed, DEV, INDEX
from whenami_timeline_force.model import build_policy, VARIANTS
from whenami_timeline_force.loading import A50, source_record, load_weights, SCHEMA
from whenami_timeline_force.losses import (
    TEACHER_REQUESTS, STUDENT_REQUESTS, requested_labels, draw, fm, current_loss, revision_loss)
from whenami_timeline_force.runtime import (
    fingerprints, reference_fingerprints, topology, wrap, learning_rate,
    optimizer_groups, gradients)
from whenami_timeline_force_160.checkpoint import CheckpointIO160 as CheckpointIO, migration_fingerprints

VIRTUAL_WORLD = 320  # original 20x16 data/noise index space; reproduced exactly at 160 via accum 2
from whenami_timeline_force.evaluate import evaluate


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("teacher", "student"), required=True)
    p.add_argument("--variant", choices=VARIANTS, default="revise")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--micro", type=int, default=1)
    p.add_argument("--global-batch", type=int, default=320)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--new-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--current-weight", type=float, default=.1)
    p.add_argument("--cognition-weight", type=float, default=.1)
    p.add_argument("--revision-weight", type=float, default=.1)
    p.add_argument("--hidden-every", type=int, default=4)
    p.add_argument("--sharding", choices=("full", "hybrid_zero2"), default="hybrid_zero2")
    p.add_argument("--activation", choices=("none", "llm", "all"), default="llm")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--first-check", type=int, default=100)
    p.add_argument("--stop-after", type=int, default=0)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--eval-max", type=int, default=0)
    p.add_argument("--engineering", action="store_true")
    p.add_argument("--bf16-storage", action="store_true", help="engineering single-card ONLY; formal uses FP32 params/Adam")
    return p


def main():
    parser = arguments()
    a = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    if (a.global_batch % (world*a.micro) or min(a.steps, a.micro, a.global_batch, a.draws,
        a.save_every, a.eval_every, a.first_check, a.hidden_every) < 1 or
        not 0 <= a.stop_after <= a.steps or not 0 <= a.warmup < a.steps):
        parser.error("invalid batch/schedule/counts")
    if a.bf16_storage and not a.engineering:
        parser.error("BF16 storage is engineering only")
    if not a.engineering and (world != 160 or int(os.environ["LOCAL_WORLD_SIZE"]) != 16 or
                             a.steps != 50000 or a.global_batch != 320):
        parser.error("160 resume recipe is10nodes×16=160,50k,globalbatch320")
    if a.stage == "student" and a.teacher is None:
        parser.error("explicit completed A+ teacher path required")
    if min(a.current_weight, a.cognition_weight, a.revision_weight, a.lr, a.new_lr) < 0:
        parser.error("negative weight/LR")
    torch.set_num_threads(1)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=40))
    error = [None]
    if rank == 0:
        try:
            a.output.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            error[0] = str(e)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise ValueError(error[0])
    dist.barrier()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    is_student = a.stage == "student"
    source = source_record(a.teacher if is_student else A50, "teacher" if is_student else "a50", a.engineering)
    if not a.engineering:
        from whenami_timeline_force.preflight import local_preflight
        local_preflight()
    data = Inputs(dev_limit=a.dev_limit)
    core, visual, transform, tokenizer, prompt = build_policy(is_student, a.variant, a.seed)
    parameter_count = sum(p.numel() for p in core.parameters() if p.requires_grad)
    if a.bf16_storage:
        core = param_bf16(core)
    elif any(p.is_floating_point() and p.dtype != torch.float32 for p in core.parameters()):
        raise ValueError("formal trainable storage must be FP32 before FSDP")
    strategy, groups = topology(a.sharding, device)
    model = wrap(core, device, strategy, groups, a.activation)
    group = groups[0] if groups else None
    if not a.resume_from:
        loaded = load_weights(model, a.teacher if is_student else A50, True, group)
    else:
        loaded = {"resume": str(a.resume_from)}
    teacher = None
    if is_student:
        tc, unused, _, _, _ = build_policy(False, "revise", a.seed)
        del unused
        tc.requires_grad_(False).eval()
        tc.vlm.vision_backbone_requires_grad = False
        if a.bf16_storage:
            tc = param_bf16(tc)
        teacher = wrap(tc, device, strategy, groups, "none").eval()
        teacher_loaded = load_weights(teacher, a.teacher, True, group)
    visual = visual.to(device).requires_grad_(False).eval()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
                if k not in ("output", "resume_from", "stop_after")}
    recipe = {"schema": SCHEMA, "settings": settings, "world_size": world,
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
        "source": fingerprints(), "reference_source": reference_fingerprints(),
        "warmstart": {"path": source["path"], "step": source["step"], "sha256": source["sha256"]},
        "train_index_sha256": sha(INDEX), "dev_manifest_sha256": sha(DEV/"MANIFEST.json"),
        "trainable_parameters": parameter_count,
        "action_interface": "ALL cognition/condition inputs from SAME evidence-versioned node",
        "data": "OXE27 four TRAIN anchors; teacher visual-only; student prefix<=3, executed intervals",
        "teacher_objective": "six equal FM requests, causal0/1/2+hindsight0/1/2 at version3, hidden1 every4",
        "student_objective": "three equal prefix-current FM + full hindsight C/z + old1 cutoffs1/2 velocity revision",
        "optimizer_precision": "BF16-storage ENGINEERING" if a.bf16_storage else "FP32 original params/Adam, BF16 FSDP compute"}
    recipe["topology_migration"] = {"virtual_world": VIRTUAL_WORLD, "physical_world": world,
        "rule": "virtual_rank=rank+micro*world; data.batch(micro=0) and draw seed use virtual rank/world",
        "code": migration_fingerprints()}
    if rank == 0:
        write_json(a.output/"recipe.json", recipe)
        write_json(a.output/"LOAD_VERIFIED.json", {"source": recipe["warmstart"], "model": loaded,
            "teacher": teacher_loaded if is_student else None})
    optimizer = torch.optim.AdamW(optimizer_groups(model, a.lr, a.new_lr), weight_decay=0., foreach=False)
    io = CheckpointIO(a.sharding, groups)
    random.seed(a.seed+rank)
    np.random.seed(a.seed+rank)
    torch.manual_seed(a.seed+rank)
    start = io.load(a.resume_from, model, optimizer, recipe) if a.resume_from else 0
    if a.stop_after and a.stop_after <= start:
        raise ValueError("stop-after must exceed restored step")
    if rank == 0:
        print("TIMELINE_FORCE_READY", a.stage, a.variant, world, recipe["optimizer_precision"], flush=True)
    evaluate(model, visual, data, transform, tokenizer, prompt, device,
             a.output/"development"/f"step-{start:06d}", is_student, limit=a.eval_max)
    progress = tqdm(total=a.steps, initial=start, disable=rank != 0, desc=f"Timeline {a.stage}")
    accum = a.global_batch//(world*a.micro)
    if not a.engineering and (world*accum != VIRTUAL_WORLD or a.micro != 1):
        raise ValueError("virtual-rank mapping needs world*accum==320, micro 1")
    recent, save_times, eval_times = deque(maxlen=100), [], []
    optimizer.zero_grad(set_to_none=True)
    for step in range(start+1, a.steps+1):
        begin = time.monotonic()
        model.train()
        if teacher is not None:
            teacher.eval()
        for group_opt, peak in zip(optimizer.param_groups, (a.lr, a.new_lr)):
            group_opt["lr"] = learning_rate(step, a.steps, peak, a.warmup)
        numeric = []
        for micro in range(accum):
            vrank = rank+micro*world
            cases = data.batch(a.micro, a.seed, step, vrank, 0)
            hide = 1 if not is_student and step % a.hidden_every == 0 else None
            # All four anchors encoded independently OUTSIDE trainable VLM;
            # student physically receives only prefix3.
            evidence, labels = batch(cases, visual, transform, tokenizer, prompt, device, hide=hide)
            requests = STUDENT_REQUESTS if is_student else TEACHER_REQUESTS
            action_labels, valid = requested_labels(labels, requests)
            noisy, tau, target = draw(action_labels, a.seed+step*VIRTUAL_WORLD+vrank, a.draws)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if is_student:
                    with torch.no_grad():
                        target_nodes = teacher(evidence=evidence, condition_only=True)
                    cmd = executed(cases, 3, device)
                    output = model(evidence=evidence.prefix(3), executed=cmd,
                                   noisy_actions=noisy, tau=tau, requests=requests)
                    action = fm(output["velocity"], target, valid)
                    current, cog = current_loss(output, target_nodes)
                    old_noisy, old_tau, _ = draw(labels.actions[:, 1], a.seed+step*991+vrank, a.draws)
                    revision, diag = revision_loss(output, target_nodes, teacher,
                                                   old_noisy, old_tau, labels.valid[:, 1])
                    loss = action+a.current_weight*current+a.cognition_weight*cog+a.revision_weight*revision
                else:
                    output = model(evidence=evidence, requests=requests, noisy_actions=noisy, tau=tau)
                    action = fm(output["velocity"], target, valid)
                    loss = action
                    current = cog = revision = action.detach()*0
                    diag = {"teacher_revision_velocity_rms": action.detach()*0,
                            "student_revision_velocity_rms": action.detach()*0}
            if not torch.isfinite(loss):
                raise FloatingPointError("timeline loss")
            (loss/accum).backward()
            numeric.append([float(x.detach()) for x in (action, current, cog, revision, loss,
                diag["teacher_revision_velocity_rms"], diag["student_revision_velocity_rms"])])
            del output, loss
            if is_student:
                del target_nodes
        audited = step in (start+1, 2, a.first_check)
        grad = gradients(model, device) if audited else None
        if audited and not all(np.isfinite(v) and v > 0 for v in grad.values()):
            raise ValueError("missing timeline gradient group: "+str(grad))
        norm = model.clip_grad_norm_(1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("timeline grad")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
            raise ValueError("frozen teacher got gradients")
        torch.cuda.synchronize()
        values = torch.tensor(np.mean(numeric, axis=0), device=device)
        dist.all_reduce(values)
        values /= world
        seconds = torch.tensor(time.monotonic()-begin, device=device)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        peak = torch.tensor(torch.cuda.max_memory_allocated()/2**30, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        recent.append(float(seconds))
        keys = ("action_fm", "current_alignment", "cognition_alignment", "revision", "total",
                "teacher_revision_velocity_rms", "student_revision_velocity_rms")
        if rank == 0:
            row = {"step": step, **dict(zip(keys, values.cpu().tolist())),
                   "seconds_max": float(seconds), "peak_allocated_gib_max": float(peak),
                   "grad_norm": float(norm), "gradients": grad,
                   "lr": [g["lr"] for g in optimizer.param_groups], "teacher_hidden": bool(hide is not None)}
            with (a.output/"train.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
            if step == start+1 and a.resume_from:
                write_json(a.output/"RESUME_VERIFIED.json", {"complete": True, "restored_step": start,
                    "first_update": step, "world": world, "optimizer_rng_restored": True})
        save_now = step % a.save_every == 0 or step in (a.first_check, a.steps, a.stop_after)
        eval_now = step % a.eval_every == 0 or step in (a.first_check, a.steps, a.stop_after)
        if save_now:
            phase = time.monotonic()
            if rank == 0:
                print("PHASE_BEGIN save", step, flush=True)
            io.save(a.output/"checkpoints"/f"step-{step:06d}", model, optimizer, step, recipe)
            save_times.append(time.monotonic()-phase)
            if rank == 0:
                print("PHASE_END save", step, save_times[-1], flush=True)
                with (a.output/"phase_events.jsonl").open("a") as f:
                    f.write(json.dumps({"phase": "save", "step": step, "seconds": save_times[-1]})+"\n")
        if eval_now:
            phase = time.monotonic()
            if rank == 0:
                print("PHASE_BEGIN eval", step, flush=True)
            evaluate(model, visual, data, transform, tokenizer, prompt, device,
                     a.output/"development"/f"step-{step:06d}", is_student, limit=a.eval_max)
            eval_times.append(time.monotonic()-phase)
            if rank == 0:
                print("PHASE_END eval", step, eval_times[-1], flush=True)
                with (a.output/"phase_events.jsonl").open("a") as f:
                    f.write(json.dumps({"phase": "eval", "step": step, "seconds": eval_times[-1]})+"\n")
        if rank == 0:
            speed = float(np.mean(recent))
            eta = str(timedelta(seconds=int((a.steps-step)*speed)))
            progress.update(1)
            progress.set_postfix(loss=f"{row['total']:.4f}", sec=f"{speed:.2f}", ETA=eta)
            if step in (start+1, a.first_check) or step % 10 == 0:
                next_save = min(a.steps, ((step//a.save_every)+1)*a.save_every)
                if step < a.first_check:
                    next_save = min(next_save, a.first_check)
                remaining_train = (a.steps-step)*speed
                remaining_saves = a.steps//a.save_every-step//a.save_every
                remaining_evals = a.steps//a.eval_every-step//a.eval_every
                eta_total = (remaining_train+remaining_saves*np.mean(save_times)+
                             remaining_evals*np.mean(eval_times)) if save_times and eval_times else None
                eta_total_text = str(timedelta(seconds=int(eta_total))) if eta_total is not None else "pending save/eval timing"
                print(f"PROGRESS {step:,}/{a.steps:,} | {speed:.2f}s/step | "
                      f"{a.global_batch/speed:.1f} windows/s | ETA_train {eta} | "
                      f"ETA_total {eta_total_text} | "
                      f"next_ckpt {next_save} | loss {row['total']:.4f} | "
                      f"peak {float(peak):.1f}GiB | mean_save {np.mean(save_times) if save_times else 'pending'} | "
                      f"mean_eval {np.mean(eval_times) if eval_times else 'pending'}", flush=True)
                with (a.output/"progress.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step, "seconds_per_step": speed,
                        "eta_train_seconds": remaining_train, "eta_total_seconds": eta_total,
                        "next_checkpoint": next_save, "peak_allocated_gib": float(peak)})+"\n")
        if step == a.stop_after:
            break
    progress.close()
    if rank == 0:
        write_json(a.output/"COMPLETE.json", {"complete": True, "step": step,
            "formal_training_complete": step == a.steps and not a.engineering,
            "stage": a.stage, "utc": datetime.now(timezone.utc).isoformat()})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
