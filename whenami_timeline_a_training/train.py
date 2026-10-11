"""Same admitted architecture plus real-dev evaluation and a fixed training recipe."""

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
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from stage3.common import BASE, STORE, REFERENCE
from stage3_oxe_eval.prepare import sha, write_json
from whenami_timeline_a.backbone import Foundation
from whenami_timeline_a.data import RawAnchors, INDEX, stack_cases, visibility
from whenami_timeline_a.model import MODES
from whenami_timeline_a.objectives import noisy_labels, velocity_loss
from whenami_timeline_a.smoke import optimizer_for, gradient_report
from whenami_timeline_a import checkpoint
from whenami_timeline_a_training.evaluate import dataset_cases, evaluate

DEFAULT_DEV = STORE / "outputs/whenami_timeline_a/dev27_16ep_01"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--micro", type=int, default=2)
    p.add_argument("--nodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--first-check", type=int, default=100)
    p.add_argument("--stop-after", type=int, default=0)
    p.add_argument("--resume-from", type=Path)
    args = p.parse_args()
    if min(args.steps, args.micro, args.save_every, args.eval_every, args.first_check) < 1:
        p.error("positive training/evaluation sizes required")
    if not 0 <= args.stop_after <= args.steps or args.nodes < 2:
        p.error("invalid stop/nodes")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    errors = [None]
    if rank == 0:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            errors[0] = repr(exc)
    if world > 1:
        dist.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise RuntimeError(errors[0])
    if world > 1:
        dist.barrier()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    foundation = Foundation(f"cuda:{local}")
    raw = RawAnchors(nodes=args.nodes)
    core = foundation.model(args.mode, args.seed)
    # Construct/evaluate dev BEFORE resetting training RNG.
    dev, total_dev, dev_manifest = dataset_cases(args.dev, foundation, args.dev_limit)
    if dev_manifest["nodes"] != args.nodes or dev_manifest["train_index_sha256"] != sha(INDEX):
        raise ValueError("development node count or training-index contract mismatch")
    model = DDP(core, device_ids=[local], find_unused_parameters=True) if world > 1 else core
    optimizer = optimizer_for(core)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("output", "resume_from", "stop_after")}
    recipe = {
        "schema": "timeline_a_teacher_research_dev_v1", "settings": settings,
        "world": world, "global_trajectories": world * args.micro,
        "index_sha256": sha(INDEX), "dev_manifest_sha256": sha(args.dev / "MANIFEST.json"),
        "endpoint": foundation.endpoint,
        "admitted_model_source": {p.name: sha(p) for p in (BASE / "whenami_timeline_a").glob("*.py")},
        "training_source": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        "dependencies": {str(p.relative_to(BASE)): sha(p) for directory in
                         ("stage3", "stage3_scale", "evaluation", "whenami_revision")
                         for p in (BASE / directory).glob("*.py")},
        "reference_source": {str(p.relative_to(REFERENCE)): sha(p) for p in REFERENCE.rglob("*.py")},
        "learning_rates": {"local_qformer": 1e-5, "original_dit": 1e-5, "temporal": 1e-4},
        "schedule": "constant; unchanged from admitted smoke optimizer",
        "mask_schedule": "odd updates fully visible; even updates one hidden interior node",
        "scope": "Phase-A teacher research, not final deployable Stage3 or Phase B",
    }
    if rank == 0:
        write_json(args.output / "recipe.json", recipe)
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    start = checkpoint.load(args.resume_from, model, optimizer, recipe) if args.resume_from else 0
    if args.resume_from and rank == 0:
        write_json(args.output / "RESTORED.json", {"step": start, "checkpoint": str(args.resume_from)})
    # Both fresh and restored models evaluate once, with full RNG restoration.
    evaluate(model, dev, foundation, args.output / "development" / f"step-{start:06d}", total_dev)
    recent = deque(maxlen=100)
    phase_times = {"save": [], "eval": []}
    progress = tqdm(total=args.steps, initial=start, disable=rank != 0, desc=f"Timeline-A {args.mode}")
    for step in range(start + 1, args.steps + 1):
        began = time.monotonic()
        cases = []
        for i in range(args.micro):
            entry, times = raw.select(args.seed, step, rank, i)
            obs, labels, provenance = raw.materialize(entry, times, foundation.statistics)
            cases.append({"inputs": foundation.encode(obs), "labels": labels, "provenance": provenance})
        x, labels = stack_cases(cases, foundation.device)
        visible, cutoff = visibility(x["times"], args.seed + step * world + rank,
                                    hide_interior=step % 2 == 0)
        noisy, tau, target = noisy_labels(labels["actions"], args.seed + step * world + rank)
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**x, visible=visible, cutoffs=cutoff, noisy_actions=noisy, tau=tau)
            loss, _ = velocity_loss(output, target, labels["action_mask"])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        gradients = gradient_report(core)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient")
        optimizer.step()
        foundation.check_frozen()
        torch.cuda.synchronize()
        value = loss.detach().float()
        if world > 1:
            dist.all_reduce(value); value /= world
        seconds = time.monotonic() - began
        recent.append(seconds)
        row = {"step": step, "seconds": seconds, "action_fm": float(value), "grad_norm": float(norm),
               "gradients": gradients, "lr": [g["lr"] for g in optimizer.param_groups],
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "identities_rank0": [c["provenance"] for c in cases]}
        if rank == 0:
            with (args.output / "train.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if step == start + 1 and args.resume_from:
                write_json(args.output / "RESUME_VERIFIED.json", {
                    "complete": True, "restored": start, "first_update": step,
                    "finite_loss_and_gradient": True, "world": world})
        save_now = step % args.save_every == 0 or step in (args.first_check, args.steps, args.stop_after)
        eval_now = step % args.eval_every == 0 or step in (args.first_check, args.steps, args.stop_after)
        if save_now:
            began_phase = time.monotonic()
            if rank == 0: print("TIMELINE_A_CHECKPOINT", step, flush=True)
            checkpoint.save(args.output / "checkpoints" / f"step-{step:06d}",
                            model, optimizer, step, recipe)
            phase_times["save"].append(time.monotonic() - began_phase)
        if eval_now:
            began_phase = time.monotonic()
            evaluate(model, dev, foundation, args.output / "development" / f"step-{step:06d}", total_dev)
            phase_times["eval"].append(time.monotonic() - began_phase)
        if rank == 0:
            average = sum(recent) / len(recent)
            eta = (args.steps - step) * average
            overhead = sum((args.steps // period - step // period) * np.mean(phase_times[name])
                           for name, period in (("save", args.save_every), ("eval", args.eval_every))
                           if phase_times[name])
            progress.update()
            progress.set_postfix(loss=float(value), s_it=round(average, 2), eta_h=round((eta + overhead)/3600, 2))
            write_json(args.output / "progress.json", {
                "utc": datetime.now(timezone.utc).isoformat(), "step": step, "total": args.steps,
                "seconds_per_step": average, "eta_train_seconds": eta,
                "eta_total_estimate_seconds": eta + overhead,
                "eta_note": "measured save/eval overhead; initialization excluded; rank0 timing",
                "phase_seconds": {k: list(v) for k, v in phase_times.items()},
            })
        if args.stop_after and step == args.stop_after < args.steps:
            break
        if step == args.steps and rank == 0:
            write_json(args.output / "COMPLETE.json", {
                "complete": True, "steps": step, "world": world, "mode": args.mode,
                "final_dev_evaluated": True})
    progress.close()
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
