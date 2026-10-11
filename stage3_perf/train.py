"""Performance variant of the admitted raw full-policy trainer.

Reuses the SAME model, objective and data protocol. Different execution recipe,
fresh output required. Original stage3_scale source and existing jobs unchanged.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from stage3.common import BASE, DEFAULT_PAST, DEFAULT_FUTURE, endpoint_record, sha256, write_json, source_fingerprint
from stage3.behavior import flow_draw
from stage3_scale.model import load_joint, objective
from stage3_scale import checkpoint_v2 as checkpoint_io
from stage3.causal_student import cosine_learning_rate
from stage3_perf.data import SingleParseWindows, BatchPrefetch, to_device
from stage3_perf.runtime import topology, wrap, recompute, PhaseTimer
from stage3_perf.progress import Progress
from stage3_perf.checkpoint import CheckpointIO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("latent", "forecast", "distill_future"), default="forecast")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--global-batch", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--fm-draws", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--new-lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--alignment-weight", type=float, default=.1)
    parser.add_argument("--distill-weight", type=float, default=.3)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial-data", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--local-backward-only", action="store_true",
                        help="single device full backward with no Adam/update/save; wiring not a training result")
    parser.add_argument("--admission-gradient-audit", action="store_true",
                        help="check distributed nonzero gradients for each intended trainable component")
    parser.add_argument("--development-every", type=int, default=2500)
    parser.add_argument("--development-limit", type=int, default=0)
    parser.add_argument("--development-root", type=Path,
                        default=Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_reader_study/data_01/train_dev"))
    parser.add_argument("--sharding", choices=("full", "hybrid", "hybrid_zero2"), default="hybrid")
    parser.add_argument("--activation-checkpoint", choices=("all", "llm", "none"), default="none")
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--profile-every", type=int, default=100)
    parser.add_argument("--first-save", type=int, default=100,
                        help="Early recovery point, then normal save cadence; 0 disables it")
    parser.add_argument("--first-eval", type=int, default=100,
                        help="Early development action check; 0 disables it")
    args = parser.parse_args()
    world, rank, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if args.local_backward_only and world != 1:
        raise ValueError("backward-only is local diagnostic")
    if min(args.global_batch, args.micro_batch) < 1 or args.global_batch % (world * args.micro_batch) or min(
        args.steps, args.micro_batch, args.fm_draws, args.save_every, args.development_every
    ) < 1:
        raise ValueError("global batch must divide world*micro and all counts be positive")
    if not 0 <= args.warmup < args.steps or args.stop_after > args.steps or min(
        args.stop_after, args.development_limit, args.alignment_weight, args.distill_weight
    ) < 0 or min(args.lr, args.new_lr) <= 0 or min(
        args.prefetch_batches, args.first_save, args.first_eval, args.profile_every
    ) < 0 or args.log_every < 1:
        raise ValueError("invalid schedule/stop")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if not args.local_backward_only and not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    reservation = [None]
    if rank == 0:
        try:
            if args.resume:
                if not args.output.is_dir():
                    raise ValueError("no run to resume")
                if (args.output / "complete.json").exists():
                    raise ValueError("run already complete")
            else:
                args.output.mkdir(parents=True, exist_ok=False)
        except (ValueError, OSError) as error:
            reservation[0] = str(error)
    if dist.is_initialized():
        dist.broadcast_object_list(reservation, src=0)
    if reservation[0]:
        raise ValueError(reservation[0])
    checkpoint_io.barrier()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if rank == 0:
        print("INITIALIZING model and OXE index; progress bar starts at first optimizer update.", flush=True)
    student, visual, teacher, transform, tokenizer, prompt_builder = load_joint(args.seed, args.variant == "distill_future")
    stats = json.loads((DEFAULT_PAST.parents[1] / "dataset_statistics.json").read_text())
    dataset = SingleParseWindows(args.data_index, transform, tokenizer, prompt_builder, stats, args.allow_partial_data)
    from stage3_scale.oxe_data import oxe_source_hashes
    if dataset.meta["oxe_source_sha256"] != oxe_source_hashes():
        raise ValueError("OXE standardization changed since preparation")
    # Bind data and every relevant implementation, not merely the small index path.
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("resume", "stop_after")}
    recipe = {
        "schema": "whenami_joint_perf_v1", "settings": settings, "world_size": world,
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
        "data_index_sha256": sha256(args.data_index),
        "past": endpoint_record(DEFAULT_PAST), "future": endpoint_record(DEFAULT_FUTURE),
        "reference_source": source_fingerprint(),
        "joint_source": {p.name: sha256(p) for p in (BASE / "stage3_scale").glob("*.py")},
        "performance_source": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "reused_source": {
            name: sha256(BASE / name) for name in (
                "stage3/common.py", "stage3/connector.py", "stage3/refinement.py",
                "stage3/reader_adaptation.py", "stage3/oracle_evidence.py",
                "stage3/behavior.py", "stage3/causal_student.py", "stage3/evaluate.py",
                "stage3/train_causal_student.py", "evaluation/past_protocol.py",
            )
        },
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "student_scope": ["VLM vision/projector/LLM", "Past QFormer", "R", "K", "full Past DiT"],
        "frozen_scope": ["DINO/Wan foundation temporal features", "Future QFormer", "optional complete oracle teacher"],
        "notes": ["Full raw-image training; no cached VLM z or Past query features.",
                  "TRAIN-role episode-boundary windows, padded actions masked; dev/test excluded.",
                  "New protocol; no multi-time-anchor recurrence or Stage4 robot execution wrapper yet."],
    }
    # STORE is mounted separately on some nodes; resolve via the endpoint store.
    from stage3.common import STORE
    recipe["reader_init_sha256"] = sha256(
        STORE / f"outputs/stage3_reader_study/runs_01/adaptive_mean_seed{args.seed}/checkpoints/step-005000.pt"
    )
    if teacher is not None:
        recipe["teacher_reader_sha256"] = sha256(
            STORE / f"outputs/stage3_reader_study/runs_01/adaptive_future_seed{args.seed}/checkpoints/step-005000.pt"
        )
    if rank == 0 and not args.resume:
        write_json(args.output / "recipe.json", recipe)
    if args.resume:
        old = json.loads((args.output / "recipe.json").read_text())
        if old != recipe:
            raise ValueError("resume recipe mismatch")
    if args.local_backward_only:
        # BF16 master weights reduce memory solely for the no-optimizer wiring check.
        student = student.to(device=device, dtype=torch.bfloat16)
        recompute(student, args.activation_checkpoint)
        if teacher is not None:
            teacher = teacher.to(device=device, dtype=torch.bfloat16)
    else:
        strategy, groups = topology(args.sharding, device)
        student = wrap(student, device, strategy, groups, args.activation_checkpoint)
        if teacher is not None:
            teacher = wrap(teacher, device, strategy, groups, args.activation_checkpoint, teacher=True).eval()
        checkpoints = CheckpointIO(args.sharding, groups)
    if rank == 0 and not args.local_backward_only:
        print("FSDP_READY", rank, "student", sum(isinstance(m, FSDP) for m in student.modules()),
              "teacher", sum(isinstance(m, FSDP) for m in teacher.modules()) if teacher is not None else 0,
              "world", world, "sharding", args.sharding, "recompute", args.activation_checkpoint,
              "micro", args.micro_batch, "accum", args.global_batch // (world * args.micro_batch),
              "prefetch", args.prefetch_batches, flush=True)
    visual = visual.to(device).eval()
    core = student.module if isinstance(student, FSDP) else student
    # Foundation targets have fixed source weights and no optimizer; cheap guards.
    frozen_modules = [visual] + ([teacher] if teacher is not None else [])
    optimizer = None
    start = 0
    # Teacher construction consumes RNG; reset matched per-rank training RNG
    # only after all models/FSDP wrappers are built. Resume restores saved RNG.
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if not args.local_backward_only:
        groups = {"base": [], "new": []}
        for name, p in student.named_parameters():
            if p.requires_grad:
                clean = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
                groups["new" if clean.startswith(("r.", "k.")) else "base"].append(p)
        optimizer = torch.optim.AdamW(
            [{"params": groups["base"], "lr": args.lr}, {"params": groups["new"], "lr": args.new_lr}],
            weight_decay=0.,
        )
        if args.resume:
            candidates = sorted(p.parent for p in (args.output / "checkpoints").glob("step-*/COMMITTED.json"))
            if not candidates:
                raise ValueError("no committed DCP checkpoint")
            start = checkpoints.load(candidates[-1], student, optimizer, recipe)
            if start == args.steps:
                from stage3_scale.evaluate_joint import evaluate
                evaluation_dir = args.output / "development" / f"step-{start:06d}"
                if evaluation_dir.exists() and not (evaluation_dir / "results.json").exists():
                    if rank == 0:
                        import uuid
                        os.replace(evaluation_dir, evaluation_dir.with_name(
                            evaluation_dir.name + f".interrupted-{uuid.uuid4().hex}"))
                    checkpoint_io.barrier()
                if not (evaluation_dir / "results.json").exists():
                    evaluate(student, visual, args.development_root, evaluation_dir,
                             transform, tokenizer, prompt_builder, stats, device, args.development_limit)
                if rank == 0:
                    write_json(args.output / "complete.json", {
                        "complete": True, "steps": start, "final_development_action_eval": True,
                        "sealed_test_accessed": False,
                    })
                checkpoint_io.barrier()
                dist.destroy_process_group()
                return
            # Replay after last committed step; do not leave duplicate log histories.
            if rank == 0:
                from stage3.train_causal_student import trim_logs
                trim_logs(args.output, start)
                for filename in ("progress.jsonl", "phase_events.jsonl"):
                    path = args.output / filename
                    if path.exists():
                        kept = []
                        for line in path.read_text().splitlines():
                            try:
                                if line.strip() and json.loads(line)["step"] <= start:
                                    kept.append(line)
                            except json.JSONDecodeError:
                                continue  # Interrupted final append.
                        path.write_text("\n".join(kept) + ("\n" if kept else ""))
    accum = args.global_batch // (world * args.micro_batch)
    if args.local_backward_only:
        accum = 1
    final_step = 1 if args.local_backward_only else args.steps
    # No scientific sampler changes: only CPU decoding now runs ahead of the GPU.
    batches = BatchPrefetch(dataset, args.micro_batch, args.seed, start,
                            min(final_step, args.stop_after or final_step),
                            accum, rank, not args.no_augmentation, args.prefetch_batches)
    progress = Progress(args.output, args.steps, start, args.global_batch,
                        args.save_every, args.development_every, rank == 0, args.log_every, args.first_save)
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(start + 1, final_step + 1):
        student.train()
        # Readout dropout matches original training, teacher stays eval.
        for module in frozen_modules:
            module.eval()
        if optimizer:
            for group, peak in zip(optimizer.param_groups, (args.lr, args.new_lr)):
                group["lr"] = cosine_learning_rate(step, args.steps, peak, args.warmup, .1)
            optimizer.zero_grad(set_to_none=True)
        else:
            student.zero_grad(set_to_none=True)
        timer = time.monotonic()
        profile = PhaseTimer(step == start + 1 or (
            args.profile_every > 0 and step % args.profile_every == 0), device)
        aggregates = {}
        ids = []
        for micro in range(accum):
            with profile.phase("data_wait"):
                cpu_batch = batches.get(step, micro)
            with profile.phase("host_to_device"):
                batch = to_device(cpu_batch, device)
            ids.extend(batch["identities"])
            noise_seed = args.seed + (step * accum + micro) * world + rank
            x, tau, _ = flow_draw(batch["actions"].repeat_interleave(args.fm_draws, 0), noise_seed)
            with profile.phase("frozen_visual_and_teacher"), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                targets = visual(batch["current"], batch["past"],
                                 batch["future"] if args.variant != "latent" else None)
                target_velocity = None
                if teacher is not None:
                    target_velocity = teacher(
                        batch["input_ids"], batch["attention_mask"], batch["current"],
                        targets["history_features"], targets["history_vae"],
                        targets["future_condition"], x, tau,
                    )
            # Every microstep synchronizes sharded gradients. Avoid no_sync full
            # gradient accumulation memory spikes for the 7B model.
            with profile.phase("student_forward_loss"), torch.autocast("cuda", dtype=torch.bfloat16):
                output = student(
                    batch["input_ids"], batch["attention_mask"], batch["current"],
                    targets["history_features"], targets["history_vae"], x, tau,
                )
                loss, metrics = objective(
                    output, batch["actions"], noise_seed, args.fm_draws,
                    targets.get("future_condition"), core.r.future_scale,
                    target_velocity, args.alignment_weight, args.distill_weight,
                    action_mask=batch.get("action_mask"),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite joint loss")
            with profile.phase("backward_including_collectives"):
                (loss / accum).backward()
            for key, value in metrics.items():
                aggregates[key] = aggregates.get(key, 0.) + float(value.detach()) / accum
        with profile.phase("gradient_clip"):
            norm = student.clip_grad_norm_(1.) if isinstance(student, FSDP) else torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient")
        if any(p.grad is not None for module in frozen_modules for p in module.parameters()):
            raise RuntimeError("frozen target/teacher gradient")
        if args.admission_gradient_audit and not args.local_backward_only and step in (1, start + 1):
            names = ["vlm_vision", "vlm_projector", "llm", "past", "r", "k", "dit"]
            sums = dict.fromkeys(names, 0.)
            for name, p in student.named_parameters():
                if p.grad is None:
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
                group = ("vlm_vision" if name.startswith("vlm.vision_backbone.") else
                         "vlm_projector" if name.startswith("vlm.projector.") else
                         "llm" if name.startswith("vlm.llm_backbone.") else
                         "past" if name.startswith("past_qformer.") else
                         "r" if name.startswith("r.") else "k" if name.startswith("k.") else
                         "dit" if name.startswith("net.") else None)
                if group:
                    sums[group] += float(p.grad.float().square().sum())
            values = torch.tensor([sums[n] for n in names], device=device)
            dist.all_reduce(values)
            if not torch.isfinite(values).all() or not (values > 0).all():
                raise RuntimeError(f"missing full-policy gradients: {dict(zip(names, values.tolist()))}")
            if rank == 0:
                write_json(args.output / f"gradient-audit-step{step:06d}.json",
                           {"complete": True, "step": step, "squared_norm_sum": dict(zip(names, values.tolist())),
                            "note": "Global sum includes HSDP replicas; validates nonzero/finite, not a comparable norm."})
        if args.local_backward_only:
            audit = {}
            for name, module in [("vlm_vision", core.vlm.vision_backbone), ("vlm_projector", core.vlm.projector),
                                 ("llm", core.vlm.llm_backbone), ("past", core.past_qformer),
                                 ("r", core.r), ("k", core.k), ("dit", core.net)]:
                val = sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None)
                audit[name] = val ** .5
                if val == 0:
                    raise RuntimeError(f"no joint gradient reached {name}")
            write_json(args.output / "backward_only.json", {
                "complete": True, "not_optimizer_training": True, "full_student_scope_gradients": audit,
                "losses": aggregates, "identities": ids, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                "seconds": time.monotonic() - timer, "phase_seconds": profile.values,
            })
            print("FULL_BACKWARD_OK", audit, flush=True)
            batches.close()
            progress.close()
            return
        with profile.phase("optimizer"):
            optimizer.step()
        torch.cuda.synchronize()
        numeric = torch.tensor(list(aggregates.values()), device=device)
        dist.all_reduce(numeric)
        numeric /= world
        phase_values = profile.global_max()
        memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                               torch.cuda.max_memory_reserved(device) / 2**30], device=device)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        record = {"step": step, "seconds": time.monotonic() - timer, "grad_norm": float(norm),
                  "lr": optimizer.param_groups[0]["lr"], "new_lr": optimizer.param_groups[1]["lr"],
                  "peak_allocated_gib_max": float(memory[0]), "peak_reserved_gib_max": float(memory[1]),
                  "profiled": profile.enabled, "phase_seconds_max": phase_values,
                  **dict(zip(aggregates, numeric.cpu().tolist()))}
        if rank == 0:
            with (args.output / "train.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            if args.resume and step == start + 1:
                write_json(args.output / "RESUME_VERIFIED.json", {
                    "complete": True, "restored_step": start, "first_updated_step": step,
                    "world_size": world, "checkpoint_schema": "whenami_perf_same_topology_deduplicated_v1",
                    "optimizer_and_rank_rng_loaded": True,
                    "note": "New process loaded committed state and completed a finite update; not a bitwise uninterrupted-run comparison.",
                })
            if step == start + 1 or step % args.log_every == 0:
                print("JOINT_TRAIN", record, flush=True)
        progress.update(record)
        if step % args.save_every == 0 or step in (args.steps, args.stop_after, args.first_save):
            began = progress.begin("save", step)
            checkpoints.save(args.output / "checkpoints" / f"step-{step:06d}", student, optimizer, step, recipe)
            progress.end("save", step, began)
        if step % args.development_every == 0 or step in (args.steps, args.first_eval):
            began = progress.begin("eval", step)
            from stage3_scale.evaluate_joint import evaluate
            evaluate(student, visual, args.development_root,
                     args.output / "development" / f"step-{step:06d}",
                     transform, tokenizer, prompt_builder, stats, device,
                     args.development_limit)
            progress.end("eval", step, began)
        if args.stop_after and step == args.stop_after < args.steps:
            if rank == 0:
                print("ORDERLY_STOP_FOR_RESUME", step, flush=True)
            break
        if step == args.steps and rank == 0:
            write_json(args.output / "complete.json", {
                "complete": True, "steps": step, "final_development_action_eval": True,
                "sealed_test_accessed": False,
            })
    batches.close()
    progress.close()
    checkpoint_io.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
