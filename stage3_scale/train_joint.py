"""Raw full-policy FSDP training with explicit frozen Future teacher.

This is a NEW joint-adaptation experiment, not continuation of the cached20k
student. Original checkpoints unchanged. Only committed distributed checkpoints
can resume. The default variant has no teacher velocity cost; distill adds it.
"""

import argparse
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
)

from stage3.common import BASE, DEFAULT_PAST, DEFAULT_FUTURE, endpoint_record, sha256, write_json, source_fingerprint
from stage3.behavior import flow_draw
from stage3_scale.model import load_joint, objective
from stage3_scale.raw_data import RawWindows
from stage3_scale import checkpoint_v2 as checkpoint_io
from stage3.causal_student import cosine_learning_rate


def wrap_fsdp(model, device, teacher=False):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.bert.modeling_bert import BertLayer
    from timm.models.vision_transformer import Block
    from stage3.reader_adaptation import AdaptedBlock
    classes = {LlamaDecoderLayer, BertLayer, Block, AdaptedBlock,
               torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer}
    policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=classes)
    model = FSDP(
        model, auto_wrap_policy=policy, use_orig_params=True, device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                       buffer_dtype=torch.float32),
        limit_all_gathers=True,
    )
    if not teacher:
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, tuple(classes)),
        )
    return model


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
    args = parser.parse_args()
    world, rank, local = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if args.local_backward_only and world != 1:
        raise ValueError("backward-only is local diagnostic")
    if args.global_batch % (world * args.micro_batch) or min(
        args.steps, args.micro_batch, args.fm_draws, args.save_every, args.development_every
    ) < 1:
        raise ValueError("global batch must divide world*micro and all counts be positive")
    if not 0 <= args.warmup < args.steps or args.stop_after > args.steps or min(
        args.stop_after, args.development_limit, args.alignment_weight, args.distill_weight
    ) < 0 or min(args.lr, args.new_lr) <= 0:
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
    student, visual, teacher, transform, tokenizer, prompt_builder = load_joint(args.seed, args.variant == "distill_future")
    stats = json.loads((DEFAULT_PAST.parents[1] / "dataset_statistics.json").read_text())
    index_header = json.loads(args.data_index.read_text())
    if index_header.get("data_mix") == "oxe_magic_soup_plus":
        from stage3_scale.oxe_data import OXERawWindows, oxe_source_hashes
        if index_header["oxe_source_sha256"] != oxe_source_hashes():
            raise ValueError("OXE standardization changed since preparation")
        dataset = OXERawWindows(args.data_index, transform, tokenizer, prompt_builder, stats, args.allow_partial_data)
    else:
        dataset = RawWindows(args.data_index, transform, tokenizer, prompt_builder, stats, args.allow_partial_data)
    # Bind data and every relevant implementation, not merely the small index path.
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("resume", "stop_after")}
    recipe = {
        "schema": "whenami_joint_raw_fsdp_v1", "settings": settings, "world_size": world,
        "data_index_sha256": sha256(args.data_index),
        "past": endpoint_record(DEFAULT_PAST), "future": endpoint_record(DEFAULT_FUTURE),
        "reference_source": source_fingerprint(),
        "joint_source": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
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
                  "All valid interior windows in TRAIN-role source episodes; dev/test excluded.",
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
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        apply_activation_checkpointing(
            student, checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, LlamaDecoderLayer),
        )
        if teacher is not None:
            teacher = teacher.to(device=device, dtype=torch.bfloat16)
    else:
        student = wrap_fsdp(student, device)
        if teacher is not None:
            teacher = wrap_fsdp(teacher, device, teacher=True).eval()
    if args.admission_gradient_audit and not args.local_backward_only:
        print("FSDP_READY", rank, "student", sum(isinstance(m, FSDP) for m in student.modules()),
              "teacher", sum(isinstance(m, FSDP) for m in teacher.modules()) if teacher is not None else 0, flush=True)
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
            start = checkpoint_io.load(candidates[-1], student, optimizer, recipe)
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
    accum = args.global_batch // (world * args.micro_batch)
    if args.local_backward_only:
        accum = 1
    final_step = 1 if args.local_backward_only else args.steps
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
        aggregates = {}
        ids = []
        for micro in range(accum):
            batch = dataset.batch(args.micro_batch, args.seed, step, micro, rank, device,
                                  augment=not args.no_augmentation)
            ids.extend(batch["identities"])
            noise_seed = args.seed + (step * accum + micro) * world + rank
            x, tau, _ = flow_draw(batch["actions"].repeat_interleave(args.fm_draws, 0), noise_seed)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
            (loss / accum).backward()
            for key, value in metrics.items():
                aggregates[key] = aggregates.get(key, 0.) + float(value.detach()) / accum
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
                           {"complete": True, "step": step, "squared_norm_sum": dict(zip(names, values.tolist()))})
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
            })
            print("FULL_BACKWARD_OK", audit, flush=True)
            return
        optimizer.step()
        torch.cuda.synchronize()
        numeric = torch.tensor(list(aggregates.values()), device=device)
        dist.all_reduce(numeric)
        numeric /= world
        record = {"step": step, "seconds": time.monotonic() - timer, "grad_norm": float(norm),
                  "lr": optimizer.param_groups[0]["lr"],
                  **dict(zip(aggregates, numeric.cpu().tolist()))}
        if rank == 0:
            with (args.output / "train.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            if step == 1 or step % 10 == 0:
                print("JOINT_TRAIN", record, flush=True)
        if step % args.save_every == 0 or step == args.steps or step == args.stop_after:
            checkpoint_io.save(args.output / "checkpoints" / f"step-{step:06d}", student, optimizer, step, recipe)
        if args.stop_after and step == args.stop_after < args.steps:
            if rank == 0:
                print("ORDERLY_STOP_FOR_RESUME", step, flush=True)
            break
        if step % args.development_every == 0 or step == args.steps:
            from stage3_scale.evaluate_joint import evaluate
            evaluate(student, visual, args.development_root,
                     args.output / "development" / f"step-{step:06d}",
                     transform, tokenizer, prompt_builder, stats, device,
                     args.development_limit)
        if step == args.steps and rank == 0:
            write_json(args.output / "complete.json", {
                "complete": True, "steps": step, "final_development_action_eval": True,
                "sealed_test_accessed": False,
            })
    checkpoint_io.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
