"""20k matched causal-student experiments, final evaluation and explicit resume.

The student starts from the same-seed adaptive_mean5k for every variant.
This is a model-only warm start, not continuation of the parent's optimizer.
"""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from stage3.cache import EndpointCache
from stage3.common import BASE, sha256, write_json
from stage3.causal_student import (
    VARIANTS, CausalStudent, student_loss, past_inputs,
    cosine_learning_rate, load_reader_checkpoint,
)
from stage3.evaluate import denormalize

sys.path.insert(0, str(BASE / "evaluation"))
from past_protocol import sample_seed, action_metrics

SOURCE_FILES = (
    "causal_student.py", "train_causal_student.py", "reader_adaptation.py",
    "oracle_evidence.py", "refinement.py", "connector.py", "behavior.py", "cache.py",
)


def source_hashes():
    return {name: sha256(Path(__file__).parent / name) for name in SOURCE_FILES}


def trim_logs(output, step):
    """An explicit resume replays updates after its checkpoint; remove only those log rows."""
    for name in ("train.jsonl", "development.jsonl", "gradient_audits.jsonl"):
        path = output / name
        if not path.exists():
            continue
        keep = []
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # Interrupted final write.
            if row["step"] <= step:
                keep.append(json.dumps(row))
        temp = path.with_suffix(".tmp")
        temp.write_text("\n".join(keep) + ("\n" if keep else ""))
        os.replace(temp, path)


def append_json(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


@torch.no_grad()
def validate(student, teacher, cache, device, variant, draws, align_weight, distill_weight, limit):
    student.eval()
    ids = cache.valid_indices[:limit] if limit else cache.valid_indices
    totals = {}
    for start in range(0, len(ids), 32):
        selected = ids[start:start + 32]
        _, metrics = student_loss(student, teacher, cache.batch(selected, device), variant,
                                  424242 + start, draws, align_weight, distill_weight)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + float(value) * len(selected)
    return {"n": len(ids), **{k: v / len(ids) for k, v in totals.items()}}


def gradient_audit(student, batch, seed):
    """Action FM alone; no future alignment or teacher supervision."""
    loss, _ = student_loss(student, None, batch, "latent", seed, draws=1)
    named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True)
    sums = {"r": 0., "k": 0., "reader_adapters": 0.}
    for (name, _), grad in zip(named, gradients):
        if grad is not None:
            key = "r" if name.startswith("r.") else ("k" if name.startswith("policy.probe.") else "reader_adapters")
            sums[key] += float(grad.float().square().sum())
    return {key + "_action_grad_norm": value ** .5 for key, value in sums.items()}


@torch.no_grad()
def evaluate(student, cache, device, output, checkpoint_hash, seeds, limit, resume):
    student.eval()
    raw = np.load(cache.path / "raw_actions.npy", mmap_mode="r")
    for noise_seed in seeds:
        root = output / f"development_eval_noise{noise_seed}"
        final = root / "results.json"
        if final.exists():
            report = json.loads(final.read_text())
            if not resume or not report["complete"] or report["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("existing evaluation does not match explicit resume")
            continue
        root.mkdir(exist_ok=resume)
        results = []
        for dataset in sorted(set(cache.datasets.tolist())):
            ids = cache.valid_indices[cache.datasets[cache.valid_indices] == dataset]
            if limit:
                ids = ids[:limit]
            predictions = []
            for i in ids:
                batch = cache.batch([i], device)
                seed = sample_seed(dataset, cache.episodes[i], cache.timesteps[i], noise_seed)
                # Only Past inputs cross the deployable interface.
                predictions.append(student.sample(*past_inputs(batch), seed).cpu().numpy()[0])
            stats = cache.manifest["endpoints"]["past"]["statistics"][dataset]["action"]
            pred = denormalize(np.stack(predictions), stats)
            target = np.asarray(raw[ids])
            metrics = action_metrics(pred, target, stats)
            np.savez_compressed(
                root / f"{dataset}_online.npz", predictions=pred, target=target,
                indices=ids, episode=cache.episodes[ids], timestep=cache.timesteps[ids],
            )
            results.append({"dataset": dataset, **metrics})
            print("DEV_ACTION", noise_seed, results[-1], flush=True)
        write_json(final, {
            "complete": True, "split": "development", "causal": True,
            "noise_seed": noise_seed, "limit_per_dataset": limit, "precision": "float32",
            "checkpoint_sha256": checkpoint_hash, "cache_manifest_sha256": cache.manifest_sha256,
            "results": results,
            "notes": ["Student inference only Past H/z/C.", "No real Future or teacher at inference.",
                      "10-point/9-interval Euler sampler; sealed test never opened."],
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--readers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fm-draws", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--lr-floor-ratio", type=float, default=.1)
    parser.add_argument("--alignment-weight", type=float, default=.1)
    parser.add_argument("--distill-weight", type=float, default=.3)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-noise-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--development-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--precision", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--resume", action="store_true", help="resume this output's latest committed checkpoint")
    parser.add_argument("--stop-after", type=int, default=0, help="test-only orderly interruption, schedule budget unchanged")
    args = parser.parse_args()
    if min(args.max_steps, args.batch_size, args.fm_draws, args.save_every, args.eval_every) < 1 or args.lr <= 0:
        raise ValueError("positive counts/lr required")
    if not 0 <= args.warmup < args.max_steps or not 0 < args.lr_floor_ratio <= 1:
        raise ValueError("invalid schedule")
    if min(args.alignment_weight, args.distill_weight, args.development_limit, args.validation_limit, args.stop_after) < 0:
        raise ValueError("nonnegative objective weights/limits required")
    if len(set(args.eval_noise_seeds)) != len(args.eval_noise_seeds) or args.stop_after > args.max_steps:
        raise ValueError("duplicate noise seeds or invalid stop-after")
    if (args.cache / "SEALED_DO_NOT_EVALUATE").exists():
        raise ValueError("sealed test prohibited")
    if (args.output / "complete.json").exists():
        raise FileExistsError("run already complete")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ("resume", "stop_after")}
    if args.resume:
        old_meta = json.loads((args.output / "config.json").read_text())
        if old_meta["settings"] != settings or old_meta["source_sha256"] != source_hashes():
            raise ValueError("resume requires identical settings and training sources")
        choices = sorted((args.output / "checkpoints").glob("step-*.pt"))
        if not choices:
            raise ValueError("no committed checkpoint to resume")
        resume_cp = torch.load(choices[-1], weights_only=True, map_location="cpu")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "checkpoints").mkdir()
        resume_cp = None
    cache = EndpointCache(args.cache)
    if cache.manifest.get("reader_study", {}).get("role") != "train_dev":
        raise ValueError("requires Reader Study train_dev cache")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    base_path = args.readers / f"adaptive_mean_seed{args.seed}" / "checkpoints/step-005000.pt"
    policy, base_record = load_reader_checkpoint(base_path, cache, device, expected_evidence="mean")
    if base_record["seed"] != args.seed:
        raise ValueError("base seed mismatch")
    torch.manual_seed(args.seed)
    student = CausalStudent(policy).to(device)
    teacher = None
    teacher_record = None
    if args.variant.startswith("distill_"):
        evidence = "future" if args.variant == "distill_future" else "mean"
        path = args.readers / f"adaptive_{evidence}_seed{args.seed}" / "checkpoints/step-005000.pt"
        teacher, teacher_record = load_reader_checkpoint(path, cache, device, frozen=True, expected_evidence=evidence)
        if teacher_record["seed"] != args.seed:
            raise ValueError("teacher seed mismatch")
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.)
    frozen = [("base." + name, p) for name, p in student.policy.base_parameters()]
    if teacher is not None:
        frozen += [("teacher." + name, p) for name, p in teacher.named_parameters()]
    versions = {name: p._version for name, p in frozen}
    meta = {
        "schema": "whenami_causal_student_v1", "settings": settings, "source_sha256": source_hashes(),
        "cache_manifest_sha256": cache.manifest_sha256, "base": base_record, "teacher": teacher_record,
        "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad),
        "notes": [
            "Same-seed adaptive_mean5k MODEL-only warm start for all variants; optimizer freshly initialized.",
            "20k NEW updates in default run, not cumulative20k including parent5k.",
            "Latent control identical capacity; no per-sample Future loss/teacher.",
            "Distill-past controls generic distillation; distill-future isolates privileged teacher evidence.",
            "Teacher velocities share student's noisy actions/time; teacher output stopgrad.",
            "Independent FM draws per observation, not recurrent state rollout or iterative R/K.",
            "No Stage4 environment integration or closed-loop evaluation; no sealed test access.",
        ],
    }
    start_step = 0
    if args.resume:
        for key in ("cache_manifest_sha256", "base", "teacher"):
            if meta[key] != old_meta[key]:
                raise ValueError(f"resume provenance mismatch {key}")
        student.load_compact_state(resume_cp["state"])
        optimizer.load_state_dict(resume_cp["optimizer"])
        torch.set_rng_state(resume_cp["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume_cp["cuda_rng_state"], device)
        start_step = resume_cp["step"]
        trim_logs(args.output, start_step)
        append_json(args.output / "resume_history.jsonl", {
            "at_utc": datetime.now(timezone.utc).isoformat(), "step": start_step,
        })
    else:
        write_json(args.output / "config.json", meta)
    trainable = [p for p in student.parameters() if p.requires_grad]

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and args.precision == "bf16" else nullcontext()

    def check_frozen():
        if student.policy.net.training or (teacher is not None and teacher.training):
            raise RuntimeError("reader/teacher entered train mode")
        if any(p.requires_grad or p.grad is not None or p._version != versions[name] for name, p in frozen):
            raise RuntimeError("original reader or teacher modified/received gradients")

    def validation(step):
        with amp():
            row = validate(student, teacher, cache, device, args.variant, args.fm_draws,
                           args.alignment_weight, args.distill_weight, args.validation_limit)
        append_json(args.output / "development.jsonl", {"step": step, **row})
        print("DEV", step, row, flush=True)

    def audit(step):
        with amp():
            row = gradient_audit(student, cache.batch(cache.train_indices[:8], device), args.seed + 707)
        if not all(np.isfinite(v) and v > 0 for v in row.values()):
            raise RuntimeError(f"action gradient did not reach R/K/reader adapters: {row}")
        append_json(args.output / "gradient_audits.jsonl", {"step": step, **row})
        print("GRADIENT", step, row, flush=True)

    def save(step):
        path = args.output / "checkpoints" / f"step-{step:06d}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save({
            "state": student.compact_state(), "optimizer": optimizer.state_dict(), "metadata": meta,
            "step": step, "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, temporary)
        os.replace(temporary, path)
        return path

    if not args.resume:
        validation(0)
        audit(0)
    for step in range(start_step + 1, args.max_steps + 1):
        student.train()
        lr = cosine_learning_rate(step, args.max_steps, args.lr, args.warmup, args.lr_floor_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr
        ids = np.random.default_rng(args.seed + step).choice(cache.train_indices, args.batch_size, replace=True)
        batch = cache.batch(ids, device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        before = time.monotonic()
        with amp():
            loss, metrics = student_loss(student, teacher, batch, args.variant, args.seed + step,
                                         args.fm_draws, args.alignment_weight, args.distill_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        check_frozen()
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {"step": step, "lr": lr, "seconds": time.monotonic() - before, "grad_norm": float(norm),
               **{k: float(v.detach()) for k, v in metrics.items()}}
        append_json(args.output / "train.jsonl", row)
        if step == 1 or step % 100 == 0:
            print("TRAIN", row, flush=True)
        if step in (1, 100) or step % 5000 == 0:
            audit(step)
        if step % args.eval_every == 0 or step == args.max_steps:
            validation(step)
        if step % args.save_every == 0 or step == args.max_steps or step == args.stop_after:
            save(step)
        if args.stop_after and step == args.stop_after < args.max_steps:
            print("ORDERLY_TEST_STOP", step, flush=True)
            return
    path = args.output / "checkpoints" / f"step-{args.max_steps:06d}.pt"
    cp = torch.load(path, weights_only=True, map_location="cpu")
    student.load_compact_state(cp["state"])
    for module in (student.r, student.policy.probe, *[b.adapter for b in student.policy.net.blocks]):
        if not all(torch.isfinite(v).all() for v in module.state_dict().values() if v.is_floating_point()):
            raise FloatingPointError("checkpoint contains nonfinite weights")
    # compact reload intentionally updates only student trainable module versions.
    check_frozen()
    evaluate(student.float(), cache, device, args.output, sha256(path),
             args.eval_noise_seeds, args.development_limit, args.resume)
    write_json(args.output / "complete.json", {
        "complete": True, "steps": args.max_steps, "variant": args.variant,
        "causal_student": True, "teacher_and_original_reader_unchanged": True,
        "sealed_test_accessed": False, "checkpoint": str(path),
        "evaluation_noise_seeds": args.eval_noise_seeds,
    })
    print("CAUSAL_STUDENT_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
