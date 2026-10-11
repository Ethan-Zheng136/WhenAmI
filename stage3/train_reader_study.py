"""Independent 2x2 reader adaptation runs, train/dev only; no sealed-test CLI."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import sys

import numpy as np
import torch

from stage3.behavior import load_behavior
from stage3.cache import EndpointCache
from stage3.common import BASE, sha256, write_json
from stage3.evaluate import denormalize
from stage3.reader_adaptation import ReaderStudyModel

sys.path.insert(0, str(BASE / "evaluation"))
from past_protocol import sample_seed, action_metrics


@torch.no_grad()
def development_fm(model, cache, device):
    model.eval()
    total = 0.
    for start in range(0, len(cache.valid_indices), 32):
        ids = cache.valid_indices[start:start + 32]
        total += float(model(cache.batch(ids, device), 424242 + start)) * len(ids)
    return {"n": len(cache.valid_indices), "action_fm": total / len(cache.valid_indices)}


@torch.no_grad()
def evaluate_development(model, cache, device, output, seed=42, limit=0):
    model.eval()
    output.mkdir(exist_ok=False)
    raw = np.load(cache.path / "raw_actions.npy", mmap_mode="r")
    records = []
    for name in sorted(set(cache.datasets.tolist())):
        ids = cache.valid_indices[cache.datasets[cache.valid_indices] == name]
        if limit:
            ids = ids[:limit]
        variants = ["evidence"] if model.evidence == "mean" else ["evidence", "mean_intervention"]
        predictions = {v: [] for v in variants}
        for i in ids:
            batch = cache.batch([i], device)
            noise = sample_seed(name, cache.episodes[i], cache.timesteps[i], seed)
            for variant in variants:
                predictions[variant].append(
                    model.sample(batch, noise, mean_intervention=variant == "mean_intervention").cpu().numpy()[0]
                )
        stats = cache.manifest["endpoints"]["past"]["statistics"][name]["action"]
        for variant in variants:
            pred = denormalize(np.stack(predictions[variant]), stats)
            target = np.asarray(raw[ids])
            metrics = action_metrics(pred, target, stats)
            np.savez_compressed(
                output / f"{name}_{variant}.npz", predictions=pred, target=target, indices=ids,
                episode=cache.episodes[ids], timestep=cache.timesteps[ids],
            )
            records.append({"dataset": name, "variant": variant, **metrics})
            print("DEV_ACTION", records[-1], flush=True)
    write_json(output / "results.json", {
        "complete": True, "split": "development", "noise_seed": seed,
        "precision": "float32", "limit_per_dataset": limit,
        "cache_manifest_sha256": cache.manifest_sha256,
        "evidence": model.evidence, "adaptive": model.adaptive, "results": records,
        "notes": [
            "Real future evidence is oracle-only, not deployable.",
            "Original Past encoder/VLM/DiT weights frozen; optional in-DiT adapters learned on train only.",
            "Physical sealed test directory is not opened by this entry point.",
        ],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", choices=("mean", "future"), required=True)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--precision", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--development-limit", type=int, default=0, help="0=all; use positive values only for smoke")
    args = parser.parse_args()
    if min(args.max_steps, args.batch_size, args.adapter_rank, args.eval_every, args.save_every) < 1 or args.lr <= 0:
        raise ValueError("positive step/batch/rank/interval/lr required")
    if (args.cache / "SEALED_DO_NOT_EVALUATE").exists():
        raise ValueError("refusing sealed test data")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoints").mkdir()
    cache = EndpointCache(args.cache)
    role = cache.manifest.get("reader_study", {}).get("role")
    if role is not None and role != "train_dev":
        raise ValueError("only train_dev cache accepted")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # Reader construction must not perturb paired K/adapter initialization.
    reader = load_behavior(cache.manifest["endpoints"]["past"], device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ReaderStudyModel(
        reader.net, cache.target_statistics(), args.evidence, args.adaptive, args.adapter_rank,
    ).to(device)
    del reader
    frozen = model.base_parameters()
    base_versions = {name: p._version for name, p in frozen}
    adapter_initial = [p.detach().clone() for p in model.adapter_parameters()]
    groups = [{"params": list(model.probe.parameters()), "lr": args.lr}]
    if args.adaptive:
        groups.append({"params": model.adapter_parameters(), "lr": args.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.)
    meta = {
        "schema": "whenami_reader_study_v1", "cache": str(cache.path.resolve()),
        "cache_manifest_sha256": cache.manifest_sha256,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_config": model.config.as_dict(), "endpoints": cache.manifest["endpoints"],
        "probe_trainable_parameters": sum(p.numel() for p in model.probe.parameters()),
        "reader_adapter_parameters": sum(p.numel() for p in model.adapter_parameters()),
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "notes": [
            "2x2 factorial: real Future/mean x frozen/adaptive readout; each row has matched architecture.",
            "Adaptive does not update original weights: rank-r residual after every original DiT block.",
            "Future encoder/VLM/Past encoder frozen; training only sees fresh training partition.",
            "Development monitored; sealed test physically excluded, no evaluation entry provided.",
            "Random-time action FM only, no forecast R, alignment, distillation or rollout auxiliary.",
        ],
    }
    write_json(args.output / "config.json", meta)

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and args.precision == "bf16" else nullcontext()

    def check_base():
        if any(p.requires_grad or p.grad is not None or p._version != base_versions[name] for name, p in frozen):
            raise RuntimeError("original reader parameters modified/received gradients")
        if model.net.training:
            raise RuntimeError("original reader must stay in eval mode")
        if not args.adaptive:
            for initial, p in zip(adapter_initial, model.adapter_parameters()):
                torch.testing.assert_close(initial, p, rtol=0, atol=0)

    def log_validation(step):
        with amp():
            values = development_fm(model, cache, device)
        if not np.isfinite(values["action_fm"]):
            raise FloatingPointError("nonfinite development loss")
        with (args.output / "development.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **values}) + "\n")
        print("DEV", step, values, flush=True)

    log_validation(0)
    for step in range(1, args.max_steps + 1):
        model.train()
        ids = np.random.default_rng(args.seed + step).choice(cache.train_indices, args.batch_size, replace=True)
        batch = cache.batch(ids, device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        with amp():
            loss = model(batch, args.seed + step)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        check_base()
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {"step": step, "seconds": time.monotonic() - start, "action_fm": float(loss.detach()), "grad_norm": float(norm)}
        with (args.output / "train.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step == 1 or step % 100 == 0:
            print("TRAIN", row, flush=True)
        if step % args.eval_every == 0 or step == args.max_steps:
            log_validation(step)
        if step % args.save_every == 0 or step == args.max_steps:
            path = args.output / "checkpoints" / f"step-{step:06d}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save({"state": model.compact_state(), "metadata": meta, "step": step,
                        "optimizer": optimizer.state_dict()}, temporary)
            os.replace(temporary, path)
    # Strict reload the prespecified FINAL checkpoint; do not select by development.
    cp = torch.load(path, weights_only=True, map_location="cpu")
    model.load_compact_state(cp["state"])
    check_base()
    evaluate_development(model.float(), cache, device, args.output / "development_eval", 42, args.development_limit)
    write_json(args.output / "complete.json", {
        "complete": True, "steps": args.max_steps, "original_reader_unchanged": True,
        "sealed_test_accessed": False, "checkpoint": str(path),
        "reader_adapter_updated": any(not torch.equal(a, p) for a, p in zip(adapter_initial, model.adapter_parameters())),
    })
    print("READER_STUDY_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
