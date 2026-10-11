"""Train ONLY the Stage-3 connector; frozen endpoints never change.

Same initialization, forward minibatches, forward noise draws and optimizer
step count for --mode forward and --mode bidirectional. Bidirectional has extra
backward compute/gradients; wall-clock and gradient budgets are NOT identical.
Saves small connector+optimizer checkpoints. No Stage1/2 files are overwritten.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from stage3.common import write_json, sha256
from stage3.cache import EndpointCache
from stage3.connector import ConnectorConfig, SharedConnector
from stage3.behavior import load_behavior
from stage3.objectives import connector_inputs, directional_losses, validate_conditions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=("forward", "bidirectional"), default="bidirectional")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8, help="per-rank minibatch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--alignment-weight", type=float, default=1.)
    ap.add_argument("--action-weight", type=float, default=1.)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-limit", type=int, default=256)
    ap.add_argument("--precision", choices=("float32", "bf16"), default="bf16")
    ap.add_argument("--checkpoint-readers", action="store_true")
    args = ap.parse_args()
    if min(args.max_steps, args.batch_size, args.save_every, args.eval_every, args.eval_limit) < 1 or args.lr <= 0:
        raise ValueError("steps/batch/interval/lr must be positive")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available() else torch.device("cpu")
    if world > 1:
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cache = EndpointCache(args.cache)
    # Rank zero alone reserves the path; a per-rank exists() check races with
    # rank zero's mkdir on a shared filesystem.
    reservation = [None]
    if rank == 0:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "checkpoints").mkdir()
        except OSError as error:
            reservation[0] = f"Cannot reserve new run directory: {error}"
    if world > 1:
        dist.broadcast_object_list(reservation, src=0)
    if reservation[0]:
        raise FileExistsError(reservation[0])
    config = ConnectorConfig(hidden=args.hidden, layers=args.layers)
    connector = SharedConnector(config, cache.target_statistics()).to(device)
    if args.mode == "forward":
        connector.freeze_backward_only_adapters()
    core = connector
    if world > 1:
        connector = DistributedDataParallel(connector, device_ids=[device.index], find_unused_parameters=False)
    readers = {
        side: load_behavior(cache.manifest["endpoints"][side], device, args.checkpoint_readers)
        for side in (["future"] if args.mode == "forward" else ["future", "past"])
    }
    # Frozen means parameters excluded from optimizer, eval mode preserved, but
    # autograd through reader inputs remains active.
    optimizer = torch.optim.AdamW(
        [p for p in core.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.
    )
    reader_versions = {side: [p._version for p in reader.parameters()] for side, reader in readers.items()}
    metadata = {
        "schema": "whenami_stage3_connector_v1", "stage": 3, "mode": args.mode,
        "cache": str(args.cache.resolve()), "cache_manifest_sha256": cache.manifest_sha256,
        "connector": config.as_dict(),
        "train_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "world_size": world, "global_batch": args.batch_size * world,
        "endpoints": cache.manifest["endpoints"], "reference_source_sha256": cache.manifest["reference_source_sha256"],
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "notes": [
            "Only connector parameters optimized; encoder/VLM features cached and fixed.",
            "R has no action-label/noisy-action/target-side feature input.",
            "Two frozen DiTs use their OWN z and normalized action labels.",
            "Frozen readout eval mode disables stochastic condition dropout.",
            "Forward-only and bidirectional match forward updates, not total compute.",
            "No Lambda, cycle, multi-time state rollout or Stage4 integration in this pilot.",
        ],
    }
    if rank == 0:
        write_json(args.output / "config.json", metadata)
        print(f"TRAIN {args.mode}: connector={sum(p.numel() for p in core.parameters())/1e6:.2f}M, "
              f"global_batch={args.batch_size * world}, train={len(cache.train_indices)}, "
              f"validation={len(cache.valid_indices)}", flush=True)

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and args.precision == "bf16" else nullcontext()

    def log_validation(step):
        if rank == 0:
            with amp():
                values = validate_conditions(core, readers, cache, device, args.batch_size, args.eval_limit)
            if not all(np.isfinite(v) for v in values.values()):
                raise FloatingPointError("nonfinite validation metric")
            with (args.output / "validation.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, **values}) + "\n")
            print("VALID", step, values, flush=True)
        if world > 1:
            dist.barrier()

    try:
        log_validation(0)
        for step in range(1, args.max_steps + 1):
            core.train()
            # Draw the same global batch for matched controls; no hidden target
            # or validation-index access in connector_inputs.
            rng = np.random.default_rng(args.seed + step)
            global_ids = rng.choice(cache.train_indices, size=args.batch_size * world, replace=True)
            ids = global_ids[rank * args.batch_size:(rank + 1) * args.batch_size]
            batch = cache.batch(ids, device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.monotonic()
            with amp():
                predictions = connector(*connector_inputs(batch, args.mode))
                loss, metrics = directional_losses(
                    core, predictions, batch, readers, args.seed + step * world + rank,
                    args.alignment_weight, args.action_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(core.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            for side, reader in readers.items():
                if reader.training or reader.net.training:
                    raise RuntimeError("frozen reader entered training mode")
                if any(p.grad is not None for p in reader.parameters()):
                    raise RuntimeError("gradient accumulated on a frozen reader parameter")
                if [p._version for p in reader.parameters()] != reader_versions[side]:
                    raise RuntimeError("frozen reader parameters were modified")
            if device.type == "cuda":
                torch.cuda.synchronize()
            numeric = {k: v.detach().float() for k, v in metrics.items()}
            if world > 1:
                for value in numeric.values():
                    dist.all_reduce(value)
                    value.div_(world)
            if rank == 0:
                record = {
                    "step": step, "seconds": time.monotonic() - start, "grad_norm": float(grad_norm),
                    **{k: float(v) for k, v in numeric.items()},
                }
                with (args.output / "train.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                if step == 1 or step % 25 == 0:
                    print("TRAIN", record, flush=True)
            if step % args.eval_every == 0 or step == args.max_steps:
                log_validation(step)
            if step % args.save_every == 0 or step == args.max_steps:
                if rank == 0:
                    path = args.output / "checkpoints" / f"step-{step:06d}.pt"
                    temporary = path.with_suffix(".tmp")
                    torch.save({
                        "connector": {k: v.detach().cpu() for k, v in core.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "step": step, "metadata": metadata,
                        "torch_rng_state": torch.get_rng_state(),
                    }, temporary)
                    os.replace(temporary, path)
                if world > 1:
                    dist.barrier()
        if rank == 0:
            write_json(args.output / "complete.json", {
                "complete": True, "steps": args.max_steps, "mode": args.mode,
                "frozen_readers_unchanged": True,
                "checkpoint": str(args.output / "checkpoints" / f"step-{args.max_steps:06d}.pt"),
            })
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
