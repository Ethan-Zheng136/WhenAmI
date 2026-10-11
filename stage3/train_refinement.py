"""Single-device, bounded R+K experiment. No edits to frozen reference sources."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from stage3.behavior import load_behavior
from stage3.cache import EndpointCache
from stage3.common import sha256, write_json
from stage3.refinement import PastRefinement, RefinementConfig, online_inputs, refinement_loss


@torch.no_grad()
def validate(model, reader, cache, device):
    model.eval()
    totals = {}
    for start in range(0, len(cache.valid_indices), 8):
        ids = cache.valid_indices[start:start + 8]
        batch = cache.batch(ids, device)
        output = model(*online_inputs(batch))
        choices = {"online": output["online"], "past": batch["past_c"]}
        if model.mode != "adapter":
            choices["oracle"] = model.oracle(*online_inputs(batch), batch["future_c"])
        for name, condition in choices.items():
            loss = reader.loss(batch["past_actions"], batch["past_z"], condition, 424242 + start)
            totals[name + "_fm"] = totals.get(name + "_fm", 0.) + float(loss) * len(ids)
        delta = ((output["online"] - batch["past_c"]) / model.r.past_scale).square().mean()
        totals["delta_standardized_mse"] = totals.get("delta_standardized_mse", 0.) + float(delta) * len(ids)
    return {k: v / len(cache.valid_indices) for k, v in totals.items()}


def action_gradient_audit(model, reader, cache, device, seed):
    """Unweighted online FM ONLY, without alignment/oracle aux. No updates."""
    model.eval()
    batch = cache.batch(cache.train_indices[:8], device)
    output = model(*online_inputs(batch))
    loss = reader.loss(batch["past_actions"], batch["past_z"], output["online"], seed)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    gradients = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True)
    sums = {"r": 0., "k_output": 0., "k_body": 0.}
    for (name, _), g in zip(parameters, gradients):
        if g is None:
            continue
        group = "r" if name.startswith("r.") else ("k_output" if name.startswith("k.output.") else "k_body")
        sums[group] += float(g.float().square().sum())
    return {k + "_online_action_grad_norm": v ** .5 for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("adapter", "predict", "teacher"), required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alignment-weight", type=float, default=.1)
    parser.add_argument("--fm-draws", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--rollout-weight", type=float, default=0.)
    parser.add_argument("--precision", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("this isolated pilot supports one local device only")
    if min(args.max_steps, args.batch_size, args.fm_draws, args.eval_every, args.save_every) < 1 or args.lr <= 0:
        raise ValueError("positive counts/lr required")
    if args.alignment_weight < 0 or args.rollout_weight < 0 or args.rollout_steps < 0:
        raise ValueError("nonnegative objective settings required")
    if (args.rollout_steps > 0) != (args.rollout_weight > 0):
        raise ValueError("rollout weight and steps must both be positive or both zero")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoints").mkdir()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = EndpointCache(args.cache)
    config = RefinementConfig()
    model = PastRefinement(config, cache.target_statistics(), args.mode).to(device)
    reader = load_behavior(cache.manifest["endpoints"]["past"], device)
    versions = [p._version for p in reader.parameters()]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.)
    metadata = {
        "schema": "whenami_stage3_refinement_v1", "stage": 3, "mode": args.mode,
        "cache": str(args.cache.resolve()), "cache_manifest_sha256": cache.manifest_sha256,
        "model_config": config.as_dict(),
        "train_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "endpoints": cache.manifest["endpoints"],
        "notes": [
            "Only R and K optimized; frozen Past DiT reads both online and oracle conditions.",
            "Online API accepts only Past H/z/C. Oracle API explicitly receives real Future C.",
            "Teacher mode averages online/oracle FM; no behavior distillation in this first trial.",
            "Rollout auxiliary (if enabled) is online-only, pure-noise, fully differentiable.",
            "One R->K pass, NOT recurrent R/K refinement or a multi-time state timeline.",
            "Same updates/batches/initialization; different variants consume different compute.",
        ],
    }
    write_json(args.output / "config.json", metadata)

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and args.precision == "bf16" else nullcontext()

    def validation(step):
        with amp():
            record = validate(model, reader, cache, device)
        if not all(np.isfinite(v) for v in record.values()):
            raise FloatingPointError("nonfinite validation")
        if step == 0:
            if record["delta_standardized_mse"] != 0. or record["online_fm"] != record["past_fm"]:
                raise RuntimeError("zero-initialized K did not preserve original Past behavior")
        with (args.output / "validation.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **record}) + "\n")
        print("VALID", step, record, flush=True)

    def audit(step):
        with amp():
            record = action_gradient_audit(model, reader, cache, device, args.seed + 707)
        with (args.output / "gradient_audits.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **record}) + "\n")
        if not all(np.isfinite(v) for v in record.values()):
            raise FloatingPointError("nonfinite audited gradients")
        if step > 0 and args.mode != "adapter" and record["r_online_action_grad_norm"] <= 0:
            raise RuntimeError("online action loss failed to reach R after K output update")
        print("ACTION_GRAD", step, record, flush=True)

    validation(0)
    audit(0)
    print("PARAMETERS", {prefix: sum(p.numel() for p in module.parameters() if p.requires_grad)
                         for prefix, module in [("r", model.r), ("k", model.k)]}, flush=True)
    for step in range(1, args.max_steps + 1):
        model.train()
        ids = np.random.default_rng(args.seed + step).choice(cache.train_indices, args.batch_size, replace=True)
        batch = cache.batch(ids, device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        with amp():
            loss, metrics = refinement_loss(
                model, reader, batch, args.seed + step, args.alignment_weight,
                args.fm_draws, args.rollout_steps, args.rollout_weight,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if reader.training or reader.net.training or any(p.grad is not None for p in reader.parameters()):
            raise RuntimeError("frozen reader trained or accumulated gradients")
        if [p._version for p in reader.parameters()] != versions:
            raise RuntimeError("frozen reader modified")
        if device.type == "cuda":
            torch.cuda.synchronize()
        values = {k: float(v.detach()) for k, v in metrics.items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError("nonfinite metric")
        record = {"step": step, "seconds": time.monotonic() - start, "grad_norm": float(norm), **values}
        with (args.output / "train.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if step == 1 or step % 50 == 0:
            print("TRAIN", record, flush=True)
        if step in (1, 10, 100) or step == args.max_steps:
            audit(step)
        if step % args.eval_every == 0 or step == args.max_steps:
            validation(step)
        if step % args.save_every == 0 or step == args.max_steps:
            path = args.output / "checkpoints" / f"step-{step:06d}.pt"
            temp = path.with_suffix(".tmp")
            torch.save({
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(), "metadata": metadata, "step": step,
                "torch_rng_state": torch.get_rng_state(),
            }, temp)
            os.replace(temp, path)
    write_json(args.output / "complete.json", {
        "complete": True, "steps": args.max_steps, "frozen_reader_unchanged": True,
        "checkpoint": str(args.output / "checkpoints" / f"step-{args.max_steps:06d}.pt"),
    })


if __name__ == "__main__":
    main()
