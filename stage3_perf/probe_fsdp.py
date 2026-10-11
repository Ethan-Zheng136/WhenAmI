"""Tiny multi-rank restore proof; CPU tests emulate two nodes, accelerator runs don't."""

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

from stage3_perf.checkpoint import CheckpointIO
from stage3_perf.runtime import topology


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--sharding", choices=("full", "hybrid", "hybrid_zero2"), default="hybrid")
    a = p.parse_args()
    if a.cpu:
        os.environ["LOCAL_WORLD_SIZE"] = "2"
        device = torch.device("cpu")
    else:
        local = int(os.environ["LOCAL_RANK"])
        if local >= torch.cuda.device_count():
            raise ValueError("one physical accelerator per rank required")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    dist.init_process_group("gloo" if a.cpu else "nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    strategy, groups = topology(a.sharding, device)
    io = CheckpointIO(a.sharding, groups)
    config = {"test": "deduplicated_hybrid", "world": world, "sharding": a.sharding}

    def build():
        torch.manual_seed(42)
        net = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 1))
        return FSDP(net, device_id=device, use_orig_params=True,
                    sharding_strategy=strategy, process_group=groups,
                    mixed_precision=None if a.cpu else MixedPrecision(
                        param_dtype=torch.bfloat16, reduce_dtype=torch.float32))

    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    x = torch.randn(4, 8, device=device) + rank

    def update(m, optimizer):
        optimizer.zero_grad()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=not a.cpu):
            m(x).float().square().mean().backward()
        m.clip_grad_norm_(1.)
        optimizer.step()

    update(model, opt)
    # Replicated shards must be equal before optimizer deduplication is valid.
    if groups:
        flattened = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
        replicas = [torch.empty_like(flattened) for _ in range(dist.get_world_size(groups[1]))]
        dist.all_gather(replicas, flattened, group=groups[1])
        for r in replicas:
            torch.testing.assert_close(r, flattened, rtol=0, atol=0)
    torch.manual_seed(555 + rank)
    io.save(a.output / "step-1", model, opt, 1, config)
    expected_rng = torch.rand(3, device=device)
    restored = build()
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.001)
    assert io.load(a.output / "step-1", restored, opt2, config) == 1
    torch.testing.assert_close(torch.rand(3, device=device), expected_rng, rtol=0, atol=0)
    update(model, opt)
    update(restored, opt2)
    for lhs, rhs in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    dist.barrier()
    if rank == 0:
        assert len(list((a.output / "step-1").glob("optimizer-*.pt"))) == io.width
        assert len(list((a.output / "step-1").glob("rng-*.pt"))) == world
        (a.output / "RESULT.json").write_text(json.dumps({
            "complete": True, "world_size": world, "shard_width": io.width,
            "optimizer_replica_deduplication_verified": True,
            "rank_rng_restored": True, "next_update_bitwise_equal": True,
            "cpu_emulation": a.cpu, "sharding": a.sharding,
        }, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
