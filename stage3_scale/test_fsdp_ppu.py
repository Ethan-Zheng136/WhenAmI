"""Executable accelerator FSDP+DCP check using tiny networks.

Each NCCL rank requires its own physical accelerator. A single-device run
checks API/save/load compatibility but NOT cross-device sharding.
"""

import argparse
from pathlib import Path
import os
import copy

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch import nn

from stage3_scale.checkpoint_v2 import save, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    if local >= torch.cuda.device_count():
        raise ValueError("one distinct physical PPU per NCCL rank is required")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    torch.manual_seed(42)

    def build():
        net = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8))
        return FSDP(
            net, device_id=device, use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        )

    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    x = torch.randn(4, 16, device=device) + rank
    def update(m, optimizer):
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x).float().square().mean()
        loss.backward()
        m.clip_grad_norm_(1.)
        optimizer.step()

    update(model, opt)
    save(args.output / "step-1", model, opt, 1, {"real_fsdp_test": True})
    restored = build()
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.001)
    assert load(args.output / "step-1", restored, opt2, {"real_fsdp_test": True}) == 1
    update(model, opt)
    update(restored, opt2)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    dist.barrier()
    if rank == 0:
        from stage3.common import write_json
        write_json(args.output / "RESULT.json", {
            "complete": True, "world_size": dist.get_world_size(),
            "physical_ppus_visible_per_node": torch.cuda.device_count(),
            "fsdp_full_shard_dcp_model_optimizer_restored": True,
            "next_update_local_parameter_shards_bitwise_equal": True,
            "checkpoint_schema": "fsdp_model_dcp_local_adam_v2",
            "note": "Tiny model API test; world_size1 falls back to NO_SHARD; not real7B multi-node admission.",
        })
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
