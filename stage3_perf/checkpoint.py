"""Same-topology checkpoints without writing HSDP replicas twenty times.

All shard groups materialize sharded model state. Only the FIRST shard group
writes DCP and its local Adam shards; every global rank saves its own RNG.
On restore every shard group collectively loads the same model checkpoint,
and replica ranks read their shared Adam shard. No topology migration.
"""

import json
import os
from pathlib import Path
import uuid

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from stage3.common import write_json
from stage3_scale.checkpoint import barrier, rank_world, rng_state, restore_rng
from stage3_scale.checkpoint_v2 import optimizer_layout, to_cpu


SCHEMA = "whenami_perf_same_topology_deduplicated_v1"


class CheckpointIO:
    def __init__(self, sharding, groups):
        self.sharding = sharding
        self.group = groups[0] if groups is not None else None
        self.rank, self.world = rank_world()
        self.width = dist.get_world_size(self.group) if dist.is_initialized() else 1
        self.slot = dist.get_rank(self.group) if dist.is_initialized() else 0
        self.writer = self.rank < self.width
        self.options = StateDictOptions(full_state_dict=False, cpu_offload=True)

    def save(self, path, model, optimizer, step, recipe):
        path = Path(path)
        token = [uuid.uuid4().hex if self.rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(token, src=0)
        temporary = path.with_name(path.name + ".incomplete-" + token[0])
        error = [None]
        if self.rank == 0:
            try:
                if path.exists():
                    raise FileExistsError(path)
                temporary.mkdir(parents=True, exist_ok=False)
            except OSError as exc:
                error[0] = repr(exc)
        if dist.is_initialized():
            dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
        barrier()
        state = get_model_state_dict(model, options=self.options)
        if self.writer:
            dcp.save({"model": state},
                     storage_writer=dcp.FileSystemWriter(temporary / "model_dcp", thread_count=1, overwrite=False),
                     process_group=self.group)
            torch.save({
                "optimizer": to_cpu(optimizer.state_dict()), "layout": optimizer_layout(model, optimizer),
                "slot": self.slot, "shard_width": self.width,
            }, temporary / f"optimizer-{self.slot:05d}.pt")
        torch.save({"rng": rng_state(), "rank": self.rank, "world_size": self.world},
                   temporary / f"rng-{self.rank:05d}.pt")
        barrier()
        if self.rank == 0:
            write_json(temporary / "COMMITTED.json", {
                "schema": SCHEMA, "complete": True, "step": step, "world_size": self.world,
                "sharding": self.sharding, "shard_width": self.width, "recipe": recipe,
                "optimizer_files": self.width, "rng_files": self.world,
                "note": "HSDP replicas share model/Adam, NOT RNG. Same topology and recipe only.",
            })
            os.replace(temporary, path)
        barrier()

    def load(self, path, model, optimizer, recipe):
        path = Path(path)
        meta = json.loads((path / "COMMITTED.json").read_text())
        if meta.get("schema") != SCHEMA or not meta.get("complete"):
            raise ValueError("not a committed performance checkpoint")
        if (meta["world_size"], meta["shard_width"], meta["sharding"], meta["recipe"]) != (
                self.world, self.width, self.sharding, recipe):
            raise ValueError("same world/topology/recipe required; no silent optimizer reset")
        local = torch.load(path / f"optimizer-{self.slot:05d}.pt", map_location="cpu", weights_only=True)
        if (local["slot"], local["shard_width"], local["layout"]) != (
                self.slot, self.width, optimizer_layout(model, optimizer)):
            raise ValueError("local optimizer layout differs")
        rng = torch.load(path / f"rng-{self.rank:05d}.pt", map_location="cpu", weights_only=True)
        if rng["rank"] != self.rank or rng["world_size"] != self.world:
            raise ValueError("wrong RNG shard")
        state = {"model": get_model_state_dict(model, options=self.options)}
        dcp.load(state, checkpoint_id=path / "model_dcp", process_group=self.group)
        set_model_state_dict(model, state["model"], options=self.options)
        optimizer.load_state_dict(local["optimizer"])
        restore_rng(rng["rng"])
        barrier()
        return meta["step"]
