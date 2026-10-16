"""Resume a 320/16 (or 160/16) hybrid_zero2 checkpoint on 160/16.

Model: DCP written by shard group 0 (16 ranks); every 16-rank shard group loads it.
Adam: optimizer-<slot>.pt keyed by shard-local slot 0..15, identical across replicas,
so width-16 groups at 160 read exactly the same files. RNG: rank r restores rng-r
(ranks 160..319 states unused; training noise/data are explicitly seeded, see train).
Recipe must match except world_size / topology_migration.
"""
import json
from pathlib import Path
import torch
from stage3_perf.checkpoint import CheckpointIO, SCHEMA
from stage3_scale.checkpoint import barrier, restore_rng
from stage3_scale.checkpoint_v2 import optimizer_layout
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
import torch.distributed.checkpoint as dcp
from stage3.common import BASE
from stage3_oxe_eval.prepare import sha

ALLOWED_SOURCE_WORLDS = (320, 160)
STRIP = ("world_size", "topology_migration")


def migration_fingerprints():
    r = {str(p.relative_to(BASE)): sha(p) for p in sorted((BASE/"whenami_timeline_force_160").glob("*.py"))}
    r["run_whenami_timeline_force_160.sh"] = sha(BASE/"run_whenami_timeline_force_160.sh")
    return r


def core(recipe):
    return {k: v for k, v in recipe.items() if k not in STRIP}


class CheckpointIO160(CheckpointIO):
    def load(self, path, model, optimizer, recipe):
        path = Path(path)
        meta = json.loads((path/"COMMITTED.json").read_text())
        if meta.get("schema") != SCHEMA or not meta.get("complete"):
            raise ValueError("not a committed performance checkpoint")
        if (meta["world_size"] not in ALLOWED_SOURCE_WORLDS or meta["world_size"] < self.world or
                meta["shard_width"] != self.width or meta["sharding"] != self.sharding or
                meta["optimizer_files"] != self.width):
            raise ValueError("need same shard width/sharding; source world 320 or 160")
        if core(meta["recipe"]) != core(recipe):
            diff = sorted(k for k in set(core(meta["recipe"])) | set(core(recipe))
                          if core(meta["recipe"]).get(k) != core(recipe).get(k))
            raise ValueError("recipe differs beyond topology: "+str(diff))
        local = torch.load(path/f"optimizer-{self.slot:05d}.pt", map_location="cpu", weights_only=True)
        if (local["slot"], local["shard_width"], local["layout"]) != (
                self.slot, self.width, optimizer_layout(model, optimizer)):
            raise ValueError("local optimizer layout differs")
        rng = torch.load(path/f"rng-{self.rank:05d}.pt", map_location="cpu", weights_only=True)
        if rng["rank"] != self.rank or rng["world_size"] != meta["world_size"]:
            raise ValueError("wrong RNG shard")
        state = {"model": get_model_state_dict(model, options=self.options)}
        dcp.load(state, checkpoint_id=path/"model_dcp", process_group=self.group)
        set_model_state_dict(model, state["model"], options=self.options)
        optimizer.load_state_dict(local["optimizer"])
        restore_rng(rng["rng"])
        barrier()
        if self.rank == 0:
            print("TOPOLOGY_MIGRATION_LOAD", meta["world_size"], "->", self.world, "step", meta["step"], flush=True)
        return meta["step"]
