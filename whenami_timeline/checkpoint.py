"""Atomic research DDP checkpoints: one small model/Adam, per-rank RNG."""

import json
from pathlib import Path
import os
import uuid

import torch
import torch.distributed as dist

from stage3_scale.checkpoint import rng_state,restore_rng


def rank_world():
    return (dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)


def barrier():
    if dist.is_initialized():dist.barrier()


def bare(model):
    return model.module if hasattr(model,"module") else model


def save(path,model,optimizer,step,recipe):
    path=Path(path);rank,world=rank_world()
    token=[uuid.uuid4().hex if rank==0 else None]
    if dist.is_initialized():dist.broadcast_object_list(token,src=0)
    temporary=path.with_name(path.name+".incomplete-"+token[0])
    error=[None]
    if rank==0:
        try:
            if path.exists():raise FileExistsError(path)
            temporary.mkdir(parents=True,exist_ok=False)
        except OSError as exc:error[0]=repr(exc)
    if dist.is_initialized():dist.broadcast_object_list(error,src=0)
    if error[0]:raise RuntimeError(error[0])
    barrier()
    if rank==0:
        torch.save({"model":bare(model).state_dict(),"optimizer":optimizer.state_dict()},
                   temporary/"timeline.pt")
    torch.save({"rank":rank,"world":world,"rng":rng_state()},temporary/f"rng-{rank:05d}.pt")
    barrier()
    if rank==0:
        (temporary/"COMMITTED.json").write_text(json.dumps({
            "complete":True,"step":step,"recipe":recipe,"world_size":world,
            "schema":"timeline_ddp_frozen_backbone_v1",
        },indent=2))
        os.replace(temporary,path)
    barrier()


def load(path,model,optimizer,recipe):
    path=Path(path);rank,world=rank_world()
    m=json.loads((path/"COMMITTED.json").read_text())
    if (m.get("schema"),m.get("complete"),m.get("world_size"),m.get("recipe"))!=(
            "timeline_ddp_frozen_backbone_v1",True,world,recipe):
        raise ValueError("checkpoint not committed or same-world/recipe mismatch")
    state=torch.load(path/"timeline.pt",map_location="cpu",weights_only=True)
    bare(model).load_state_dict(state["model"],strict=True)
    optimizer.load_state_dict(state["optimizer"])
    local=torch.load(path/f"rng-{rank:05d}.pt",map_location="cpu",weights_only=True)
    if local["rank"]!=rank or local["world"]!=world:raise ValueError("wrong RNG rank")
    restore_rng(local["rng"]);barrier()
    return m["step"]
