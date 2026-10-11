"""Atomic model/Adam/RNG checkpoints; no production checkpoint conversion."""

import os
from pathlib import Path
import json
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist

from stage3_oxe_eval.prepare import sha,write_json


def bare(model):
    return model.module if hasattr(model,"module") else model


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),
            "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    if state["cuda"]:torch.cuda.set_rng_state_all(state["cuda"])


def save(path,model,optimizer,step,recipe):
    path=Path(path)
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    token=[uuid.uuid4().hex if rank==0 else None]
    if world>1:dist.broadcast_object_list(token,src=0)
    tmp=path.with_name(path.name+".incomplete-"+token[0])
    error=[None]
    if rank==0:
        try:
            if path.exists():raise FileExistsError(path)
            tmp.mkdir(parents=True,exist_ok=False)
        except OSError as exc:error[0]=repr(exc)
    if world>1:dist.broadcast_object_list(error,src=0)
    if error[0]:raise RuntimeError(error[0])
    if world>1:dist.barrier()
    if rank==0:
        torch.save({"model":bare(model).state_dict(),"optimizer":optimizer.state_dict()},tmp/"model_optimizer.pt")
    torch.save(rng_state(),tmp/f"rng-{rank:05d}.pt")
    if world>1:dist.barrier()
    if rank==0:
        write_json(tmp/"COMMITTED.json",{"complete":True,"step":step,"world":world,"recipe":recipe,
                    "mode":getattr(getattr(bare(model),"timeline",None),"mode",None),
                    "schema":"timeline_a_ddp_v1"})
        os.replace(tmp,path)
    if world>1:dist.barrier()


def load(path,model,optimizer,recipe):
    path=Path(path)
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    meta=json.loads((path/"COMMITTED.json").read_text())
    if not meta.get("complete") or meta.get("schema")!="timeline_a_ddp_v1" or meta["recipe"]!=recipe or meta["world"]!=world:
        raise ValueError("checkpoint recipe/world mismatch")
    if meta["mode"]!=getattr(getattr(bare(model),"timeline",None),"mode",None):raise ValueError("routing mode changed")
    values=torch.load(path/"model_optimizer.pt",map_location="cpu",mmap=True,weights_only=True)
    bare(model).load_state_dict(values["model"],strict=True)
    optimizer.load_state_dict(values["optimizer"])
    del values
    # Trusted RNG payload created by this module (numpy/python state is non-tensor).
    set_rng(torch.load(path/f"rng-{rank:05d}.pt",map_location="cpu",weights_only=False))
    return meta["step"]
