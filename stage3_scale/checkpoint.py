"""Collective DCP save/load with explicit commit and per-rank RNG.

No gathered full Adam state on rank0. Restore requires the same world size,
data recipe and code manifest; uncommitted checkpoints are not resumable.
"""

import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions

from stage3.common import write_json


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def barrier():
    if dist.is_initialized():
        dist.barrier()


def rng_state():
    rng = np.random.get_state()
    return {
        "torch": torch.get_rng_state(), "python": random.getstate(),
        "numpy": [rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]],
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    a = state["numpy"]
    np.random.set_state((a[0], np.asarray(a[1], dtype=np.uint32), a[2], a[3], a[4]))
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def save(path, model, optimizer, step, recipe):
    path = Path(path)
    rank, world = rank_world()
    token = [uuid.uuid4().hex if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(token, src=0)
    temporary = path.with_name(path.name + f".incomplete-{token[0]}")
    error = [None]
    if rank == 0:
        try:
            if path.exists():
                raise FileExistsError(path)
            temporary.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error[0] = str(exc)
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise OSError(f"collective checkpoint reservation failed: {error[0]}")
    barrier()
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    dcp.save({"model": model_state, "optimizer": optim_state},
             storage_writer=dcp.FileSystemWriter(temporary / "dcp", thread_count=1, overwrite=False))
    torch.save(rng_state(), temporary / f"rng-{rank:05d}.pt")
    barrier()
    if rank == 0:
        write_json(temporary / "COMMITTED.json", {
            "complete": True, "step": step, "world_size": world, "recipe": recipe,
            "sampling": "stateless seed/step/microstep/rank, num_workers=0",
        })
        os.replace(temporary, path)
    barrier()


def load(path, model, optimizer, recipe):
    path = Path(path)
    rank, world = rank_world()
    meta = json.loads((path / "COMMITTED.json").read_text())
    if not meta["complete"] or meta["world_size"] != world or meta["recipe"] != recipe:
        raise ValueError("resume world size / data / architecture / budget changed")
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=path / "dcp")
    set_state_dict(model, optimizer, model_state_dict=state["model"],
                   optim_state_dict=state["optimizer"], options=options)
    restore_rng(torch.load(path / f"rng-{rank:05d}.pt", weights_only=True, map_location="cpu"))
    barrier()
    return meta["step"]
