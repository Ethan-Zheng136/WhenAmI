"""Same-topology FSDP checkpoints with collective model and rank-local Adam/RNG.

The previous global optimizer state_dict failed on32 ranks with empty local
parameter shards (`optimizer.param_groups` absent). Model weights still use
collective DCP. Each rank independently saves its actual optimizer slots,
groups, parameter-layout signature and RNG, with no rank0 optimizer gathering.
No world-size/topology change is supported.
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
from stage3_scale.checkpoint import rank_world, barrier, rng_state, restore_rng


def optimizer_layout(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [
        [{"name": names[id(p)], "local_shape": list(p.shape), "dtype": str(p.dtype),
          "requires_grad": p.requires_grad} for p in group["params"]]
        for group in optimizer.param_groups
    ]


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def save(path, model, optimizer, step, recipe):
    path = Path(path)
    rank, world = rank_world()
    token = [uuid.uuid4().hex if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(token, src=0)
    temporary = path.with_name(path.name + ".incomplete-" + token[0])
    error = [None]
    if rank == 0:
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
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state = get_model_state_dict(model, options=options)
    dcp.save({"model": state}, storage_writer=dcp.FileSystemWriter(
        temporary / "model_dcp", thread_count=1, overwrite=False))
    # Capture only actual LOCAL optimizer state; zero-sized FSDP parameters and
    # groups are preserved rather than converted to a rank-inconsistent global schema.
    payload = {
        "optimizer": to_cpu(optimizer.state_dict()), "rng": rng_state(),
        "layout": optimizer_layout(model, optimizer), "rank": rank, "world_size": world,
    }
    torch.save(payload, temporary / f"optimizer-rng-{rank:05d}.pt")
    barrier()
    if rank == 0:
        write_json(temporary / "COMMITTED.json", {
            "schema": "fsdp_model_dcp_local_adam_v2", "complete": True,
            "step": step, "world_size": world, "recipe": recipe,
            "optimizer_layout": "rank-local, same world and parameter order required",
        })
        os.replace(temporary, path)
    barrier()


def load(path, model, optimizer, recipe):
    path = Path(path)
    rank, world = rank_world()
    meta = json.loads((path / "COMMITTED.json").read_text())
    if meta.get("schema") != "fsdp_model_dcp_local_adam_v2" or not meta["complete"]:
        raise ValueError("not a committed v2 checkpoint")
    if meta["world_size"] != world or meta["recipe"] != recipe:
        raise ValueError("same topology/recipe required")
    local = torch.load(path / f"optimizer-rng-{rank:05d}.pt", map_location="cpu", weights_only=True)
    if local["rank"] != rank or local["world_size"] != world or local["layout"] != optimizer_layout(model, optimizer):
        raise ValueError("local FSDP optimizer layout differs")
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state = {"model": get_model_state_dict(model, options=options)}
    dcp.load(state, checkpoint_id=path / "model_dcp")
    set_model_state_dict(model, state["model"], options=options)
    # PyTorch restores floating optimizer slots to their parameter's device.
    optimizer.load_state_dict(local["optimizer"])
    restore_rng(local["rng"])
    barrier()
    return meta["step"]
