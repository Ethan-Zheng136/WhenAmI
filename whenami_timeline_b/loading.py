"""Strict A10k model-only warm start, on private CPU/GPU or sharded FSDP tensors."""

from pathlib import Path
import json
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict,set_model_state_dict,StateDictOptions

from stage3.common import STORE,BASE,REFERENCE
from stage3_oxe_eval.prepare import sha

TEACHER=STORE/"outputs/whenami_timeline_joint/training/timeline_joint_timeline_320_50k_01/main/checkpoints/step-010000"


def teacher_record(path=TEACHER):
    path=Path(path);meta=json.loads((path/"COMMITTED.json").read_text())
    if not meta["complete"] or meta["step"]!=10000 or meta["recipe"]["settings"]["mode"]!="timeline":
        raise ValueError("this pilot pins committed timeline A10k")
    for name,value in meta["recipe"]["source"].items():
        if sha(BASE/name)!=value:raise ValueError("A source contract drift: "+name)
    for name,value in meta["recipe"]["reference_source"].items():
        if sha(REFERENCE/name)!=value:raise ValueError("A reference drift: "+name)
    return {"path":str(path),"step":10000,"committed_sha256":sha(path/"COMMITTED.json"),
            "source_recipe":meta["recipe"],"scope":"model warm start only; B optimizer starts fresh"}


def load_model_only(model,path=TEACHER,group=None,sharded=False):
    path=Path(path)
    metadata=dcp.FileSystemReader(path/"model_dcp").read_metadata()
    keys={name.removeprefix("model.") for name in metadata.state_dict_metadata}
    options=StateDictOptions(full_state_dict=False,cpu_offload=False,strict=True)
    state=get_model_state_dict(model,options=options) if sharded else model.state_dict()
    aux={k for k in state if k.startswith("forecast_head.")}
    if set(state)-aux!=keys:
        raise ValueError("A->B live-state coverage mismatch: "+str((keys-(set(state)-aux),(set(state)-aux)-keys)))
    core={k:v for k,v in state.items() if k not in aux}
    for k,value in core.items():
        if tuple(value.shape)!=tuple(metadata.state_dict_metadata["model."+k].size):
            raise ValueError("A shape mismatch: "+k)
    # Destinations MUST already be private GPU/FSDP allocations, not mmap views.
    if not sharded and any(v.device.type!="cuda" for v in core.values()):
        raise ValueError("unsharded warm start must use private accelerator storage")
    dcp.load({"model":core},checkpoint_id=path/"model_dcp",process_group=group,
             no_dist=not torch.distributed.is_initialized())
    if sharded:
        result=set_model_state_dict(model,core,options=StateDictOptions(
            full_state_dict=False,cpu_offload=False,strict=False))
        if set(result.missing_keys)!=aux or result.unexpected_keys:
            raise ValueError("unexpected post-warmstart keys: "+str(result))
    return {"loaded_A_state_entries":len(keys),"new_auxiliary_entries":sorted(aux)}
