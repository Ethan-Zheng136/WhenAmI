"""Read-only model-only DCP load to private accelerator storage from pinned A5k."""

import json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp

from stage3.common import BASE,STORE,REFERENCE
from stage3_oxe_eval.prepare import sha
from stage3_oxe_eval50.policy import param_bf16
from whenami_timeline_joint.model import load_joint

CHECKPOINT=STORE/"outputs/whenami_timeline_joint/training/timeline_joint_timeline_320_50k_01/main/checkpoints/step-005000"


def load_fixed(checkpoint=CHECKPOINT,device="cuda:0"):
    checkpoint=Path(checkpoint);device=torch.device(device)
    meta=json.loads((checkpoint/"COMMITTED.json").read_text())
    if not meta["complete"] or meta["step"]!=5000:raise ValueError("admission pins A5k")
    recipe=meta["recipe"]
    for key,value in recipe["source"].items():
        if sha(BASE/key)!=value:raise ValueError("running A source mismatch: "+key)
    for key,value in recipe["reference_source"].items():
        if sha(REFERENCE/key)!=value:raise ValueError("reference mismatch: "+key)
    model,visual,transform,tokenizer,prompt=load_joint(recipe["settings"]["mode"],recipe["settings"]["seed"])
    model=param_bf16(model.requires_grad_(False).eval()).to(device)
    visual=visual.requires_grad_(False).eval().to(device)
    state=model.state_dict()
    metadata=dcp.FileSystemReader(checkpoint/"model_dcp").read_metadata()
    if set(metadata.state_dict_metadata)!={"model."+k for k in state}:raise ValueError("DCP key mismatch")
    for name,x in state.items():
        if x.device!=device or x.shape!=metadata.state_dict_metadata["model."+name].size:
            raise ValueError("not a private matching accelerator tensor: "+name)
    dcp.load({"model":state},checkpoint_id=checkpoint/"model_dcp",no_dist=True)
    for name,x in state.items():
        if not torch.isfinite(x).all():raise ValueError("nonfinite A tensor "+name)
    record={"checkpoint":str(checkpoint),"step":meta["step"],
            "committed_sha256":sha(checkpoint/"COMMITTED.json"),"strict_model_entries":len(state),
            "all_loaded_finite":True,"scope":"model-only load; no A optimizer resumed"}
    return model,visual,transform,tokenizer,prompt,record
