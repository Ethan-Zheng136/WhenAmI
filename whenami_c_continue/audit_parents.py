"""Read every saved local Adam/RNG payload; mmap avoids materializing 240GiB.

No optimizer values are modified. Moment layouts/dtypes and actual Adam step
counters are audited, not every element of the full moment tensors.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from stage3_oxe_eval.prepare import write_json
from whenami_c_continue.contracts import PARENTS,source_record


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result={}
    for case in PARENTS:
        parent=source_record(case);root=Path(parent["path"])
        metadata=dcp.FileSystemReader(root/"model_dcp").read_metadata().state_dict_metadata
        if len(metadata)!=1549:raise ValueError("C full-state coverage differs")
        scalar={"model":{"read_scale":torch.empty((),dtype=torch.float32)}}
        dcp.load(scalar,checkpoint_id=root/"model_dcp",no_dist=True)
        if float(scalar["model"]["read_scale"])!=parent["read_scale"]:
            raise ValueError("parent C read_scale payload mismatch")
        slots=[]
        for i in range(16):
            state=torch.load(root/f"optimizer-{i:05d}.pt",weights_only=True,mmap=True,map_location="cpu")
            if state["slot"]!=i or state["shard_width"]!=16:
                raise ValueError("wrong optimizer shard")
            opt=state["optimizer"]
            if [g["lr"] for g in opt["param_groups"]]!=parent["parent_terminal_lr"]:
                raise ValueError("parent Adam terminal LR does not match log")
            if len(state["layout"])!=2 or len(opt["param_groups"])!=2:
                raise ValueError("parent Adam group count differs")
            for group in opt["param_groups"]:
                if group["weight_decay"]!=0. or tuple(group["betas"])!=(.9,.999) or group["eps"]!=1e-8:
                    raise ValueError("parent Adam hyperparameters differ")
            steps=[]
            for s in opt["state"].values():
                steps.append(float(s["step"]))
                if (not torch.isfinite(s["step"]) or s["exp_avg"].shape!=s["exp_avg_sq"].shape
                        or s["exp_avg"].dtype!=torch.float32 or s["exp_avg_sq"].dtype!=torch.float32):
                    raise ValueError("parent optimizer moment metadata mismatch")
            if not steps or max(steps)!=20000 or min(steps)<=0:
                raise ValueError("parent Adam counters reset/missing")
            slots.append({"slot":i,"state_count":len(steps),"step_min":min(steps),"step_max":max(steps)})
            del state,opt
        for rank in range(320):
            state=torch.load(root/f"rng-{rank:05d}.pt",weights_only=True,map_location="cpu")
            if state["rank"]!=rank or state["world_size"]!=320 or not state["rng"]:
                raise ValueError("parent rank RNG invalid")
        result[case]={"complete":True,"parent":parent,"model_entries":len(metadata),
                      "optimizer_shards":slots,"rng_shards_verified":320,"read_scale":parent["read_scale"]}
        print("PARENT_ADAM_RNG_AUDITED",case,flush=True)
    write_json(a.output,{"complete":True,"utc":datetime.now(timezone.utc).isoformat(),
        "parents":result,"scope":"All32 Adam shards/640 RNG records parsed; moment shape/dtype/counters checked. Not a full320 model restore or exhaustive moment-value scan."})


if __name__=="__main__":main()
