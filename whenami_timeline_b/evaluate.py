"""Causal-only per-episode action predictions, equal wrapped calls per rank."""

from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import torch.distributed as dist

from stage3_scale.checkpoint import rng_state,restore_rng
from stage3_oxe_eval.prepare import write_json
from whenami_timeline_b.data import student_inputs


@torch.no_grad()
def evaluate(student,visual,data,transform,tokenizer,prompt,device,out,seeds=(42,)):
    out=Path(out)
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    rng=rng_state();was=student.training
    student.eval();visual.eval();rows=[];predictions=[]
    try:
        for j in range((len(data.clips)+world-1)//world):
            ordinal=j*world+rank
            case=data.dev(min(ordinal,len(data.clips)-1))
            prefix,label=student_inputs([case],visual,transform,tokenizer,prompt,device)
            values=[]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                encoded=student(prefix=prefix,condition_only=True)
                conditioning={k:encoded[k] for k in ("conditions","reader_z")}
                for seed in seeds:
                    s=int.from_bytes(hashlib.sha256(
                        f"Bpilot10k:{case['provenance']['episode_key']}:{seed}".encode()).digest()[:8],"little")%(2**63-1)
                    x=torch.randn((1,16,7),generator=torch.Generator(device=device).manual_seed(s),device=device)
                    ts=torch.linspace(0,1,10,device=device)
                    for i in range(9):
                        v=student(noisy_actions=x,tau=ts[i].expand(1),conditioning=conditioning)["velocity"]
                        x=x+v.float()*(ts[i+1]-ts[i])
                    if not torch.isfinite(x).all():raise FloatingPointError("B action sampling")
                    values.append(x[0])
            prediction=torch.stack(values)
            name=case["provenance"]["dataset"];st=data.stats[name]["action"]
            unused=torch.tensor(np.asarray(st["min"])==np.asarray(st["max"]),device=device)
            normp=torch.where(unused,0.,prediction.clamp(-1,1))
            mask=label.valid[0]
            error=(normp[:,:,:6]-label.actions[0,:,:6]).square().mean(-1)
            mse=(error*mask).sum(-1)/mask.sum()
            unclipped=torch.where(unused,0.,prediction)
            ue=(unclipped[:,:,:6]-label.actions[0,:,:6]).square().mean(-1)
            grip=(((prediction[:,:,6]>=.5)==(label.actions[0,:,6]>=.5))*mask).sum(-1)/mask.sum()
            if ordinal<len(data.clips):
                rows.append({"dataset":name,"episode_key":case["provenance"]["episode_key"],
                    "mse6":float(mse.mean()),"unclipped_mse6":float(((ue*mask).sum(-1)/mask.sum()).mean()),
                    "first_action_mse6":float(error[:,0].mean()),"gripper_accuracy":float(grip.mean())})
                predictions.append({"index":ordinal,"prediction":prediction.cpu().numpy(),
                    "target":label.actions[0].cpu().numpy(),"valid":mask.cpu().numpy()})
        write_json(out/f"rank-{rank:05d}.json",rows)
        np.savez_compressed(out/f"rank-{rank:05d}.npz",
            indices=np.asarray([r["index"] for r in predictions],np.int64),
            predictions=np.stack([r["prediction"] for r in predictions]) if predictions else np.empty((0,len(seeds),16,7)),
            targets=np.stack([r["target"] for r in predictions]) if predictions else np.empty((0,16,7)),
            masks=np.stack([r["valid"] for r in predictions]) if predictions else np.empty((0,16),bool))
        if world>1:dist.barrier()
        if rank==0:
            merged=[]
            for r in range(world):merged.extend(json.loads((out/f"rank-{r:05d}.json").read_text()))
            if len(merged)!=len(data.clips) or len({(r["dataset"],r["episode_key"]) for r in merged})!=len(merged):
                raise ValueError("missing B eval cases")
            domains={}
            keys=("mse6","unclipped_mse6","first_action_mse6","gripper_accuracy")
            for name in sorted({r["dataset"] for r in merged}):
                subset=[r for r in merged if r["dataset"]==name]
                domains[name]={"episodes":len(subset),**{k:float(np.mean([r[k] for r in subset])) for k in keys}}
            macro={k:float(np.mean([r[k] for r in domains.values()])) for k in keys}
            write_json(out/"RESULT.json",{"complete":True,"causal_student_only":True,
                "student_observed_nodes":2,"current_anchor":"last observed prefix node",
                "episodes":len(merged),"domains":domains,"macro":macro,"noise_seeds":list(seeds),
                "scope":"Causal student DEV action errors; not environment success; no true-future model input"})
            print("TIMELINE_B_DEV",out.name,macro,flush=True)
        if world>1:dist.barrier()
    finally:
        student.train(was);visual.eval();restore_rng(rng)
