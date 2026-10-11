"""Same root FSDP call count on every rank; re-encode learned VLM at each checkpoint."""

from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import torch.distributed as dist

from stage3_scale.checkpoint import rng_state, restore_rng
from stage3_oxe_eval.prepare import write_json
from whenami_timeline_joint.data import tensorize,frozen_features


@torch.no_grad()
def evaluate(model,visual,data,transform,tokenizer,prompt,device,out):
    out=Path(out)
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    rng=rng_state();was=model.training
    model.eval();visual.eval();rows=[]
    try:
        for j in range((len(data.clips)+world-1)//world):
            ordinal=j*world+rank
            case=data.dev(min(ordinal,len(data.clips)-1))
            inputs,labels=tensorize([case],transform,tokenizer,prompt,device)
            seed=int.from_bytes(hashlib.sha256(
                ("timeline-joint-dev42/"+case["provenance"]["episode_key"]).encode()).digest()[:8],"little")%(2**63-1)
            g=torch.Generator(device=device).manual_seed(seed)
            x0=torch.randn((inputs["times"].numel(),16,7),generator=g,device=device)
            scores={}
            with torch.autocast("cuda",dtype=torch.bfloat16):
                features=frozen_features(visual,inputs["pixels"])
                for mode in ("full","masked"):
                    visible=torch.ones_like(inputs["times"],dtype=torch.bool)
                    if mode=="masked":visible[:,visible.shape[1]//2]=False
                    enc=model(**inputs,**features,visible=visible,condition_only=True)
                    x=x0.clone();ts=torch.linspace(0,1,10,device=device)
                    for k in range(9):
                        v=model(noisy_actions=x,tau=ts[k].expand(len(x)),conditioning=enc)["velocity"]
                        x=x+v.float()*(ts[k+1]-ts[k])
                    if not torch.isfinite(x).all():raise FloatingPointError("nonfinite sampled actions")
                    target=labels["actions"].flatten(0,1)
                    mask=labels["action_mask"].flatten(0,1)
                    st=data.stats[case["provenance"]["dataset"]]["action"]
                    unused=torch.tensor(np.asarray(st["min"])==np.asarray(st["max"]),device=device)
                    clipped=torch.where(unused,0.,x.clamp(-1,1))
                    err=(clipped[:,:,:6]-target[:,:,:6]).square().mean(-1)
                    mse=(err*mask).sum(-1)/mask.sum(-1)
                    grip=(((x[:,:,6]>=.5)==(target[:,:,6]>=.5))*mask).sum(-1)/mask.sum(-1)
                    score={"mse6":float(mse.mean()),"first_action_mse6":float(err[:,0].mean()),
                           "gripper_accuracy":float(grip.mean())}
                    if mode=="masked":score["hidden_node_mse6"]=float(mse[~visible.flatten()].mean())
                    scores[mode]=score
            if ordinal<len(data.clips):
                rows.append({"dataset":case["provenance"]["dataset"],
                             "episode_key":case["provenance"]["episode_key"],**scores})
        write_json(out/f"rank-{rank:05d}.json",rows)
        if world>1:dist.barrier()
        if rank==0:
            all_rows=[]
            for r in range(world):all_rows.extend(json.loads((out/f"rank-{r:05d}.json").read_text()))
            if len(all_rows)!=len(data.clips) or len({(x["dataset"],x["episode_key"]) for x in all_rows})!=len(all_rows):
                raise ValueError("missing/duplicate dev rows")
            domains={}
            for name in sorted({r["dataset"] for r in all_rows}):
                subset=[r for r in all_rows if r["dataset"]==name]
                domains[name]={"episodes":len(subset),**{
                    mode:{key:float(np.mean([r[mode][key] for r in subset])) for key in subset[0][mode]}
                    for mode in ("full","masked")}}
            macro={mode:{key:float(np.mean([r[mode][key] for r in domains.values()]))
                         for key in next(iter(domains.values()))[mode]} for mode in ("full","masked")}
            write_json(out/"RESULT.json",{"complete":True,"episodes":len(all_rows),
                "domains":domains,"macro":macro,"raw_vlm_recomputed":True,
                "scope":"Phase-A privileged teacher development, not online student or closed-loop",
                "sampler":"same FP32 identity noise,9 Euler intervals,BF16 policy; no GT-action noising"})
            print("JOINT_TIMELINE_DEV",out.name,macro,flush=True)
        if world>1:dist.barrier()
    finally:
        model.train(was);visual.eval();restore_rng(rng)
