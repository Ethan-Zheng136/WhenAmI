"""Actual raw OXE Phase-A training. Single-PPU tested; multinode needs admission."""

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from stage3.common import BASE,REFERENCE
from stage3_oxe_eval.prepare import sha,write_json
from whenami_timeline_a.backbone import Foundation
from whenami_timeline_a.data import INDEX,RawAnchors,stack_cases,visibility
from whenami_timeline_a.model import MODES,Config
from whenami_timeline_a.objectives import noisy_labels,velocity_loss
from whenami_timeline_a.smoke import optimizer_for,gradient_report
from whenami_timeline_a import checkpoint


def tensor_digest(value):
    """Hash full tensor content, not a sampled norm."""
    x=value.detach().cpu().contiguous()
    return hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_digest(model,optimizer):
    p={name:tensor_digest(v) for name,v in checkpoint.bare(model).state_dict().items()}
    opt=optimizer.state_dict()
    o={str(key):{name:tensor_digest(v) if torch.is_tensor(v) else v
                 for name,v in value.items()} for key,value in opt["state"].items()}
    return {"model":p,"optimizer":o,"param_groups":opt["param_groups"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=MODES,required=True)
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--micro",type=int,default=1)
    p.add_argument("--nodes",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--save-every",type=int,default=500)
    p.add_argument("--stop-after",type=int,default=0)
    p.add_argument("--resume-from",type=Path)
    p.add_argument("--audit-state",action="store_true",help="full model/Adam hashes each step, admission only")
    args=p.parse_args()
    if min(args.steps,args.micro,args.save_every)<1 or args.nodes<2 or not 0<=args.stop_after<=args.steps:
        p.error("invalid schedule")
    rank=int(os.environ.get("RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0))
    torch.set_num_threads(1);torch.cuda.set_device(local)
    if world>1:dist.init_process_group("nccl",timeout=timedelta(minutes=30))
    if rank==0:args.output.mkdir(parents=True,exist_ok=False)
    if world>1:dist.barrier()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    foundation=Foundation(f"cuda:{local}")
    dataset=RawAnchors(nodes=args.nodes)
    core=foundation.model(args.mode,args.seed)
    model=DDP(core,device_ids=[local],find_unused_parameters=True) if world>1 else core
    opt=optimizer_for(core)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ("output","stop_after","resume_from","audit_state")}
    recipe={
        "schema":"timeline_a_raw_shared_full_reader_v1","settings":settings,
        "world":world,"global_trajectories":world*args.micro,"anchors":args.nodes,"draws":2,
        "config":asdict(Config()),"endpoint":foundation.endpoint,"index_sha256":sha(INDEX),
        "source":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
        "science_dependencies":{str(p.relative_to(BASE)):sha(p) for name in
            ("stage3","stage3_scale","evaluation","whenami_revision")
            for p in (BASE/name).glob("*.py")},
        "reference_source":{str(p.relative_to(REFERENCE)):sha(p) for p in REFERENCE.rglob("*.py")},
        "mask_schedule":"odd: all visible, even: one interior node absent in evidence AND reader z",
        "scope":"Privileged Phase-A condition learning; not deployable Phase-B or validated novelty.",
    }
    if rank==0:write_json(args.output/"recipe.json",recipe)
    random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed+rank)
    start=0
    if args.resume_from:
        start=checkpoint.load(args.resume_from,model,opt,recipe)
        if rank==0:write_json(args.output/"RESTORED.json",{"step":start,"source":str(args.resume_from)})
    recent=deque(maxlen=50)
    progress=tqdm(total=args.steps,initial=start,disable=rank!=0,desc="Timeline-A "+args.mode)
    for step in range(start+1,args.steps+1):
        began=time.monotonic()
        cases=[]
        for i in range(args.micro):
            entry,times=dataset.select(args.seed,step,rank,i)
            obs,labels,prov=dataset.materialize(entry,times,foundation.statistics)
            cases.append({"inputs":foundation.encode(obs),"labels":labels,"provenance":prov})
        x,y=stack_cases(cases,foundation.device)
        visible,cutoff=visibility(x["times"],args.seed+step*world+rank,hide_interior=step%2==0)
        noisy,tau,target=noisy_labels(y["actions"],args.seed+step*world+rank,draws=2)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(**x,visible=visible,cutoffs=cutoff,noisy_actions=noisy,tau=tau)
            loss,metrics=velocity_loss(output,target,y["action_mask"],draws=2)
        if not torch.isfinite(loss):raise FloatingPointError("loss")
        loss.backward()
        grad=gradient_report(core)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError("gradient")
        opt.step();foundation.check_frozen();torch.cuda.synchronize()
        loss_value=loss.detach().float()
        if world>1:dist.all_reduce(loss_value);loss_value/=world
        elapsed=time.monotonic()-began;recent.append(elapsed)
        record={"step":step,"action_fm":float(loss_value),"grad_norm":float(norm),
                "seconds":elapsed,"gradients":grad,"hidden_nodes_local":int((~visible).sum()),
                "identities_local":[c["provenance"] for c in cases]}
        if rank==0:
            with (args.output/"train.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
            if args.audit_state:
                write_json(args.output/f"state-digest-{step:06d}.json",state_digest(model,opt))
            progress.update()
            progress.set_postfix(loss=record["action_fm"],s_it=round(sum(recent)/len(recent),2))
            write_json(args.output/"progress.json",{
                "utc":datetime.now(timezone.utc).isoformat(),"step":step,"steps":args.steps,
                "seconds_per_step":sum(recent)/len(recent),
                "eta_training_seconds":(args.steps-step)*sum(recent)/len(recent),
                "note":"ETA excludes checkpoint IO/hash audit; future distributed speed unvalidated"})
        if step%args.save_every==0 or step in (args.steps,args.stop_after):
            if rank==0:print("TIMELINE_A_SAVE",step,flush=True)
            checkpoint.save(args.output/"checkpoints"/f"step-{step:06d}",model,opt,step,recipe)
        if args.stop_after and step==args.stop_after<args.steps:
            if rank==0:print("TIMELINE_A_ORDERLY_STOP",step,flush=True)
            break
        if step==args.steps and rank==0:
            write_json(args.output/"COMPLETE.json",{"complete":True,"step":step,"mode":args.mode,"world":world})
    progress.close()
    if world>1:dist.barrier();dist.destroy_process_group()


if __name__=="__main__":main()
