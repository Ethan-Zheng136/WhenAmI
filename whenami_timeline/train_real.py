"""Streaming actual OXE multi-anchor research with a frozen pretrained policy.

Not a continuation/replacement of the running full-policy Stage3. Every step
selects actual TRAIN episodes and encodes raw images; only timeline is optimized.
"""

import argparse
from dataclasses import asdict
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from stage3.common import BASE,check_endpoint
from whenami_timeline import checkpoint
from whenami_timeline.backbone import FrozenPastBackbone
from whenami_timeline.data import RawTimelines,feature_batch
from whenami_timeline.model import Timeline,MODES
from whenami_timeline.objectives import timeline_objective


def raw_batch(dataset,backbone,count,seed,step,rank):
    cases=[]
    for i in range(count):
        entry,times=dataset.select(seed,step,rank,i)
        raw,labels,provenance=dataset.materialize(entry,times,backbone.transform,backbone.statistics)
        cases.append({"policy":backbone.encode(raw),"labels":labels,"provenance":provenance})
    inputs,labels=feature_batch(cases,np.arange(len(cases)),backbone.device)
    return inputs,labels,[c["provenance"] for c in cases]


def global_mean(value,device):
    x=torch.as_tensor(value,dtype=torch.float32,device=device)
    if dist.is_initialized():dist.all_reduce(x);x/=dist.get_world_size()
    return x


@torch.no_grad()
def validate(model,cases,reader,device):
    # Fixed INTERNAL development cases drawn only from original train roles.
    module=checkpoint.bare(model);module.eval();values=[]
    for i in range(len(cases)):
        inputs,labels=feature_batch(cases,[i],device)
        out=module(**inputs)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            _,m=timeline_objective(out,inputs,labels,reader,200000+i,draws=2)
        values.append([float(m[k]) for k in ("action_fm","transition_mse","total")])
    return dict(zip(("action_fm","transition_mse","total"),
                    global_mean(np.mean(values,axis=0),device).cpu().tolist()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=Path,default=Path(
        "/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=MODES,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--steps",type=int,default=5000)
    p.add_argument("--micro",type=int,default=2)
    p.add_argument("--draws",type=int,default=2)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--save-every",type=int,default=500)
    p.add_argument("--dev-every",type=int,default=250)
    p.add_argument("--stop-after",type=int,default=0)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if min(args.steps,args.micro,args.draws,args.save_every,args.dev_every)<1 or args.lr<=0:
        raise ValueError("invalid training schedule")
    if args.stop_after<0 or args.stop_after>args.steps:raise ValueError("stop beyond schedule")
    rank=int(os.environ.get("RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0));device=torch.device("cuda",local)
    torch.cuda.set_device(device);torch.set_num_threads(1)
    if world>1:dist.init_process_group("nccl",timeout=timedelta(minutes=30))
    error=[None]
    if rank==0:
        try:
            if args.resume:
                if not args.output.is_dir() or (args.output/"COMPLETE.json").exists():
                    raise ValueError("no incomplete research run to resume")
            else:args.output.mkdir(parents=True,exist_ok=False)
        except (OSError,ValueError) as exc:error[0]=str(exc)
    if world>1:dist.broadcast_object_list(error,src=0)
    if error[0]:raise ValueError(error[0])
    checkpoint.barrier()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if rank==0:print("REAL_TIMELINE_LOAD frozen Stage2 backbone; no running Stage3 weights changed",flush=True)
    backbone=FrozenPastBackbone(device)
    train=RawTimelines(args.index,role="train")
    dev=RawTimelines(args.index,role="internal_dev")
    devcases=[]
    # Four fixed domain-balanced internal cases per rank. Domain coverage depends
    # on actual world size and is reported, not misrepresented as final27 evaluation.
    for i in range(4):
        domain=dev.domains[(rank*4+i)%len(dev.domains)]
        entry,times=dev.select(97531,1,rank,i,domain=domain)
        raw,labels,provenance=dev.materialize(entry,times,backbone.transform,backbone.statistics)
        devcases.append({"policy":backbone.encode(raw),"labels":labels,"provenance":provenance})
    del dev
    torch.manual_seed(args.seed)
    model=Timeline(mode=args.mode).to(device)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ("resume","stop_after")}
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    recipe={
        "schema":"real_timeline_frozen_policy_research_v1","settings":settings,
        "world_size":world,"global_trajectories":world*args.micro,"anchors_per_trajectory":3,
        "trainable_parameters":sum(p.numel() for p in model.parameters()),
        "config":asdict(model.config),"frozen_endpoint":backbone.endpoint,
        "source":{f.name:sha(f) for f in Path(__file__).parent.glob("*.py")},
        "science_source":{str(f.relative_to(BASE)):sha(f) for d in ("stage3","stage3_scale","whenami_revision")
                          for f in (BASE/d).glob("*.py")},
        "reference_source":{str(f.relative_to(BASE)):sha(f) for f in (BASE/"WoG_16rank_probe").rglob("*.py")},
        "index_sha256":sha(args.index),"role_rule":"original TRAIN only;sha256(timeline01/key)%20 ==0 internal_dev",
        "validation":"fixed4 cases/rank, shared DiT action FM; not closed-loop or main-table evaluation",
        "notes":["Frozen per-anchor vision/VLM/Past encodings; raw source sampled anew every train update.",
                 "Timeline internal states are128D tokens; no claim of identified physical latent.",
                 "same-world/same-recipe restart only; no source checkpoint updates."],
    }
    if rank==0 and not args.resume:(args.output/"recipe.json").write_text(json.dumps(recipe,indent=2))
    if args.resume and json.loads((args.output/"recipe.json").read_text())!=recipe:
        raise ValueError("research recipe drift")
    if world>1:model=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.)
    start=0
    if args.resume:
        cps=sorted((args.output/"checkpoints").glob("step-*/COMMITTED.json"))
        if not cps:raise ValueError("no committed research checkpoint")
        start=checkpoint.load(cps[-1].parent,model,optimizer,recipe)
        if rank==0:
            for name in ("train.jsonl","internal_dev.jsonl"):
                file=args.output/name
                if file.exists():
                    keep=[]
                    for line in file.read_text().splitlines():
                        try:
                            row=json.loads(line)
                            if row["step"]<=start:keep.append(json.dumps(row))
                        except json.JSONDecodeError:pass
                    file.write_text("\n".join(keep)+("\n" if keep else ""))
    # Sampling is counter-based; the per-rank stream is recorded/restored for
    # any future stochastic layers, not used to hide a data-order reset.
    if not args.resume:
        random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed+rank)
    if start==0:
        initial=validate(model,devcases,backbone.reader,device)
        if rank==0:
            (args.output/"initial_internal_dev.json").write_text(json.dumps(initial,indent=2))
            print("INITIAL_REAL_DEV",initial,flush=True)
    recent=deque(maxlen=100);progress=tqdm(total=args.steps,initial=start,disable=rank!=0,desc=f"Timeline {args.mode}")
    for step in range(start+1,args.steps+1):
        began=time.monotonic();model.train();optimizer.zero_grad()
        inputs,labels,identities=raw_batch(train,backbone,args.micro,args.seed,step,rank)
        out=model(**inputs)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss,metrics=timeline_objective(out,inputs,labels,backbone.reader,
                                             args.seed+step*world+rank,draws=args.draws)
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite timeline loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError("nonfinite timeline gradient")
        optimizer.step();backbone.assert_frozen();torch.cuda.synchronize()
        numeric=global_mean([float(metrics[k]) for k in ("action_fm","transition_mse","total")],device)
        elapsed=time.monotonic()-began;recent.append(elapsed)
        record={"step":step,"seconds":elapsed,"lr":args.lr,"grad_norm":float(norm),
                **dict(zip(("action_fm","transition_mse","total"),numeric.cpu().tolist()))}
        if rank==0:
            with (args.output/"train.jsonl").open("a") as f:f.write(json.dumps(record)+"\n")
            if args.resume and step==start+1:
                (args.output/"RESUME_VERIFIED.json").write_text(json.dumps({
                    "complete":True,"restored_step":start,"first_updated_step":step,
                    "world_size":world,"finite_loss_and_gradient":True,
                },indent=2))
            if step==start+1 or step%10==0:
                eta=(args.steps-step)*sum(recent)/len(recent)/3600
                print("REAL_TIMELINE",record,"ETA_train_hours",round(eta,2),flush=True)
            progress.update();progress.set_postfix(loss=record["action_fm"],s_per_step=sum(recent)/len(recent))
        if step%args.save_every==0 or step in (args.steps,args.stop_after):
            if rank==0:print("TIMELINE_SAVE",step,flush=True)
            checkpoint.save(args.output/"checkpoints"/f"step-{step:06d}",model,optimizer,step,recipe)
        if step%args.dev_every==0 or step in (args.steps,args.stop_after):
            results=validate(model,devcases,backbone.reader,device)
            if rank==0:
                with (args.output/"internal_dev.jsonl").open("a") as f:f.write(json.dumps({"step":step,**results})+"\n")
                print("TIMELINE_INTERNAL_DEV",step,results,flush=True)
        if args.stop_after and step==args.stop_after<args.steps:
            if rank==0:print("ORDERLY_TIMELINE_STOP",step,flush=True)
            break
        if step==args.steps and rank==0:
            (args.output/"COMPLETE.json").write_text(json.dumps({
                "complete":True,"steps":step,"research_only":True,"main_model_unchanged":True,
            },indent=2))
    check_endpoint(backbone.endpoint);progress.close();checkpoint.barrier()
    if world>1:dist.destroy_process_group()


if __name__=="__main__":main()
