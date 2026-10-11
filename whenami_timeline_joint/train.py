"""Full-VLM raw OXE timeline teacher with established sharded checkpoint backend."""

import argparse
from collections import deque
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from stage3.common import BASE,REFERENCE,DEFAULT_FUTURE,endpoint_record,check_endpoint
from stage3_oxe_eval.prepare import sha,write_json
from stage3_perf.checkpoint import CheckpointIO
from whenami_timeline_a.data import visibility,INDEX
from whenami_timeline_a.objectives import noisy_labels,velocity_loss
from whenami_timeline_joint.model import load_joint,MODES
from whenami_timeline_joint.data import Inputs,DEV,tensorize,frozen_features
from whenami_timeline_joint.runtime import wrap,topology,learning_rate,gradient_norms
from whenami_timeline_joint.evaluate import evaluate


def fingerprints():
    directories=("whenami_timeline_joint","whenami_timeline_a","whenami_revision",
                 "stage3","stage3_scale","stage3_perf","evaluation")
    return {str(p.relative_to(BASE)):sha(p) for directory in directories for p in (BASE/directory).glob("*.py")}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=MODES,required=True)
    p.add_argument("--steps",type=int,default=50000)
    p.add_argument("--micro",type=int,default=1)
    p.add_argument("--global-batch",type=int,default=320)
    p.add_argument("--nodes",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=1e-5)
    p.add_argument("--new-lr",type=float,default=1e-4)
    p.add_argument("--warmup",type=int,default=1000)
    p.add_argument("--sharding",choices=("full","hybrid","hybrid_zero2"),default="hybrid_zero2")
    p.add_argument("--activation",choices=("none","llm","all"),default="llm")
    p.add_argument("--save-every",type=int,default=5000)
    p.add_argument("--eval-every",type=int,default=2500)
    p.add_argument("--first-check",type=int,default=100)
    p.add_argument("--stop-after",type=int,default=0)
    p.add_argument("--resume-from",type=Path)
    p.add_argument("--dev",type=Path,default=DEV)
    p.add_argument("--dev-limit",type=int,default=0)
    args=p.parse_args()
    rank,world=int(os.environ.get("RANK",0)),int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0));device=torch.device("cuda",local)
    if min(args.steps,args.micro,args.global_batch,args.save_every,args.eval_every,args.first_check)<1:
        p.error("positive schedule values required")
    if args.global_batch%(world*args.micro) or not 0<=args.warmup<args.steps or not 0<=args.stop_after<=args.steps:
        p.error("global batch/topology/schedule mismatch")
    torch.set_num_threads(1);torch.cuda.set_device(device)
    dist.init_process_group("nccl",timeout=timedelta(minutes=40))
    error=[None]
    if rank==0:
        try:args.output.mkdir(parents=True,exist_ok=False)
        except OSError as e:error[0]=str(e)
    dist.broadcast_object_list(error,src=0)
    if error[0]:raise ValueError(error[0])
    dist.barrier()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    core,visual,transform,tokenizer,prompt=load_joint(args.mode,args.seed)
    data=Inputs(args.nodes,args.dev,args.dev_limit)
    record=endpoint_record(DEFAULT_FUTURE)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ("output","stop_after","resume_from")}
    recipe={"schema":"full_vlm_timeline_a_v1","settings":settings,"world_size":world,
        "local_world_size":int(os.environ.get("LOCAL_WORLD_SIZE",1)),
        "source":fingerprints(),"reference_source":{str(p.relative_to(REFERENCE)):sha(p) for p in REFERENCE.rglob("*.py")},
        "endpoint":record,"index_sha256":sha(INDEX),"dev_manifest_sha256":sha(args.dev/"MANIFEST.json"),
        "trainable_parameters":sum(p.numel() for p in core.parameters() if p.requires_grad),
        "trainable":["VLM vision/projector/LLM","local QFormer","temporal conditioner","full DiT"],
        "frozen":["standalone per-frame DINO/Wan"],
        "scope":"Phase-A teacher, action-supervised; no physical action-conditioned transitions or Phase-B",
        "mask":"alternate full evidence and one hidden interior node; no hidden-image VLM bypass",
        "augmentation":False}
    if rank==0:write_json(args.output/"recipe.json",recipe)
    strategy,groups=topology(args.sharding,device)
    model=wrap(core,device,strategy,groups,args.activation)
    io=CheckpointIO(args.sharding,groups)
    visual=visual.to(device).eval()
    pretrained,new=[],[]
    for name,param in model.named_parameters():
        clean=name.replace("_fsdp_wrapped_module.","").replace("_checkpoint_wrapped_module.","")
        if param.requires_grad:(new if clean.startswith("timeline.") else pretrained).append(param)
    optimizer=torch.optim.AdamW([{"params":pretrained,"lr":args.lr},{"params":new,"lr":args.new_lr}],weight_decay=0.)
    random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed+rank)
    start=0
    if args.resume_from:
        start=io.load(args.resume_from,model,optimizer,recipe)
        if rank==0:write_json(args.output/"RESTORED.json",{"step":start,"source":str(args.resume_from),"full_optimizer":True})
    if rank==0:print("FULL_VLM_TIMELINE_READY",recipe["trainable_parameters"],world,args.mode,flush=True)
    evaluate(model,visual,data,transform,tokenizer,prompt,device,args.output/"development"/f"step-{start:06d}")
    accum=args.global_batch//(world*args.micro)
    recent=deque(maxlen=100);save_times=[];eval_times=[]
    progress=tqdm(total=args.steps,initial=start,disable=rank!=0,desc="Full timeline "+args.mode)
    for step in range(start+1,args.steps+1):
        began=time.monotonic();model.train();visual.eval();optimizer.zero_grad(set_to_none=True)
        for g,peak in zip(optimizer.param_groups,(args.lr,args.new_lr)):g["lr"]=learning_rate(step,args.steps,peak,args.warmup)
        losses=[]
        for micro in range(accum):
            cases=data.batch(args.micro,args.seed,step,rank,micro)
            inputs,labels=tensorize(cases,transform,tokenizer,prompt,device)
            mask_seed=args.seed+(step*accum+micro)*world+rank
            vis,cut=visibility(inputs["times"],mask_seed,hide_interior=step%2==0)
            noisy,tau,target=noisy_labels(labels["actions"],mask_seed,draws=2)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                features=frozen_features(visual,inputs["pixels"])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output=model(**inputs,**features,visible=vis,cutoffs=cut,noisy_actions=noisy,tau=tau)
                loss,_=velocity_loss(output,target,labels["action_mask"],draws=2)
            if not torch.isfinite(loss):raise FloatingPointError("loss")
            (loss/accum).backward();losses.append(float(loss.detach()))
        # Collective norm audit only at early/periodic points; shard-local norms
        # cannot alone establish that every learned module was actually trained.
        grads=gradient_norms(model,device) if step in (1,2,start+1,args.first_check) else None
        norm=model.clip_grad_norm_(1.)
        if not torch.isfinite(norm):raise FloatingPointError("gradient")
        optimizer.step()
        if any(p.grad is not None for p in visual.parameters()):raise ValueError("frozen visual gradient")
        torch.cuda.synchronize()
        loss_global=torch.tensor(sum(losses)/len(losses),device=device);dist.all_reduce(loss_global);loss_global/=world
        sec=torch.tensor(time.monotonic()-began,device=device);dist.all_reduce(sec,op=dist.ReduceOp.MAX)
        memory=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=device);dist.all_reduce(memory,op=dist.ReduceOp.MAX)
        recent.append(float(sec))
        if rank==0:
            row={"step":step,"action_fm":float(loss_global),"grad_norm":float(norm),"seconds_max":float(sec),
                 "peak_allocated_gib_max":float(memory),"lr":[g["lr"] for g in optimizer.param_groups],
                 "gradients":grads}
            with (args.output/"train.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            if step==start+1 and args.resume_from:
                write_json(args.output/"RESUME_VERIFIED.json",{"complete":True,"restored_step":start,
                    "first_update":step,"world":world,"optimizer_restored":True,"finite_loss_gradient":True})
        save_now=step%args.save_every==0 or step in (args.first_check,args.steps,args.stop_after)
        eval_now=step%args.eval_every==0 or step in (args.first_check,args.steps,args.stop_after)
        if save_now:
            phase=time.monotonic()
            if rank==0:print("JOINT_TIMELINE_SAVE",step,flush=True)
            io.save(args.output/"checkpoints"/f"step-{step:06d}",model,optimizer,step,recipe)
            save_times.append(time.monotonic()-phase)
        if eval_now:
            phase=time.monotonic()
            evaluate(model,visual,data,transform,tokenizer,prompt,device,args.output/"development"/f"step-{step:06d}")
            eval_times.append(time.monotonic()-phase)
        if rank==0:
            avg=sum(recent)/len(recent);eta=(args.steps-step)*avg
            overhead=(args.steps//args.save_every-step//args.save_every)*(sum(save_times)/len(save_times) if save_times else 0)
            overhead+=(args.steps//args.eval_every-step//args.eval_every)*(sum(eval_times)/len(eval_times) if eval_times else 0)
            progress.update();progress.set_postfix(loss=float(loss_global),s_it=round(avg,2),eta_h=round((eta+overhead)/3600,2))
            write_json(args.output/"progress.json",{"utc":datetime.now(timezone.utc).isoformat(),"step":step,
                "total":args.steps,"seconds_per_step":avg,"eta_train_seconds":eta,
                "eta_total_estimate_seconds":eta+overhead,"global_trajectories":args.global_batch,
                "global_anchors":args.global_batch*args.nodes,"peak_allocated_gib_max":float(memory),
                "scope":"estimate; initialization excluded; save/eval based on measured times"})
        if args.stop_after and step==args.stop_after<args.steps:break
        if step==args.steps and rank==0:
            write_json(args.output/"COMPLETE.json",{"complete":True,"steps":step,"world":world,"mode":args.mode})
    check_endpoint(record);progress.close();dist.barrier();dist.destroy_process_group()


if __name__=="__main__":main()
