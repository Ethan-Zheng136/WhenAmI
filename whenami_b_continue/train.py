"""B0/B1 fixed-teacher causal transfer: full VLM FSDP and gated sharded restarts."""

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

from stage3.common import BASE,REFERENCE
from stage3_oxe_eval.prepare import sha,write_json
from stage3_perf.checkpoint import CheckpointIO
from whenami_timeline_joint.runtime import topology,wrap,learning_rate,gradient_norms
from whenami_timeline_joint.model import load_joint
from whenami_timeline_joint.data import Inputs,DEV,INDEX
from whenami_timeline_transfer.contracts import select_anchor
from whenami_timeline_b.model import build_student,draw,objective
from whenami_timeline_b.loading import load_model_only
from whenami_b_continue.contracts import PARENTS,START,END,SCHEDULE,source_record,continuation_lr,restore
from whenami_b_continue.replay_check import compare_saved
from whenami_timeline_b.data import batch
from whenami_timeline_b.evaluate import evaluate
from tqdm import tqdm

VARIANTS=("action_only","condition_align")


def fingerprints():
    dirs=("whenami_b_continue","whenami_timeline_b50","whenami_timeline_b","whenami_timeline_transfer","whenami_timeline_joint",
          "whenami_timeline_a","whenami_revision","stage3","stage3_scale","stage3_perf","evaluation")
    return {str(p.relative_to(BASE)):sha(p) for d in dirs for p in (BASE/d).glob("*.py")}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--case",choices=tuple(PARENTS),required=True)
    p.add_argument("--stop-after",type=int,default=0)
    p.add_argument("--resume-from",type=Path)
    args=p.parse_args()
    args.steps=END;args.micro=1;args.global_batch=320;args.seed=42
    args.lr=1e-5;args.aux_lr=1e-4;args.alignment_weight=.1
    args.sharding="hybrid_zero2";args.activation="llm"
    args.save_every=2500;args.eval_every=1000;args.first_check=START+100
    args.dev_limit=0;args.variant=PARENTS[args.case][3]
    rank,world=int(os.environ.get("RANK",0)),int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0));device=torch.device("cuda",local)
    if world!=320 or int(os.environ.get("LOCAL_WORLD_SIZE",1))!=16:
        p.error("continuation requires the SAME 20x16=320 topology")
    if args.stop_after and not START<args.stop_after<=END:
        p.error("stop-after must be an absolute update in (20000,50000]")
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
    payload=[None]
    if rank==0:
        try:payload[0]={"record":source_record(args.case)}
        except Exception as exc:payload[0]={"error":repr(exc)}
    dist.broadcast_object_list(payload,src=0)
    if "error" in payload[0]:raise ValueError(payload[0]["error"])
    source=payload[0]["record"]
    args.lr,args.aux_lr=source["parent_terminal_lr"]
    parent_meta=json.loads((Path(source["path"])/"COMMITTED.json").read_text())
    core,visual,transform,tokenizer,prompt=build_student(args.seed)
    data=Inputs(dev_limit=args.dev_limit)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ("output","resume_from","stop_after")}
    recipe={"schema":"B_explicit_20k_to50k_continuation_v1","settings":settings,"world_size":world,
        "local_world_size":int(os.environ.get("LOCAL_WORLD_SIZE",1)),"source":fingerprints(),
        "reference_source":{str(p.relative_to(REFERENCE)):sha(p) for p in REFERENCE.rglob("*.py")},
        "teacher":source["teacher"],"parent":source,
        "continuation":{"start":START,"end":END,"additional_updates":END-START,"schedule":SCHEDULE,
                        "lr":source["parent_terminal_lr"],"parent_recipe":parent_meta["recipe"]},"train_index_sha256":sha(INDEX),"dev_manifest_sha256":sha(DEV/"MANIFEST.json"),
        "student_trainable_parameters":sum(p.numel() for p in core.parameters() if p.requires_grad),
        "student_prefix":"two actual nodes, current=second; full teacher third node physically removed",
        "teacher_compute_in_both_arms":True,
        "scope":"Explicit additional30k updates from own completedB20k, restoring Adam/RNG, holding terminal LR; not a from-scratch50k schedule"}
    if rank==0:write_json(args.output/"recipe.json",recipe)
    strategy,groups=topology(args.sharding,device)
    student=wrap(core,device,strategy,groups,args.activation)
    # Restore own full B model/Adam/RNG, never model-only warmstart the student.
    group=groups[0] if groups else None
    teacher_core,unused_visual,_,_,_=load_joint("timeline",args.seed)
    del unused_visual
    teacher_core.requires_grad_(False).eval()
    teacher_core.vlm.vision_backbone_requires_grad=False
    teacher=wrap(teacher_core,device,strategy,groups,"none").eval()
    teacher_warm=load_model_only(teacher,Path(source["teacher"]["path"]),group,True)
    visual=visual.to(device).requires_grad_(False).eval()
    original,aux=[],[]
    for name,param in student.named_parameters():
        clean=name.replace("_fsdp_wrapped_module.","").replace("_checkpoint_wrapped_module.","")
        if param.requires_grad:(aux if clean.startswith("forecast_head.") else original).append(param)
    optimizer=torch.optim.AdamW([{"params":original,"lr":args.lr},{"params":aux,"lr":args.aux_lr}],weight_decay=0.)
    io=CheckpointIO(args.sharding,groups)
    random.seed(args.seed+rank);np.random.seed(args.seed+rank);torch.manual_seed(args.seed+rank)
    start,resume_proof=restore(io,student,optimizer,source,recipe,args.resume_from)
    if args.stop_after and args.stop_after<=start:raise ValueError("requested segment has no new updates")
    resume_proof.update(rank=rank,world=world)
    write_json(args.output/f"restore-rank-{rank:05d}.json",resume_proof)
    if rank==0:
        write_json(args.output/"LOAD_VERIFIED.json",{"teacher":teacher_warm,"student":resume_proof,
            "parent":source,"teacher_manifest":source["teacher"]["committed_sha256"]})
        print("B_CONTINUATION_READY",args.variant,recipe["student_trainable_parameters"],world,flush=True)
    expected_dev=(args.resume_from.parent.parent/"development"/f"step-{start:06d}"
                  if args.resume_from else Path(source["final_dev"]))
    seeds=tuple(json.loads((expected_dev/"RESULT.json").read_text())["noise_seeds"])
    current_dev=args.output/"development"/f"step-{start:06d}"
    evaluate(student,visual,data,transform,tokenizer,prompt,device,current_dev,seeds)
    replay=[None]
    if rank==0:
        try:
            check=compare_saved(expected_dev,current_dev)
            write_json(args.output/"CHECKPOINT_REPLAY_VERIFIED.json",check)
            replay[0]={"complete":True}
        except Exception as exc:replay[0]={"error":repr(exc)}
    dist.broadcast_object_list(replay,src=0)
    if "error" in replay[0]:raise ValueError(replay[0]["error"])
    accum=args.global_batch//(world*args.micro)
    recent=deque(maxlen=100);save_times=[];eval_times=[]
    progress=tqdm(total=args.steps,initial=start,disable=rank!=0,desc="Continue "+args.case)
    for step in range(start+1,args.steps+1):
        begin=time.monotonic();student.train();teacher.eval();visual.eval();optimizer.zero_grad(set_to_none=True)
        for g,terminal in zip(optimizer.param_groups,source["parent_terminal_lr"]):
            g["lr"]=continuation_lr(step,terminal)
        numeric=[]
        for micro in range(accum):
            cases=data.batch(args.micro,args.seed,step,rank,micro)
            full,features,prefix,label=batch(cases,visual,transform,tokenizer,prompt,device)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                encoded=teacher(**full,**features,visible=torch.ones_like(full["times"],dtype=torch.bool),condition_only=True)
                target_condition=select_anchor(encoded,len(cases),3,1)["conditions"].detach()
            noisy,tau,target=draw(label.actions,args.seed+(step*accum+micro)*world+rank)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output=student(prefix=prefix,noisy_actions=noisy,tau=tau)
                loss,metrics=objective(output,target,label.valid,target_condition,args.variant,args.alignment_weight)
            if not torch.isfinite(loss):raise FloatingPointError("B loss")
            (loss/accum).backward()
            numeric.append([float(metrics[k]) for k in ("action_fm","condition_cosine","weighted_alignment","total")])
        audit=step in (start+1,start+2,args.first_check)
        grads=gradient_norms(student,device) if audit else None
        if audit:
            auxnorm=torch.zeros((),device=device)
            for name,param in student.named_parameters():
                if "forecast_head." in name and param.grad is not None:auxnorm+=param.grad.float().square().sum()
            dist.all_reduce(auxnorm);grads["forecast_head"]=float(auxnorm.sqrt())
        norm=student.clip_grad_norm_(1.)
        if not torch.isfinite(norm):raise FloatingPointError("B grad")
        optimizer.step()
        if any(p.grad is not None for p in teacher.parameters()) or any(p.grad is not None for p in visual.parameters()):
            raise ValueError("frozen teacher/visual received gradient")
        torch.cuda.synchronize()
        values=torch.tensor(np.mean(numeric,axis=0),device=device,dtype=torch.float32)
        dist.all_reduce(values);values/=world
        sec=torch.tensor(time.monotonic()-begin,device=device);dist.all_reduce(sec,op=dist.ReduceOp.MAX)
        memory=torch.tensor(torch.cuda.max_memory_allocated()/2**30,device=device);dist.all_reduce(memory,op=dist.ReduceOp.MAX)
        recent.append(float(sec))
        if rank==0:
            row={"step":step,**dict(zip(("action_fm","condition_cosine","weighted_alignment","total"),values.cpu().tolist())),
                 "grad_norm":float(norm),"seconds_max":float(sec),"gradients":grads,
                 "peak_allocated_gib_max":float(memory),"lr":[g["lr"] for g in optimizer.param_groups]}
            with (args.output/"train.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
            if step==start+1:
                write_json(args.output/"RESUME_VERIFIED.json",{"complete":True,"restored_step":start,
                    "first_update":step,"world":world,"B_optimizer_rng_restored":True,"original_parent_step":START,
                    "parent_manifest_sha256":source["committed_sha256"],"lr_schedule":SCHEDULE})
        save_now=step%args.save_every==0 or step in (args.first_check,args.steps,args.stop_after)
        # Every saved checkpoint gets a replay reference, including 22.5k/27.5k
        # that are not multiples of the regular1k evaluation interval.
        eval_now=save_now or step%args.eval_every==0 or step in (args.first_check,args.steps,args.stop_after)
        if save_now:
            begin_phase=time.monotonic()
            io.save(args.output/"checkpoints"/f"step-{step:06d}",student,optimizer,step,recipe)
            save_times.append(time.monotonic()-begin_phase)
        if eval_now:
            begin_phase=time.monotonic()
            evaluate(student,visual,data,transform,tokenizer,prompt,device,
                     args.output/"development"/f"step-{step:06d}",(42,43,44) if step%5000==0 else (42,))
            eval_times.append(time.monotonic()-begin_phase)
        if rank==0:
            average=sum(recent)/len(recent);eta=(args.steps-step)*average
            overhead=(args.steps//args.save_every-step//args.save_every)*(np.mean(save_times) if save_times else 0)
            future_evals=(set(range((step//args.eval_every+1)*args.eval_every,args.steps+1,args.eval_every))
                          |set(range((step//args.save_every+1)*args.save_every,args.steps+1,args.save_every)))
            overhead+=len(future_evals)*(np.mean(eval_times) if eval_times else 0)
            progress.update();progress.set_postfix(loss=float(values[0]),s_it=round(average,2),eta_h=round((eta+overhead)/3600,2))
            write_json(args.output/"progress.json",{"utc":datetime.now(timezone.utc).isoformat(),"step":step,"total":args.steps,
                "seconds_per_step":average,"eta_train_seconds":eta,"eta_total_estimate_seconds":float(eta+overhead),
                "peak_allocated_gib_max":float(memory),"additional_updates_done":step-START,"scope":"B20k->50k; terminal LR held, ETA excludes initialization"})
        if args.stop_after and step==args.stop_after<args.steps:break
        if step==args.steps and rank==0:
            write_json(args.output/"COMPLETE.json",{"complete":True,"steps":step,"world":world,"variant":args.variant,
                "teacher_fixed_step":source["teacher"]["step"],"causal_only_evaluation":True,
                "original_parent_step":START,"additional_updates":END-START,"case":args.case,"schedule":SCHEDULE})
    progress.close();dist.barrier();dist.destroy_process_group()


if __name__=="__main__":main()
