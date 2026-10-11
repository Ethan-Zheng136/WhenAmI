"""Loaded A5k teacher plus causal student: real TRAIN forward/backward admission."""

import argparse
import gc
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from stage3_oxe_eval.prepare import sha,write_json
from stage3_perf.runtime import recompute
from whenami_timeline_joint.data import Inputs,tensorize,frozen_features
from whenami_timeline_joint.runtime import gradient_norms
from whenami_timeline_transfer.loader import load_fixed,CHECKPOINT
from whenami_timeline_transfer.contracts import make_prefix
from whenami_timeline_transfer.model import CausalStudent,teacher_target,action_draw,objective
from stage3_oxe_eval50.policy import param_bf16
from whenami_timeline_joint.model import load_joint


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0)
    teacher,visual,transform,tokenizer,prompt,record=load_fixed()
    device=torch.device("cuda:0")
    data=Inputs(dev_limit=1)
    cases=data.batch(1,20261005,1,0)
    inputs,labels=tensorize(cases,transform,tokenizer,prompt,device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        features=frozen_features(visual,inputs["pixels"])
        target=teacher_target(teacher,inputs,features,1)
    prefix=make_prefix(inputs,features,1)
    # Rebuild architecture: deepcopy of the timm/HF model can fail on removable
    # hook weakrefs. Copy tensor values into independent accelerator storage.
    base,unused_visual,_,_,_=load_joint("timeline",42)
    del unused_visual
    base=param_bf16(base).to(device)
    base.load_state_dict(teacher.state_dict(),strict=True)
    gc.collect()
    if any(a.data_ptr()==b.data_ptr() for a,b in zip(base.parameters(),teacher.parameters())):
        raise ValueError("teacher/student parameters alias")
    base.requires_grad_(True);base.vlm.vision_backbone_requires_grad=True
    recompute(base,"llm")
    student=CausalStudent(base).to(device)
    student.eval()
    future_changed={k:v for k,v in inputs.items()}
    future_changed["pixels"]={k:v.clone() for k,v in inputs["pixels"].items()}
    for x in future_changed["pixels"].values():x[:,2]=1000
    altered={k:v.clone() for k,v in features.items()}
    for x in altered.values():x[:,2]=1000
    other_prefix=make_prefix(future_changed,altered,1)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        out=student(prefix)
        changed=student(other_prefix)
    for k in out:
        if not torch.equal(out[k],changed[k]):raise ValueError("future leak in real student: "+k)

    actions=labels["actions"][:,1]
    mask=labels["action_mask"][:,1]
    noisy,tau,target_velocity=action_draw(actions,42)
    teacher_before=teacher.reader.final_layer.linear.weight.detach().clone()
    predictor_before=student.future_predictor.weight.detach().clone()
    core_before=student.base.reader.final_layer.linear.weight.detach().clone()
    # One SGD engineering update avoids claiming an untested full8B Adam setup.
    optimizer=torch.optim.SGD(student.parameters(),lr=1e-3,momentum=0.)
    student.train();optimizer.zero_grad(set_to_none=True)
    began=time.monotonic()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        result=student(prefix,noisy,tau)
        loss,metrics=objective(result,target_velocity,mask,target,weight=.1)
    if not torch.isfinite(loss):raise FloatingPointError("transfer loss")
    loss.backward()
    gradients=gradient_norms(student.base,device)
    forecast_norm=float(sum(p.grad.float().square().sum() for p in student.future_predictor.parameters()
                            if p.grad is not None).sqrt())
    if not all(v>0 and np.isfinite(v) for v in gradients.values()) or not forecast_norm>0:
        raise ValueError("B gradient path missing: "+str(gradients))
    if any(p.grad is not None or p.requires_grad for p in teacher.parameters()):
        raise ValueError("teacher unexpectedly trained")
    norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1.)
    optimizer.step();torch.cuda.synchronize()
    if not torch.equal(teacher_before,teacher.reader.final_layer.linear.weight):
        raise ValueError("student update changed teacher")
    if torch.equal(predictor_before,student.future_predictor.weight):
        raise ValueError("auxiliary head did not update")
    if torch.equal(core_before,student.base.reader.final_layer.linear.weight):
        raise ValueError("action head did not update")
    write_json(args.output/"RESULT.json",{
        "complete":True,"utc":datetime.now(timezone.utc).isoformat(),
        "source_teacher":record,"training_case":cases[0]["provenance"],
        "student_prefix_nodes":prefix.times.shape[1],"teacher_nodes":inputs["times"].shape[1],
        "future_input_perturbation_student_bitwise_invariant":True,
        "teacher_frozen_and_unchanged":True,"action_gradient_groups":gradients,
        "forecast_head_grad_norm":forecast_norm,"loss":float(loss.detach()),
        "loss_components":{k:float(v) for k,v in metrics.items()},
        "grad_norm":float(norm),"update_seconds":time.monotonic()-began,
        "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
        "optimizer":"SGD one engineering update, BF16 base parameters; NOT planned finalB Adam training",
        "scope":"Current-decision transfer interface only; no fullB trainer, rollout result or effectiveness claim",
        "source":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
    })
    print("REAL_PHASE_B_INTERFACE_PASS",gradients,flush=True)


if __name__=="__main__":main()
