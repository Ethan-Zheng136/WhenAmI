"""Actual fixed A10k teacher/student plus causal27dev checks; one SGD step only."""

import argparse
from datetime import datetime,timezone
from pathlib import Path
import gc
import json

import numpy as np
import torch

from stage3_oxe_eval.prepare import write_json,sha
from stage3_oxe_eval50.policy import param_bf16
from stage3_perf.runtime import recompute
from whenami_timeline_joint.model import load_joint
from whenami_timeline_joint.data import Inputs
from whenami_timeline_joint.runtime import gradient_norms
from whenami_timeline_transfer.contracts import select_anchor
from whenami_timeline_b.model import build_student,draw,objective
from whenami_timeline_b.loading import TEACHER,teacher_record,load_model_only
from whenami_timeline_b.data import batch,student_inputs
from whenami_timeline_b.evaluate import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device("cuda:0")
    source=teacher_record()
    teacher,visual,transform,tokenizer,prompt=load_joint("timeline",42)
    teacher=param_bf16(teacher.requires_grad_(False).eval()).to(device)
    teacher.vlm.vision_backbone_requires_grad=False
    visual=visual.to(device).requires_grad_(False).eval()
    load_model_only(teacher)
    student,extra_visual,_,_,_=build_student(42)
    del extra_visual
    student=param_bf16(student).to(device)
    # Copy already loaded A model tensors to independent student storage, retaining
    # auxiliary identity initialization. No second large disk load.
    mismatch=student.load_state_dict(teacher.state_dict(),strict=False)
    if set(mismatch.missing_keys)!={"forecast_head.weight","forecast_head.bias"} or mismatch.unexpected_keys:
        raise ValueError("teacher/student tensor mapping")
    for name,p in teacher.named_parameters():
        if p.data_ptr()==dict(student.named_parameters())[name].data_ptr():raise ValueError("alias")
    student.requires_grad_(True);student.vlm.vision_backbone_requires_grad=True
    # Auxiliary head FP32 to test an actual update without BF16 rounding masking it.
    student.forecast_head.float()
    recompute(student,"llm")
    gc.collect()
    data=Inputs(dev_limit=1)
    cases=data.batch(1,20261005,2,0)
    full,features,prefix,label=batch(cases,visual,transform,tokenizer,prompt,device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        encoded=teacher(**full,**features,visible=torch.ones_like(full["times"],dtype=torch.bool),condition_only=True)
        target_condition=select_anchor(encoded,1,3,1)["conditions"].detach()
    noisy,tau,target=draw(label.actions,42)
    report={}
    optimizer=torch.optim.SGD(student.parameters(),lr=1e-3,momentum=0)
    for variant in ("action_only","condition_align"):
        optimizer.zero_grad(set_to_none=True);student.train();torch.manual_seed(123)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=student(prefix=prefix,noisy_actions=noisy,tau=tau)
            loss,metrics=objective(output,target,label.valid,target_condition,variant)
        loss.backward()
        gradients=gradient_norms(student,device)
        if not all(np.isfinite(v) and v>0 for v in gradients.values()):raise ValueError("B body gradients")
        aux=sum(float(p.grad.float().square().sum()) for p in student.forecast_head.parameters() if p.grad is not None)**.5
        if variant=="action_only" and aux!=0:raise ValueError("teacher target trained B0 auxiliary")
        if variant=="condition_align" and aux<=0:raise ValueError("B1 alignment gradient absent")
        report[variant]={"losses":{k:float(v) for k,v in metrics.items()},"gradients":gradients,"aux_gradient":aux}
    # One actual update to B1; teacher remains unchanged.
    probe=teacher.reader.final_layer.linear.weight.detach().clone()
    original=student.forecast_head.weight.detach().clone()
    torch.nn.utils.clip_grad_norm_(student.parameters(),1.);optimizer.step()
    if torch.equal(original,student.forecast_head.weight):raise ValueError("no B update")
    if not torch.equal(probe,teacher.reader.final_layer.linear.weight):raise ValueError("teacher changed")
    if any(p.grad is not None for p in teacher.parameters()):raise ValueError("teacher received gradient")
    student.zero_grad(set_to_none=True);student.eval()
    # Change full future pixels/language before the independently built prefix.
    import copy
    changed=copy.deepcopy(cases)
    changed[0]["observations"]["images"][2]=255-changed[0]["observations"]["images"][2]
    changed[0]["observations"]["instructions"][2]="different hidden future instruction"
    p1,_=student_inputs(cases,visual,transform,tokenizer,prompt,device)
    p2,_=student_inputs(changed,visual,transform,tokenizer,prompt,device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        o1=student(prefix=p1,condition_only=True);o2=student(prefix=p2,condition_only=True)
    if not all(torch.equal(o1[k],o2[k]) for k in o1):raise ValueError("future leaked in actual B input")
    evaluate(student,visual,data,transform,tokenizer,prompt,device,args.output/"causal_dev27")
    write_json(args.output/"RESULT.json",{"complete":True,"utc":datetime.now(timezone.utc).isoformat(),
        "teacher":source,"arms":report,"actual_A10k_model_loaded":True,
        "student_teacher_nonaliased":True,"future_raw_image_and_language_invariance":True,
        "teacher_frozen":True,"causal_dev_domains":27,"actual_update":"one BF16 body SGD B1 engineering update, not full Adam admission",
        "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
        "source":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
        "scope":"computation/causality admission only, no learned B effectiveness claim"})
    print("B10K_REAL_INTERFACE_PASS",report,flush=True)


if __name__=="__main__":main()
