"""Actual B1/A50k load, original-B parity, causal windows, real gradients and SGD.

Engineering proof only. No full-FP32-Adam single-device test or learned-performance claim.
"""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from stage3_oxe_eval.prepare import sha, write_json
from stage3_oxe_eval50.policy import param_bf16
from stage3_perf.runtime import recompute
from whenami_timeline_joint.model import load_joint
from whenami_timeline_joint.runtime import gradient_norms
from whenami_timeline_b.model import Student
from whenami_revision_control.model import build_student, draw, objective, VARIANTS
from whenami_revision_control.loading import STUDENT, TEACHER, source_record, load_model
from whenami_revision_control.data import Inputs, batch, teacher_prefix, observed_case
from whenami_revision_control.evaluate import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device("cuda:0")
    sources=source_record()
    student,visual,transform,tokenizer,prompt=build_student("revised_nodes",42)
    student=param_bf16(student).to(device)
    visual=visual.to(device).requires_grad_(False).eval()
    loaded_student=load_model(student,STUDENT,student=True)
    teacher,unused_visual,_,_,_=load_joint("timeline",42)
    del unused_visual
    teacher=param_bf16(teacher.requires_grad_(False).eval()).to(device)
    teacher.vlm.vision_backbone_requires_grad=False
    loaded_teacher=load_model(teacher,TEACHER)
    for name,value in teacher.named_parameters():
        if value.data_ptr()==dict(student.named_parameters())[name].data_ptr():
            raise ValueError("teacher/student storage aliases")
    recompute(student,"llm")
    data=Inputs(dev_limit=1)
    cases=data.batch(1,20261007,2,0)
    inp,label,observed,features=batch(cases,visual,transform,tokenizer,prompt,device)
    previous,pf=teacher_prefix(cases,visual,transform,tokenizer,prompt,device,2)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        pre=teacher(**previous,**pf,visible=torch.ones_like(previous["times"],dtype=torch.bool),condition_only=True)
        post=teacher(**observed,**features,visible=torch.ones_like(observed["times"],dtype=torch.bool),condition_only=True)
        before=pre["conditions"].reshape(1,2,16,32).detach()
        after=post["conditions"].reshape(1,3,16,32)[:,:2].detach()
    noisy,tau,target=draw(label.actions,42)
    # Initial action parity, before making the new output layer FP32 for visible SGD.
    student.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        original=Student.forward(student,prefix=inp.base,noisy_actions=noisy,tau=tau)
        for variant in VARIANTS:
            student.variant=student.memory.mode=variant
            output=student(inputs=inp,noisy_actions=noisy,tau=tau)
            for k in ("conditions","reader_z","velocity"):
                if not torch.equal(output[k],original[k]):raise ValueError("original-B zero-residual parity: "+k)
    student.bridge.output.float()
    optimizer=torch.optim.SGD([p for p in student.parameters() if p.requires_grad],lr=1e-3)
    arms={}
    for variant in VARIANTS:
        student.variant=student.memory.mode=variant
        optimizer.zero_grad(set_to_none=True);student.train();torch.manual_seed(123)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=student(inputs=inp,noisy_actions=noisy,tau=tau)
            loss,metrics=objective(output,target,label.valid,before,after)
        loss.backward()
        gradients=gradient_norms(student,device)
        for name,mod in (("memory",student.memory),("bridge",student.bridge)):
            gradients[name]=sum(float(p.grad.float().square().sum()) for p in mod.parameters() if p.grad is not None)**.5
        if not all(np.isfinite(v) and v>0 for v in gradients.values()):
            raise ValueError("actual main/new gradient group missing")
        arms[variant]={"losses":{k:float(v) for k,v in metrics.items()},"gradients":gradients}
    old_bridge=student.bridge.output.weight.detach().clone()
    teacher_probe=teacher.reader.final_layer.linear.weight.detach().clone()
    torch.nn.utils.clip_grad_norm_(student.parameters(),1.);optimizer.step()
    if torch.equal(old_bridge,student.bridge.output.weight):raise ValueError("no actual SGD update")
    if not torch.equal(teacher_probe,teacher.reader.final_layer.linear.weight):
        raise ValueError("teacher updated")
    if any(p.grad is not None for p in teacher.parameters()):raise ValueError("teacher got gradient")
    student.zero_grad(set_to_none=True);student.eval()
    # Controlled open-bridge intervention proves a path, NOT that SGD improved performance.
    saved=student.bridge.output.weight.detach().clone()
    with torch.no_grad():student.bridge.output.weight.normal_(std=.02)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=student(inputs=inp,noisy_actions=noisy,tau=tau)
        out["node_conditions"].retain_grad()
        action=(out["velocity"].float()-target).square().mean()
    action.backward()
    old_grad=float(out["node_conditions"].grad[:,:-1].float().abs().sum())
    if old_grad<=0:raise ValueError("action loss does not reach revised old-node estimates")
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        changed=student(inputs=inp,noisy_actions=noisy,tau=tau,unrevised_action=True)
    intervention=float((out["velocity"].float()-changed["velocity"].float()).abs().max())
    if intervention<=0:raise ValueError("old-node readout intervention has no action effect")
    with torch.no_grad():student.bridge.output.weight.copy_(saved)
    student.zero_grad(set_to_none=True)
    # Later raw evidence and the last target's expert actions cannot enter earlier decisions.
    tampered=copy.deepcopy(cases)
    tampered[0]["observations"]["images"][2]=255-tampered[0]["observations"]["images"][2]
    tampered[0]["observations"]["instructions"][2]="hidden later instruction with a different length"
    changed_evidence=tampered
    early1,_,_,_=batch([observed_case(cases[0],0,2)],visual,transform,tokenizer,prompt,device)
    early2,_,_,_=batch([observed_case(tampered[0],0,2)],visual,transform,tokenizer,prompt,device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        o1=student(inputs=early1,condition_only=True);o2=student(inputs=early2,condition_only=True)
    if not torch.equal(o1["conditions"],o2["conditions"]):raise ValueError("later raw evidence leaked")
    tampered=copy.deepcopy(cases);tampered[0]["labels"]["actions"][2]=torch.nan
    no_label,_,_,_=batch(tampered,visual,transform,tokenizer,prompt,device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        o1=student(inputs=inp,condition_only=True);o2=student(inputs=no_label,condition_only=True)
    if not torch.equal(o1["conditions"],o2["conditions"]):raise ValueError("current/future target commands leaked")
    # A's earlier-prefix target itself must not see the newly arriving third image.
    pi,pf2=teacher_prefix(changed_evidence,visual,transform,tokenizer,prompt,device,2)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        repeated=teacher(**pi,**pf2,visible=torch.ones_like(pi["times"],dtype=torch.bool),condition_only=True)
    if not torch.equal(repeated["conditions"],pre["conditions"]):raise ValueError("teacher prefix contract")
    evaluate(student,visual,data,transform,tokenizer,prompt,device,args.output/"causal_dev27")
    write_json(args.output/"RESULT.json",{
        "complete":True,"utc":datetime.now(timezone.utc).isoformat(),"sources":sources,
        "student_load":loaded_student,"teacher_load":loaded_teacher,"arms":arms,
        "initial_B_condition_cognition_velocity_bitwise_equal":True,
        "teacher_student_nonaliased":True,"teacher_frozen":True,
        "later_evidence_and_future_target_action_invariance":True,
        "controlled_open_bridge_old_node_action_gradient":old_grad,
        "controlled_open_bridge_velocity_intervention_max_abs":intervention,
        "causal_dev_domains":27,"actual_update":"one BF16-body SGD update; bridge output FP32; not full Adam/320-rank admission",
        "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
        "source":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
        "scope":"Engineering test of a three-observed-node LAST-action protocol; no learned-effectiveness claim"})
    print("REVISION_REAL_SMOKE_PASSED",arms,flush=True)


if __name__=="__main__":main()
