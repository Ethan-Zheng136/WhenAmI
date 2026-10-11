"""Actual four B20k parents + original frozen teachers, real next-step backward.

No local full-Adam resume claim: actual Adam/RNG is audited separately, tiny4-rank
restoration is tested, and full320 restoration/replay is mandatory on submission.
"""
import argparse
import gc
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

from stage3_oxe_eval.prepare import sha,write_json
from stage3_oxe_eval50.policy import param_bf16
from stage3_perf.runtime import recompute
from whenami_timeline_b.model import build_student,draw,objective
from whenami_timeline_b_handoff.runtime import load_dcp
from whenami_timeline_b.loading import load_model_only
from whenami_timeline_b.data import batch
from whenami_timeline_joint.data import Inputs
from whenami_timeline_joint.model import load_joint
from whenami_timeline_joint.runtime import gradient_norms
from whenami_timeline_transfer.contracts import select_anchor
from whenami_b_continue.contracts import PARENTS,source_record,START


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device("cuda:0")
    student,visual,transform,tokenizer,prompt=build_student(42)
    student=param_bf16(student).to(device)
    visual=visual.to(device).requires_grad_(False).eval()
    teacher,unused,_,_,_=load_joint("timeline",42);del unused
    teacher=param_bf16(teacher.requires_grad_(False).eval()).to(device)
    teacher.vlm.vision_backbone_requires_grad=False
    for name,p in teacher.named_parameters():
        if p.data_ptr()==dict(student.named_parameters())[name].data_ptr():
            raise ValueError("teacher/student storage alias")
    data=Inputs(dev_limit=1)
    cases=data.batch(1,42,START+1,0)
    full,features,prefix,label=batch(cases,visual,transform,tokenizer,prompt,device)
    # All four branches use the same actual first continuation example/noise.
    noisy,tau,target=draw(label.actions,42+(START+1)*320)
    recompute(student,"llm")
    report={};last_teacher=None
    for case in PARENTS:
        source=source_record(case)
        student.zero_grad(set_to_none=True)
        loaded=load_dcp(student,source["path"])
        if source["teacher"]["path"]!=last_teacher:
            load_model_only(teacher,Path(source["teacher"]["path"]))
            last_teacher=source["teacher"]["path"]
        teacher.eval();student.train()
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            enc=teacher(**full,**features,visible=torch.ones_like(full["times"],dtype=torch.bool),condition_only=True)
            target_condition=select_anchor(enc,1,3,1)["conditions"].detach()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=student(prefix=prefix,noisy_actions=noisy,tau=tau)
            loss,metrics=objective(out,target,label.valid,target_condition,source["variant"],.1)
        if not torch.isfinite(loss):raise ValueError("nonfinite actual continuation objective")
        loss.backward()
        grads=gradient_norms(student,device)
        if not all(np.isfinite(v) and v>0 for v in grads.values()):
            raise ValueError("actual B body gradient missing")
        aux=sum(float(p.grad.float().square().sum()) for p in student.forecast_head.parameters() if p.grad is not None)**.5
        if (source["variant"]=="action_only" and aux!=0) or (source["variant"]=="condition_align" and aux<=0):
            raise ValueError("wrong auxiliary objective")
        if any(p.grad is not None for p in teacher.parameters()):
            raise ValueError("teacher got a gradient")
        report[case]={"complete":True,"parent":source,"loaded_B_entries":len(student.state_dict()),
                      "teacher_step":source["teacher"]["step"],
                      "metrics":{k:float(v) for k,v in metrics.items()},
                      "gradients":grads,"forecast_head_gradient":aux}
        print("REAL_PARENT_BACKWARD_PASSED",case,report[case]["metrics"],flush=True)
        del loss,out,enc
        student.zero_grad(set_to_none=True);gc.collect()
    write_json(a.output/"RESULT.json",{"complete":True,"utc":datetime.now(timezone.utc).isoformat(),
        "cases":report,"teacher_frozen":True,"student_teacher_private":True,
        "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
        "source":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
        "scope":"Four actual B20k model loads and corresponding teacher forward/backward; no local full-Adam update, no training-effectiveness claim."})


if __name__=="__main__":main()
