"""Actual two C20k parents + original frozen teachers, real next-step backward.

No local full-Adam resume claim: actual Adam/RNG is audited separately, tiny4-rank
restoration is tested, and full320 restoration/replay is mandatory on submission.
"""
import argparse
import copy
import gc
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

from stage3_oxe_eval.prepare import sha,write_json
from stage3.common import DEFAULT_FUTURE
from stage3_oxe_eval50.policy import param_bf16
from stage3_perf.runtime import recompute
from whenami_timeline_b.model import draw
from whenami_timeline_c.model import objective,TimelineStudent
from whenami_timeline_c.loading import build_student,load_model
from whenami_timeline_b_handoff.runtime import build as cold_build
from whenami_timeline_c.data import batch,Inputs
from whenami_timeline_joint.model import JointTimeline
from whenami_timeline_joint.runtime import gradient_norms
from whenami_timeline_transfer.contracts import select_anchor
from whenami_c_continue.contracts import PARENTS,source_record,START,load_complete_model


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device("cuda:0")
    # Construct body from config instead of reading an irrelevant Stage1 model
    # twice. Every student/teacher tensor is then strictly overwritten by its
    # actual C20k/A50k checkpoint. No change to cluster training construction.
    config=json.loads((DEFAULT_FUTURE.parents[1]/"config.json").read_text())
    base,visual,transform,tokenizer,prompt=cold_build(config,device)
    student=TimelineStudent(base.vlm,base.local_encoder,base.reader,"future_aux",4)
    del base
    student=param_bf16(student).to(device)
    visual=visual.to(device).requires_grad_(False).eval()
    teacher=JointTimeline(copy.deepcopy(student.vlm),copy.deepcopy(student.local_encoder),
                          copy.deepcopy(student.reader),"timeline")
    teacher=param_bf16(teacher.requires_grad_(False).eval()).to(device)
    teacher.vlm.vision_backbone_requires_grad=False
    for name,p in teacher.named_parameters():
        if p.data_ptr()==dict(student.named_parameters())[name].data_ptr():
            raise ValueError("teacher/student storage alias")
    data=Inputs(future_gap=4,dev_limit=1)
    cases=data.batch(1,42,START+1,0)
    full,features,prefix,label=batch(cases,visual,transform,tokenizer,prompt,device)
    # Both branches use the same actual first continuation example/noise.
    noisy,tau,target=draw(label.actions,42+(START+1)*320)
    recompute(student,"llm")
    report={};last_teacher=None
    for case in PARENTS:
        source=source_record(case)
        student.zero_grad(set_to_none=True)
        student.variant=source["variant"]
        loaded=load_complete_model(student,source["path"])
        if float(student.read_scale)!=source["read_scale"]:raise ValueError("C read_scale not loaded")
        student.set_step(START+1,1000)
        if float(student.read_scale)!=source["read_scale"]:raise ValueError("C read_scale restarted")
        if source["teacher"]["path"]!=last_teacher:
            load_model(teacher,Path(source["teacher"]["path"]))
            last_teacher=source["teacher"]["path"]
        teacher.eval();student.train()
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            enc=teacher(**full,**features,visible=torch.ones_like(full["times"],dtype=torch.bool),condition_only=True)
            target_condition=enc["conditions"].reshape(1,3,16,32)[:,1:3].detach()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=student(prefix=prefix,noisy_actions=noisy,tau=tau)
            loss,metrics=objective(out,target,label.valid,target_condition,.1)
        if not torch.isfinite(loss):raise ValueError("nonfinite actual continuation objective")
        loss.backward()
        grads=gradient_norms(student,device)
        if not all(np.isfinite(v) and v>0 for v in grads.values()):
            raise ValueError("actual B body gradient missing")
        if any(p.grad is not None for p in teacher.parameters()):
            raise ValueError("teacher got a gradient")
        report[case]={"complete":True,"parent":source,"loaded_C_entries":len(student.state_dict()),
                      "teacher_step":source["teacher"]["step"],
                      "metrics":{k:float(v) for k,v in metrics.items()},
                      "gradients":grads,"read_scale":float(student.read_scale),"condition_tokens":out["conditions"].shape[1]}
        print("REAL_C_PARENT_BACKWARD_PASSED",case,report[case]["metrics"],flush=True)
        del loss,out,enc
        student.zero_grad(set_to_none=True);gc.collect()
    write_json(a.output/"RESULT.json",{"complete":True,"utc":datetime.now(timezone.utc).isoformat(),
        "cases":report,"teacher_frozen":True,"student_teacher_private":True,
        "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
        "source":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
        "scope":"Two actual C20k model loads and corresponding teacher forward/backward; no local full-Adam update, no training-effectiveness claim."})


if __name__=="__main__":main()
