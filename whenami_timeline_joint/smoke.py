"""Actual8B FSDP backward/masking test. No single-PPU full Adam claim."""

import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from stage3.common import DEFAULT_FUTURE,endpoint_record,check_endpoint
from stage3_oxe_eval.prepare import write_json
from whenami_timeline_joint.model import load_joint,MODES
from whenami_timeline_joint.runtime import wrap,topology,gradient_norms
from whenami_timeline_joint.data import Inputs,tensorize,frozen_features
from whenami_timeline_joint.train import fingerprints
from whenami_timeline_a.objectives import noisy_labels,velocity_loss
from whenami_timeline_joint.evaluate import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device("cuda:0")
    dist.init_process_group("nccl",init_method="file://"+str(args.output/"rdzv"),rank=0,world_size=1)
    source=endpoint_record(DEFAULT_FUTURE)
    core,visual,transform,tokenizer,prompt=load_joint("timeline",42)
    params=sum(p.numel() for p in core.parameters() if p.requires_grad)
    strategy,groups=topology("full",device)
    model=wrap(core,device,strategy,groups,"llm")
    visual=visual.to(device).eval()
    data=Inputs(dev_limit=1)
    cases=data.batch(1,42,1,0)
    inputs,labels=tensorize(cases,transform,tokenizer,prompt,device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        features=frozen_features(visual,inputs["pixels"])
    vis=torch.ones_like(inputs["times"],dtype=torch.bool);vis[:,1]=False
    report={}
    for mode in MODES:
        core.timeline.mode=mode
        model.train();model.zero_grad(set_to_none=True);torch.manual_seed(123)
        noisy,tau,target=noisy_labels(labels["actions"],42,draws=2)
        started=time.monotonic()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=model(**inputs,**features,visible=vis,noisy_actions=noisy,tau=tau)
            loss,_=velocity_loss(out,target,labels["action_mask"],draws=2)
        loss.backward()
        norms=gradient_norms(model,device)
        if not all(torch.isfinite(torch.tensor(v)) and v>0 for v in norms.values()):
            raise ValueError("a full-model gradient path is missing: "+str(norms))
        if any(p.grad is not None for p in visual.parameters()):raise ValueError("frozen visual gradient")
        torch.cuda.synchronize()
        report[mode]={"loss":float(loss.detach()),"gradients":norms,
                      "forward_backward_seconds":time.monotonic()-started,
                      "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30}
        print("ACTUAL_FULL_MODEL_BACKWARD",mode,report[mode],flush=True)
        del out,loss,noisy,target
        model.zero_grad(set_to_none=True)
        model.eval()
        altered={**inputs,"pixels":{key:value.clone() for key,value in inputs["pixels"].items()}}
        for value in altered["pixels"].values():value[:,1]=1000
        altered_f={key:value.clone() for key,value in features.items()}
        for value in altered_f.values():value[:,1]=1000
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            a=model(**inputs,**features,visible=vis,condition_only=True)
            b=model(**altered,**altered_f,visible=vis,condition_only=True)
        for key in a:
            if not torch.equal(a[key],b[key]):raise ValueError("actual model hidden-input leak: "+key)
        report[mode]["hidden_pixels_and_features_invariant"]=True
        del a,b,altered,altered_f
    # Test exactly the collective-safe raw-VLM evaluation used by the launcher,
    # on one existing-DEV episode from each of27 domains. No training score claim.
    core.timeline.mode="timeline"
    evaluate(model,visual,data,transform,tokenizer,prompt,device,args.output/"development27")
    check_endpoint(source)
    write_json(args.output/"RESULT.json",{"complete":True,"checked_utc":datetime.now(timezone.utc).isoformat(),
        "trainable_parameters":params,"actual_FSDP":True,"world":1,"arms":report,
        "model_sources":fingerprints(),"actual_full_FP32_Adam_update_tested":False,
        "raw_updated_vlm_development_27_domains_tested":True,
        "note":"full pretrained model backward and masks tested; actual sharded optimizer checkpoint admission still requires cluster"})
    dist.destroy_process_group()


if __name__=="__main__":main()
