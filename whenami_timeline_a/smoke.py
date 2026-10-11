"""Real27-domain encode and matched3-arm shared-DiT optimization smoke."""

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from stage3.common import BASE,DEFAULT_FUTURE,REFERENCE
from stage3_oxe_eval.prepare import sha,write_json
from whenami_timeline_a.backbone import Foundation
from whenami_timeline_a.data import RawAnchors,stack_cases,visibility,INDEX
from whenami_timeline_a.model import Config,MODES
from whenami_timeline_a.objectives import action_loss,sample_actions


def gradient_report(model):
    groups={"local_qformer":model.local_encoder,"original_dit":model.reader,
            "temporal_input":model.timeline.input,"temporal_attention":model.timeline.blocks,
            "temporal_output":model.timeline.output,"source_cognition":model.timeline.cognition_input}
    return {name:float(sum(p.grad.detach().float().square().sum() for p in module.parameters()
                           if p.grad is not None).sqrt()) for name,module in groups.items()}


def optimizer_for(model):
    return torch.optim.AdamW([
        {"params":model.local_encoder.parameters(),"lr":1e-5},
        {"params":model.reader.parameters(),"lr":1e-5},
        {"params":model.timeline.parameters(),"lr":1e-4},
    ],weight_decay=0.)


@torch.no_grad()
def evaluate_fixture(model,cases,device,seed=97531):
    model.eval()
    values=[]
    # Every domain is scored; no choosing favorable smoke episodes.
    for i,case in enumerate(cases):
        x,y=stack_cases([case],device)
        visible,cutoff=visibility(x["times"],seed+i,hide_interior=True)
        generator=torch.Generator(device=device).manual_seed(seed+i)
        noise=torch.randn((*x["times"].shape,16,7),generator=generator,device=device)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=model(**x,visible=visible,cutoffs=cutoff)
            loss,m=action_loss(model,out,y["actions"],y["action_mask"],seed+i,draws=2)
            sampled=sample_actions(model,out,noise)
        error=(sampled.clamp(-1,1)[...,:6]-y["actions"][...,:6]).square().mean(-1)
        mask=y["action_mask"]
        mse=((error*mask).sum(-1)/mask.sum(-1)).cpu()
        values.append({"dataset":case["provenance"]["dataset"],"action_fm":float(loss),
                       "sampled_mse6":float(mse.mean()),
                       "masked_node_sampled_mse6":float(mse[~visible.cpu()].mean()),
                       "sampled":sampled.cpu(),"noise":noise.cpu()})
    return values


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--nodes",type=int,default=3)
    args=p.parse_args()
    if args.steps<2:raise ValueError("at least2 updates for zero-initialized branch gradient admission")
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.cuda.set_device(0)
    f=Foundation()
    raw=RawAnchors(nodes=args.nodes)
    cases=[];first_observations=None
    for i,domain in enumerate(tqdm(raw.domains,desc="Timeline-A real27 encode")):
        entry,times=raw.select(20261004,domain=domain)
        observations,labels,provenance=raw.materialize(entry,times,f.statistics)
        inputs=f.encode(observations)
        cases.append({"inputs":inputs,"labels":labels,"provenance":provenance})
        if first_observations is None:first_observations=observations
        print("TIMELINE_A_RAW",domain,provenance["source_times"],flush=True)
    del raw;gc.collect()
    torch.save(cases,args.output/"train_fixture.pt")
    write_json(args.output/"fixture_manifest.json",{
        "complete":True,"domains":len(cases),"anchors":len(cases)*args.nodes,
        "index_sha256":sha(INDEX),"fixture_sha256":sha(args.output/"train_fixture.pt"),
        "provenance":[c["provenance"] for c in cases],
        "scope":"TRAIN-only shared smoke fixture; no held-out effectiveness claim"})

    # Actual raw-image evidence tests, after frozen vision and cognition paths.
    changed={**first_observations,"images":first_observations["images"].copy()}
    changed["images"][1]=255-changed["images"][1]
    alt=f.encode(changed)
    raw_encoding_is_local={k:bool(torch.equal(cases[0]["inputs"][k][0],alt[k][0]))
                           for k in ("patches","vae","z")}
    if not all(raw_encoding_is_local.values()):raise ValueError("raw encoder leaks across nodes")
    reports={};param_counts={};initial_signatures={}
    for mode in MODES:
        torch.manual_seed(42)
        model=f.model(mode).train()
        opt=optimizer_for(model)
        param_counts[mode]=sum(p.numel() for p in model.parameters())
        initial_signatures[mode]={k:float(v.float().sum()) for k,v in model.state_dict().items()
                                  if k.startswith("timeline.")}
        initial=evaluate_fixture(model,cases,f.device)
        rows=[];gradient_max={}
        for step in tqdm(range(1,args.steps+1),desc="Timeline-A "+mode):
            began=time.monotonic()
            # Two distinct domains/update, exact same pair/order in all arms.
            ids=[(2*(step-1))%len(cases),(2*(step-1)+1)%len(cases)]
            x,y=stack_cases([cases[i] for i in ids],f.device)
            mask,cutoff=visibility(x["times"],42+step,hide_interior=step%2==0)
            torch.manual_seed(42000+step)
            model.train();opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                encoded=model(**x,visible=mask,cutoffs=cutoff)
                loss,m=action_loss(model,encoded,y["actions"],y["action_mask"],42+step,draws=2)
            if not torch.isfinite(loss):raise FloatingPointError("loss")
            loss.backward()
            gradients=gradient_report(model)
            gradient_max={k:max(gradient_max.get(k,0.),v) for k,v in gradients.items()}
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError("gradient")
            opt.step();f.check_frozen();torch.cuda.synchronize()
            row={"step":step,"mode":mode,"loss":float(loss),"grad_norm":float(norm),
                 "seconds":time.monotonic()-began,"gradients":gradients,
                 "domains":[cases[i]["provenance"]["dataset"] for i in ids],
                 "hidden_nodes":int((~mask).sum())}
            rows.append(row);print("TIMELINE_A_UPDATE",row,flush=True)
        if not all(v>0 and np.isfinite(v) for v in gradient_max.values()):
            raise ValueError("missing learned branch gradient "+str(gradient_max))
        final=evaluate_fixture(model,cases,f.device)
        # Hidden raw-frame replacement cannot affect ANY condition or reader z.
        x,_=stack_cases([cases[0]],f.device)
        alternative={k:alt[k][None].to(f.device) for k in x}
        hidden,cutoff=visibility(x["times"],42,hide_interior=True)
        hidden[:]=True
        hidden[:,1]=False  # the actual raw image changed above is node1
        model.eval()
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
            a=model(**x,visible=hidden,cutoffs=cutoff)
            b=model(**alternative,visible=hidden,cutoffs=cutoff)
        for key in ("conditions","reader_z"):
            if not torch.equal(a[key],b[key]):raise ValueError("raw masked node leaked: "+mode+"/"+key)
        torch.save({"initial":initial,"final":final},args.output/(mode+"_samples.pt"))
        write_json(args.output/(mode+"_updates.json"),rows)
        reports[mode]={
            "updates":args.steps,"trainable_parameters":param_counts[mode],"gradient_max":gradient_max,
            "initial_smoke_action_fm":float(np.mean([v["action_fm"] for v in initial])),
            "final_smoke_action_fm":float(np.mean([v["action_fm"] for v in final])),
            "initial_smoke_sampled_mse6":float(np.mean([v["sampled_mse6"] for v in initial])),
            "final_smoke_sampled_mse6":float(np.mean([v["sampled_mse6"] for v in final])),
            "raw_hidden_frame_invariance":True,
        }
        del opt,model;gc.collect();torch.cuda.empty_cache()
    if len(set(param_counts.values()))!=1:raise ValueError("unequal capacity")
    if any(initial_signatures[m]!=initial_signatures[MODES[0]] for m in MODES):raise ValueError("unequal new initialization")
    f.check_frozen()
    write_json(args.output/"RESULT.json",{
        "complete":True,"utc":datetime.now(timezone.utc).isoformat(),
        "models":reports,"raw_encoding_is_local":raw_encoding_is_local,
        "source":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
        "endpoint":f.endpoint,"config":asdict(Config()),
        "notes":["Real images and full shared pretrained DiT, not synthetic low-D actions.",
                 "All27 training-fixture cases scored; these are not validation/generalization results.",
                 "Phase-A future privileged; no causal deployment claim.",
                 "No new physical transition operator or novel-method superiority established."]})


if __name__=="__main__":main()
