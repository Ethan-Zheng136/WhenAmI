"""Actual OXE27 images + frozen pretrained VLM/Past + shared DiT backward checks.

This is a fixed small numerical admission, not a research performance estimate.
"""

from dataclasses import asdict
from pathlib import Path
from datetime import datetime,timezone
import argparse
import copy
import hashlib
import json
import time
import shutil

import numpy as np
import torch
from tqdm import tqdm

from stage3.common import BASE,check_endpoint
from whenami_timeline.backbone import FrozenPastBackbone
from whenami_timeline.data import RawTimelines,feature_batch
from whenami_timeline.model import Timeline,TimelineConfig,MODES
from whenami_timeline.objectives import timeline_objective


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--index",type=Path,default=Path(
        "/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json"))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);device=torch.device("cuda",0);torch.cuda.set_device(device)
    begin=time.monotonic()
    source=Path(__file__).parent;recipe={
        "scope":"REAL raw-image/DiT gradient smoke; fixed27 TRAIN episodes, NOT performance experiment",
        "created_utc":datetime.now(timezone.utc).isoformat(),"modes":MODES,"steps_per_mode":8,"batch_trajectories":2,
        "anchors":3,"fm_draws":2,"lr":1e-4,"timeline_seed":42,"backbone":"frozen existing Past Stage2",
        "source":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob("*.py")},
        "index_sha256":hashlib.sha256(a.index.read_bytes()).hexdigest(),
        "plan_sha256":hashlib.sha256((source/"PLAN_REAL_01.md").read_bytes()).hexdigest(),
    }
    snap=a.output/"source_snapshot";snap.mkdir()
    for name in recipe["source"]:shutil.copy2(source/name,snap/name)
    shutil.copy2(source/"PLAN_REAL_01.md",snap)
    (a.output/"recipe.json").write_text(json.dumps(recipe,indent=2))
    print("LOAD_REAL_PAST",flush=True)
    backbone=FrozenPastBackbone(device);dataset=RawTimelines(a.index)
    cases=[]
    for domain in tqdm(dataset.domains,desc="Real OXE27 temporal encodings"):
        entry,times=dataset.select(42,1,domain=domain)
        sample,labels,info=dataset.materialize(entry,times,backbone.transform,backbone.statistics)
        encoded=backbone.encode(sample)
        assert encoded["h"].shape==(3,16,1024) and encoded["z"].shape==(3,1,4096)
        assert all(torch.isfinite(encoded[k]).all() for k in ("h","z","base_condition"))
        cases.append({"policy":encoded,"labels":labels,"provenance":info})
        print("REAL_TIMELINE",domain,times.tolist(),flush=True)
    torch.save({"cases":cases,"endpoint":backbone.endpoint,"recipe":recipe},a.output/"admission_features.pt")
    # Frozen feature caching ONLY for this numerical comparison. Cluster runner
    # will independently decode current raw samples, never train on these27 forever.
    reports=[];reference_initial=None
    for mode in MODES:
        torch.manual_seed(42);model=Timeline(mode=mode).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.)
        initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if reference_initial is None:reference_initial=initial_state
        else:
            for k in initial_state:torch.testing.assert_close(initial_state[k],reference_initial[k],rtol=0,atol=0)
        losses=[];grad_audit={}
        for step in range(1,9):
            idx=np.random.default_rng(np.random.SeedSequence([42,step])).integers(0,len(cases),size=2)
            inputs,labels=feature_batch(cases,idx,device)
            optimizer.zero_grad();out=model(**inputs)
            if step==1:torch.testing.assert_close(out["conditions"],inputs["base_condition"],rtol=0,atol=0)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss,metrics=timeline_objective(out,inputs,labels,backbone.reader,4242+step,draws=2)
            if not torch.isfinite(loss):raise FloatingPointError("real timeline loss nonfinite")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError("real timeline gradient nonfinite")
            optimizer.step();backbone.assert_frozen()
            losses.append({"step":step,"total":float(loss.detach()),"action_fm":float(metrics["action_fm"]),
                           "transition_mse":float(metrics["transition_mse"]),"per_anchor_fm":metrics["per_anchor_action_fm"].cpu().tolist()})
        # PURE action-loss gradient, AFTER residual output opens. Not auxiliary
        # gradients masquerading as action supervision of the timeline.
        optimizer.zero_grad();out=model(**inputs)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            loss,_=timeline_objective(out,inputs,labels,backbone.reader,6001,draws=2,transition_weight=0)
        loss.backward()
        for name in ("h_input","z_input","transition","backward_attention","backward_update","relevance_gate","bank_attention","readout"):
            values=[p.grad for p in getattr(model,name).parameters() if p.grad is not None]
            grad_audit[name]=sum(float(g.float().square().sum()) for g in values)**.5
        must=("h_input","z_input","readout")
        if mode!="independent":must+=("transition",)
        if mode in ("revision","direct_history"):must+=("backward_attention","backward_update","relevance_gate","bank_attention")
        assert all(grad_audit[k]>0 for k in must),grad_audit
        backbone.assert_frozen()
        with torch.no_grad():
            stable=model(**inputs)["conditions"]
            altered={k:v.clone() for k,v in inputs.items()}
            for k in ("h","z","base_condition","executed_actions"):altered[k][:,2]+=100
            changed=model(**altered)["conditions"]
            torch.testing.assert_close(stable[:,:2],changed[:,:2],rtol=0,atol=0)
            cp=a.output/f"{mode}_smoke.pt"
            torch.save({"timeline":model.state_dict(),"optimizer":optimizer.state_dict(),"config":asdict(model.config),
                        "mode":mode,"steps":8},cp)
            restored=Timeline(mode=mode).to(device)
            restored.load_state_dict(torch.load(cp,map_location=device,weights_only=True)["timeline"],strict=True)
            torch.testing.assert_close(stable,restored(**inputs)["conditions"],rtol=0,atol=0)
        report={"mode":mode,"steps":8,"losses":losses,"action_only_grad_norm":grad_audit,
                "prefix_causality_passed":True,"save_load_conditions_identical":True,
                "trainable_parameters":sum(p.numel() for p in model.parameters())}
        reports.append(report);print("REAL_TIMELINE_BACKWARD_OK",json.dumps(report),flush=True)
        del model,optimizer,restored
    check_endpoint(backbone.endpoint)
    result={"complete":True,"scope":recipe["scope"],"actual_domains":len(cases),"actual_anchors":len(cases)*3,
            "reports":reports,"peak_device_gib":torch.cuda.max_memory_allocated()/2**30,
            "elapsed_seconds":time.monotonic()-begin,"original_endpoint_unchanged":True,
            "future_targets_used_as_policy_input":False,
            "not_method_superiority_evidence":True,"not_full_8b_optimizer_training":True,
            "no_main_dev_or_sealed_test_access":True}
    (a.output/"RESULT.json").write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
