import copy
import json
import multiprocessing
import os
from pathlib import Path
from functools import partial

import pytest
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

from stage3_perf.checkpoint import CheckpointIO
from stage3_perf.runtime import topology
from whenami_timeline_joint.test_joint import fixture,CFG
from whenami_timeline_transfer.contracts import make_prefix,select_anchor
from whenami_timeline_b.model import Student,draw,objective
from whenami_timeline_b50.loading import load_model_only
from whenami_timeline_a.model import ReadEvidence

torch.set_num_threads(1)


def build():
    a,x=fixture()
    student=Student(a.vlm,a.local_encoder,a.reader,"timeline",CFG)
    student.timeline.load_state_dict(a.timeline.state_dict(),strict=True)
    features={k:x.pop(k) for k in ("patches","vae")};x.pop("visible")
    return student,x,features


def test_forecast_head_decoupled():
    s,x,f=build();s.eval();prefix=make_prefix(x,f,1)
    noisy,tau,_=draw(torch.ones(2,16,7),42)
    a=s(prefix,noisy,tau)
    with torch.no_grad():s.forecast_head.weight.add_(20)
    b=s(prefix,noisy,tau)
    torch.testing.assert_close(a["velocity"],b["velocity"],rtol=0,atol=0)
    assert not torch.allclose(a["forecast"],b["forecast"])


def test_full_future_cannot_enter_student():
    s,x,f=build();s.eval()
    y,g=copy.deepcopy(x),copy.deepcopy(f)
    for v in y["pixels"].values():v[:,2]=torch.nan
    for v in g.values():v[:,2]=torch.nan
    y["times"][:,2]+=10000
    y["input_ids"].reshape(2,3,-1)[:,2]=41
    a=s(make_prefix(x,f,1),condition_only=True)
    b=s(make_prefix(y,g,1),condition_only=True)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    with pytest.raises(TypeError):s(x,condition_only=True)


@pytest.mark.parametrize("variant",("action_only","condition_align"))
def test_gradients_and_teacher_detach(variant):
    s,x,f=build()
    labels=torch.ones(2,16,7,requires_grad=True)
    noisy,tau,target=draw(labels,42)
    out=s(make_prefix(x,f,1),noisy,tau)
    teacher=torch.randn_like(out["forecast"],requires_grad=True)
    loss,metrics=objective(out,target,torch.ones(2,16,dtype=torch.bool),teacher,variant)
    loss.backward()
    assert labels.grad is None and teacher.grad is None
    for mod in (s.vlm.vision_backbone,s.vlm.projector,s.vlm.llm_backbone,s.local_encoder,s.timeline,s.reader):
        assert sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)>0
    if variant=="condition_align":assert s.forecast_head.weight.grad.abs().sum()>0
    else:
        assert s.forecast_head.weight.grad is None
        assert float(metrics["weighted_alignment"])==0


def test_decoder_root_matches_current_action():
    s,x,f=build();s.eval();prefix=make_prefix(x,f,1)
    noisy,tau,_=draw(torch.ones(2,16,7),5,draws=1)
    a=s(prefix,noisy,tau)
    enc=s(prefix,condition_only=True)
    b=s(noisy_actions=noisy,tau=tau,conditioning={k:enc[k] for k in ("conditions","reader_z")})
    torch.testing.assert_close(a["velocity"],b["velocity"],rtol=0,atol=0)


def _worker(rank,root):
    torch.set_num_threads(1);os.environ["LOCAL_WORLD_SIZE"]="2"
    dist.init_process_group("gloo",init_method="file://"+str(Path(root)/"rdzv"),rank=rank,world_size=4)
    strategy,groups=topology("hybrid_zero2",torch.device("cpu"));io=CheckpointIO("hybrid_zero2",groups)
    def wrapped(module):
        allowed={id(module.vlm.vision_backbone),id(module.vlm.projector),
                 id(module.vlm.llm_backbone),id(module.reader)}
        def policy(module,recurse,nonwrapped_numel):
            if recurse:return True
            return id(module) in allowed or isinstance(module,ReadEvidence)
        return FSDP(module,device_id=torch.device("cpu"),use_orig_params=True,
                    # Match production's callable block boundaries. Do not wrap
                    # Local itself: encode_queries is a helper, not its forward,
                    # so doing so would bypass FSDP unsharding of query params.
                    # Likewise do not wrap MHA.out_proj Linear: PyTorch MHA uses
                    # its weight directly instead of invoking its module forward.
                    # TinyVLM's SimpleNamespace test output is not traversed by
                    # FSDP hooks; real runtime wraps its tensor-returning blocks.
                    auto_wrap_policy=policy,
                    sharding_strategy=strategy,process_group=groups)
    a,x=fixture()
    a=wrapped(a)
    ao=torch.optim.AdamW(a.parameters(),lr=.001)
    # Create a genuine A-key DCP checkpoint with the established backend.
    io.save(Path(root)/"A",a,ao,0,{"A":True})
    for variant in ("action_only","condition_align"):
        core,inputs,features=build()
        b=wrapped(core)
        loaded=load_model_only(b,Path(root)/"A",groups[0],sharded=True)
        assert loaded["new_auxiliary_entries"]==["forecast_head.bias","forecast_head.weight"]
        opt=torch.optim.AdamW(b.parameters(),lr=.001)
        prefix=make_prefix(inputs,features,1)
        noisy,tau,target=draw(torch.ones(2,16,7)*(rank+1),42+rank)
        teacher=torch.randn(2,3,4);mask=torch.ones(2,16,dtype=torch.bool)
        def update(model,optimizer):
            optimizer.zero_grad()
            out=model(prefix=prefix,noisy_actions=noisy,tau=tau)
            loss,_=objective(out,target,mask,teacher,variant)
            loss.backward();model.clip_grad_norm_(1.);optimizer.step()
        update(b,opt)
        recipe={"B":variant}
        path=Path(root)/variant
        io.save(path,b,opt,1,recipe)
        other,_,_=build()
        restored=wrapped(other)
        new_opt=torch.optim.AdamW(restored.parameters(),lr=.001)
        assert io.load(path,restored,new_opt,recipe)==1
        update(b,opt);update(restored,new_opt)
        for p,q in zip(b.parameters(),restored.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
        dist.barrier()
    dist.destroy_process_group()


def test_cpu4rank_A_to_B_warmstart_and_B_optimizer_restart(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    ctx=multiprocessing.get_context("spawn")
    workers=[ctx.Process(target=_worker,args=(rank,str(tmp_path))) for rank in range(4)]
    for p in workers:p.start()
    for p in workers:
        p.join(180)
        if p.is_alive():p.terminate();p.join()
        assert p.exitcode==0
