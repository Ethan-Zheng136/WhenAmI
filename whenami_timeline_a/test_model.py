"""Small CPU models test contracts, not robot-policy quality."""

import copy
import json
import multiprocessing
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from whenami_timeline_a.model import Config,TimelinePolicy,MODES
from whenami_timeline_a.objectives import action_loss,sample_actions,noisy_labels,velocity_loss
from whenami_timeline_a.data import choose_times,visibility
from whenami_timeline_a import checkpoint

torch.set_num_threads(1)
CFG=Config(feature=12,cognition=20,queries=3,hidden=16,heads=4,layers=2,condition=4)


class Local(nn.Module):
    def __init__(self):
        super().__init__()
        self.p=nn.Linear(12,12);self.v=nn.Linear(5,12)
        self.query=nn.Parameter(torch.randn(3,12))
        self.v_proj=nn.Linear(12,4)
    def encode_queries(self,p,v):
        return self.p(p.mean(1))[:,None]+self.v(v.mean(1))[:,None]+self.query


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.z=nn.Linear(20,7);self.c=nn.Linear(4,7);self.x=nn.Linear(7,7)
    def forward(self,x,t,z,per_token):
        return self.x(x)+self.z(z)+self.c(per_token.mean(1))[:,None]


def fixture(mode="timeline",open_residual=True,k=3):
    torch.manual_seed(42)
    m=TimelinePolicy(Local(),Reader(),CFG,mode)
    if open_residual:
        with torch.no_grad():m.timeline.output[-1].weight.normal_(std=.03)
    b=2
    times=torch.arange(k)[None].expand(b,-1).clone()*4
    x={"patches":torch.randn(b,k,8,12),"vae":torch.randn(b,k,4,5),
       "z":torch.randn(b,k,1,20),"times":times,"visible":torch.ones(b,k,dtype=torch.bool)}
    return m,x


@pytest.mark.parametrize("mode",MODES)
def test_masked_node_all_channels_have_zero_influence(mode):
    m,x=fixture(mode);m.eval();x["visible"][:,1]=False
    y={k:v.clone() for k,v in x.items()}
    for name in ("patches","vae","z"):y[name][:,1]=torch.nan
    a=m(**x);b=m(**y)
    torch.testing.assert_close(a["conditions"],b["conditions"],rtol=0,atol=0)
    torch.testing.assert_close(a["reader_z"],b["reader_z"],rtol=0,atol=0)
    assert torch.isfinite(b["conditions"]).all()


@pytest.mark.parametrize("mode",MODES)
def test_causal_cutoff_cannot_leak_through_intermediate_nodes(mode):
    m,x=fixture(mode,k=5);m.eval();x["cutoffs"]=x["times"].clone()
    y={k:v.clone() for k,v in x.items()}
    for name in ("patches","vae","z"):y[name][:,2:]+=1000
    a,b=m(**x),m(**y)
    torch.testing.assert_close(a["conditions"][:,:2],b["conditions"][:,:2],rtol=0,atol=0)
    torch.testing.assert_close(a["reader_z"][:,:2],b["reader_z"][:,:2],rtol=0,atol=0)


def test_hidden_gradient_is_zero_but_visible_evidence_gradient_exists():
    m,x=fixture();x["visible"][:,1]=False
    for name in ("patches","vae","z"):x[name].requires_grad_(True)
    out=m(**x);out["conditions"][:,1].square().mean().backward()
    for name in ("patches","vae","z"):
        assert torch.count_nonzero(x[name].grad[:,1])==0
        assert x[name].grad[:,[0,2]].abs().sum()>0


def test_independent_hidden_prediction_does_not_use_other_nodes():
    m,x=fixture("independent");x["visible"][:,1]=False
    y={k:v.clone() for k,v in x.items()};y["patches"][:,[0,2]]+=100
    a,b=m(**x),m(**y)
    torch.testing.assert_close(a["conditions"][:,1],b["conditions"][:,1],rtol=0,atol=0)


def test_timeline_hidden_prediction_uses_other_nodes():
    m,x=fixture();x["visible"][:,1]=False
    y={k:v.clone() for k,v in x.items()};y["patches"][:,0]*=2
    assert not torch.allclose(m(**x)["conditions"][:,1],m(**y)["conditions"][:,1])


@pytest.mark.parametrize("mode",MODES)
def test_null_evidence_is_finite_and_has_no_image_gradient(mode):
    m,x=fixture(mode);x["visible"][:]=False
    a=m(**x);assert torch.isfinite(a["conditions"]).all()
    y={k:v.clone() for k,v in x.items()};y["patches"]*=100
    torch.testing.assert_close(a["conditions"],m(**y)["conditions"],rtol=0,atol=0)


@pytest.mark.parametrize("mode",MODES)
def test_time_translation_invariance(mode):
    m,x=fixture(mode);x["cutoffs"]=x["times"].clone()
    y={k:v.clone() for k,v in x.items()};y["times"]+=876;y["cutoffs"]+=876
    torch.testing.assert_close(m(**x)["conditions"],m(**y)["conditions"],rtol=0,atol=0)


def test_all_modes_parameter_initialization_matches():
    ms=[fixture(mode)[0] for mode in MODES]
    for m in ms[1:]:
        for k,v in ms[0].state_dict().items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)


def test_singleton_all_modes_equal():
    outputs=[fixture(mode,k=1) for mode in MODES]
    first=outputs[0][0](**outputs[0][1])
    for m,x in outputs[1:]:torch.testing.assert_close(first["conditions"],m(**x)["conditions"],rtol=0,atol=0)


def test_pooled_same_condition_at_all_targets_with_same_evidence():
    m,x=fixture("pooled")
    out=m(**x)["conditions"]
    for i in range(1,3):torch.testing.assert_close(out[:,0],out[:,i],rtol=0,atol=0)


@pytest.mark.parametrize("mode",MODES)
def test_zero_residual_base_identity(mode):
    m,x=fixture(mode,open_residual=False)
    out=m(**x)
    h=m.local_encoder.encode_queries(x["patches"].flatten(0,1),x["vae"].flatten(0,1)).reshape(2,3,3,12)
    if mode=="pooled":h=h.mean(1,keepdim=True).expand_as(h)
    torch.testing.assert_close(out["conditions"],m.local_encoder.v_proj(h),rtol=0,atol=0)


def test_action_only_loss_trains_every_relevant_module():
    m,x=fixture();x["visible"][:,1]=False
    labels=torch.randn(2,3,16,7,requires_grad=True)
    mask=torch.ones(2,3,16,dtype=torch.bool);mask[0,0,3:]=False
    loss,metrics=action_loss(m,m(**x),labels,mask,77)
    loss.backward();assert labels.grad is None
    for part in (m.local_encoder,m.reader,m.timeline.input,m.timeline.blocks,m.timeline.output):
        assert sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None)>0
    assert metrics["per_anchor"].shape==(2,3)
    with pytest.raises(TypeError):m(**x,actions=labels)


def test_action_mask_excludes_padded_targets():
    m,x=fixture();m.eval();encoded=m(**x)
    a=torch.randn(2,3,16,7);mask=torch.ones(2,3,16,dtype=torch.bool);mask[:,:,2:]=False
    b=a.clone();b[:,:,2:]=999
    # Reader fixture has no cross-time attention, so this isolates scoring mask.
    l1,_=action_loss(m,encoded,a,mask,5)
    l2,_=action_loss(m,encoded,b,mask,5)
    torch.testing.assert_close(l1,l2,rtol=0,atol=0)


def test_real_time_sampling_and_label_free_sampler():
    for n in (4,7,25,300):
        t=choose_times(n,np.random.default_rng(3),nodes=3)
        assert len(set(t))==3 and min(t)>=0 and max(t)<n-1
    m,x=fixture();m.eval()
    noise=torch.randn(2,3,16,7)
    a=sample_actions(m,m(**x),noise)
    assert a.shape==noise.shape and torch.isfinite(a).all()


def test_same_process_model_adam_rng_restore(tmp_path):
    m,x=fixture();o=torch.optim.AdamW(m.parameters(),lr=.001)
    for _ in range(2):
        o.zero_grad();l,_=action_loss(m,m(**x),torch.ones(2,3,16,7),torch.ones(2,3,16,dtype=torch.bool),9)
        l.backward();o.step()
    checkpoint.save(tmp_path/"cp",m,o,2,{"fixture":1})
    expected=[random.random(),float(np.random.rand()),float(torch.rand(()))]
    n,y=fixture();op=torch.optim.AdamW(n.parameters(),lr=.001)
    assert checkpoint.load(tmp_path/"cp",n,op,{"fixture":1})==2
    actual=[random.random(),float(np.random.rand()),float(torch.rand(()))]
    assert actual==expected
    for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],rtol=0,atol=0)
    for mod,opt in ((m,o),(n,op)):
        opt.zero_grad();l,_=action_loss(mod,mod(**x),torch.ones(2,3,16,7),torch.ones(2,3,16,dtype=torch.bool),9)
        l.backward();opt.step()
    for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],rtol=0,atol=0)


def _ddp_worker(rank,root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method="file://"+str(Path(root)/"rendezvous"),rank=rank,world_size=2)
    m,x=fixture();model=DDP(m,find_unused_parameters=True);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    actions=torch.ones(2,3,16,7)*(rank+1);mask=torch.ones(2,3,16,dtype=torch.bool)
    # Reader is inside wrapped root, so DDP sees the complete graph.
    noisy,tau,target=noisy_labels(actions,42+rank)
    opt.zero_grad();encoded=model(**x,noisy_actions=noisy,tau=tau);loss,_=velocity_loss(encoded,target,mask)
    loss.backward();opt.step()
    checkpoint.save(Path(root)/"ddp_cp",model,opt,1,{"ddp":1})
    n,_=fixture();restored=DDP(n,find_unused_parameters=True);other=torch.optim.AdamW(restored.parameters(),lr=.001)
    assert checkpoint.load(Path(root)/"ddp_cp",restored,other,{"ddp":1})==1
    for mod,o in ((model,opt),(restored,other)):
        o.zero_grad();v=mod(**x,noisy_actions=noisy,tau=tau);l,_=velocity_loss(v,target,mask)
        l.backward();o.step()
    for k,v in m.state_dict().items():torch.testing.assert_close(v,n.state_dict()[k],rtol=0,atol=0)
    dist.barrier();dist.destroy_process_group()


def test_cpu_two_rank_real_ddp_and_checkpoint(tmp_path,monkeypatch):
    # CPU only: do not initialize PPU contexts in spawned test workers.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    ctx=multiprocessing.get_context("spawn")
    jobs=[ctx.Process(target=_ddp_worker,args=(rank,str(tmp_path))) for rank in range(2)]
    for job in jobs:job.start()
    for job in jobs:
        job.join(120)
        if job.is_alive():job.terminate();job.join()
        assert job.exitcode==0
