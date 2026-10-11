"""Pure CPU graph tests before any real-image operation."""

import numpy as np
import pytest
import torch
from torch import nn

from whenami_timeline.model import Timeline,TimelineConfig
from whenami_timeline.data import choose_times
from whenami_timeline.objectives import timeline_objective


def fixture(mode="revision", opened=True):
    torch.manual_seed(24)
    c=TimelineConfig(feature_dim=12,cognition_dim=20,tokens=4,condition_dim=3,hidden=16,heads=4)
    model=Timeline(c,mode)
    if opened:
        with torch.no_grad():model.readout[-1].weight.normal_(std=.01)
    x={"h":torch.randn(2,3,4,12),"z":torch.randn(2,3,1,20),"base_condition":torch.randn(2,3,4,3),
       "times":torch.tensor([[0,3,5],[4,6,10]]),"executed_actions":torch.randn(2,3,4,7),
       "executed_mask":torch.zeros(2,3,4,dtype=torch.bool)}
    for b in range(2):
        for t in (1,2):x["executed_mask"][b,t,:int(x["times"][b,t]-x["times"][b,t-1])]=True
    labels={"actions":torch.randn(2,3,16,7),"action_mask":torch.ones(2,3,16,dtype=torch.bool)}
    return model,x,labels


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.map=nn.Linear(3,7,bias=False)
        self.requires_grad_(False)
    def forward(self,x,tau,z,per_token):
        return .1*x+self.map(per_token.mean(1))[:,None]


@pytest.mark.parametrize("mode",["independent","forward","direct_history","revision"])
def test_future_anchors_cannot_change_earlier_actions(mode):
    m,x,_=fixture(mode)
    altered={k:v.clone() for k,v in x.items()}
    for k in ("h","z","base_condition","executed_actions"):altered[k][:,2]+=100
    a,b=m(**x),m(**altered)
    torch.testing.assert_close(a["conditions"][:,:2],b["conditions"][:,:2],rtol=0,atol=0)
    for i in (0,1):
        torch.testing.assert_close(a["belief_versions"][i],b["belief_versions"][i],rtol=0,atol=0)


def test_same_prefix_and_clock_shift():
    m,x,_=fixture()
    full=m(**x)
    sliced={k:v[:,:2] for k,v in x.items()}
    early=m(**sliced)
    torch.testing.assert_close(full["conditions"][:,:2],early["conditions"],rtol=0,atol=0)
    shifted=x|{"times":x["times"]+10000}
    torch.testing.assert_close(m(**shifted)["conditions"],full["conditions"],rtol=0,atol=0)


def test_zero_residual_preserves_exact_base():
    m,x,_=fixture(opened=False)
    torch.testing.assert_close(m(**x)["conditions"],x["base_condition"],rtol=0,atol=0)


def test_action_only_gradient_reaches_transition_backward_gate_bridge():
    m,x,y=fixture()
    reader=Reader()
    loss,metrics=timeline_objective(m(**x),x,y,reader,seed=55,draws=2,transition_weight=0)
    loss.backward()
    for module in (m.h_input,m.z_input,m.transition,m.backward_attention,m.backward_update,
                   m.relevance_gate,m.bank_attention,m.readout):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads)>0
    assert all(p.grad is None for p in reader.parameters())
    assert metrics["per_anchor_action_fm"].shape==(2,3)


def test_early_condition_cannot_read_later_executed_target_actions():
    m,x,_=fixture()
    x["executed_actions"].requires_grad_(True)
    m(**x)["conditions"][:,0].sum().backward()
    grad=x["executed_actions"].grad
    assert grad is None or not grad.any()


def test_padding_ignored_and_invalid_interval_rejected():
    m,x,_=fixture()
    changed=x["executed_actions"].clone();changed[~x["executed_mask"]]=float("nan")
    torch.testing.assert_close(m(**(x|{"executed_actions":changed}))["conditions"],m(**x)["conditions"],rtol=0,atol=0)
    with pytest.raises(ValueError):
        m(**(x|{"times":torch.tensor([[0,4,5],[4,6,10]])}))
    with pytest.raises(TypeError):
        m(**x,future_condition=torch.randn(2,4,3))


def test_all_action_anchors_and_padding_masks():
    m,x,y=fixture()
    y["action_mask"][:,:,4:]=False
    out=m(**x);reader=Reader()
    total,metrics=timeline_objective(out,x,y,reader,77,draws=2,transition_weight=0)
    torch.testing.assert_close(total,metrics["per_anchor_action_fm"].mean())
    # Check each anchor receives independent supervision, not only last node.
    out["conditions"].retain_grad()
    total.backward()
    assert (out["conditions"].grad.square().sum((2,3))>0).all()


def test_control_capacity_and_init_identical():
    a,_,_=fixture("revision");b,_,_=fixture("direct_history")
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],rtol=0,atol=0)


def test_real_distinct_anchors_even_short_episodes():
    for length in (4,5,9,17,300):
        for seed in range(20):
            t=choose_times(length,np.random.default_rng(seed))
            assert len(t)==3 and (np.diff(t)>0).all() and t.min()>=0 and t.max()<length-1
    with pytest.raises(ValueError):choose_times(3,np.random.default_rng(0))


def _checkpoint_worker(rank,rendezvous,root):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from pathlib import Path
    from whenami_timeline import checkpoint
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=f"file://{rendezvous}",rank=rank,world_size=2)
    model,inputs,labels=fixture()
    # Match real runner: only the small timeline in DDP, frozen shared reader.
    model=DDP(model,find_unused_parameters=True)
    reader=Reader()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.)
    recipe={"tiny_real_timeline_contract":True}
    def update(m,opt,seed):
        opt.zero_grad()
        loss,_=timeline_objective(m(**inputs),inputs,labels,reader,seed+rank,draws=2)
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    update(model,optimizer,1)
    torch.manual_seed(55+rank)
    path=Path(root)/"step1"
    checkpoint.save(path,model,optimizer,1,recipe)
    expected=torch.rand(4)
    restored,_,_=fixture()
    restored=DDP(restored,find_unused_parameters=True)
    opt2=torch.optim.AdamW(restored.parameters(),lr=1e-4,weight_decay=0.)
    assert checkpoint.load(path,restored,opt2,recipe)==1
    torch.testing.assert_close(torch.rand(4),expected,rtol=0,atol=0)
    update(model,optimizer,2);update(restored,opt2,2)
    for lhs,rhs in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(lhs,rhs,rtol=0,atol=0)
    dist.destroy_process_group()


def test_actual_ddp_save_resume_next_update(tmp_path):
    import multiprocessing as mp
    ctx=mp.get_context("spawn")
    workers=[ctx.Process(target=_checkpoint_worker,args=(rank,str(tmp_path/"rendezvous"),str(tmp_path)))
             for rank in range(2)]
    for worker in workers:worker.start()
    for worker in workers:
        worker.join(120)
        if worker.is_alive():worker.terminate();worker.join()
        assert worker.exitcode==0
