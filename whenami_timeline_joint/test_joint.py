import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from whenami_timeline_joint.model import JointTimeline,AddressedConditioner,MODES
from whenami_timeline_a.model import Config
from whenami_timeline_a.test_model import Local,Reader,fixture as previous_fixture
from whenami_timeline_a.objectives import noisy_labels,velocity_loss

torch.set_num_threads(1)
CFG=Config(feature=12,cognition=20,queries=3,hidden=16,heads=4,layers=2,condition=4)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone=nn.Linear(12,20)
        self.projector=nn.Linear(20,20)
        self.llm_backbone=nn.Embedding(64,20)
        self.vision_backbone_requires_grad=True
    def forward(self,input_ids,attention_mask,pixel_values,**kwargs):
        img=self.projector(self.vision_backbone(pixel_values["rgb"].flatten(1)))[:,None]
        x=self.llm_backbone(input_ids)+img
        return SimpleNamespace(hidden_states=[torch.cat((x[:,:1],x.new_zeros(len(x),2,20),x[:,1:]),1)])


def fixture(mode="timeline"):
    torch.manual_seed(42)
    m=JointTimeline(TinyVLM(),Local(),Reader(),mode,CFG)
    with torch.no_grad():m.timeline.output[-1].weight.normal_(std=.03)
    x={"pixels":{"rgb":torch.randn(2,3,3,2,2)},"input_ids":torch.tensor([[1,2,3,0]]*6),
       "attention_mask":torch.tensor([[True,True,True,False]]*6),
       "patches":torch.randn(2,3,8,12),"vae":torch.randn(2,3,4,5),
       "times":torch.tensor([[0,4,11]]*2),"visible":torch.tensor([[True,False,True]]*2)}
    return m,x


@pytest.mark.parametrize("mode",MODES)
def test_joint_action_gradient_reaches_vlm_and_temporal(mode):
    m,x=fixture(mode);labels=torch.ones(2,3,16,7,requires_grad=True)
    noisy,tau,target=noisy_labels(labels,42)
    out=m(**x,noisy_actions=noisy,tau=tau)
    loss,_=velocity_loss(out,target,torch.ones(2,3,16,dtype=torch.bool))
    loss.backward();assert labels.grad is None
    for module in (m.vlm.vision_backbone,m.vlm.projector,m.vlm.llm_backbone,
                   m.local_encoder,m.timeline.input,m.timeline.blocks,m.reader):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0


@pytest.mark.parametrize("mode",MODES)
def test_hidden_raw_pixels_cannot_leak(mode):
    m,x=fixture(mode);m.eval()
    y=copy.deepcopy(x);y["pixels"]["rgb"][:,1]=torch.nan;y["patches"][:,1]=torch.nan;y["vae"][:,1]=torch.nan
    a=m(**x,condition_only=True);b=m(**y,condition_only=True)
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


@pytest.mark.parametrize("mode",MODES)
def test_future_cutoff_prevents_indirect_z_leakage(mode):
    m,x=fixture(mode);m.eval();x["visible"][:]=True;x["cutoffs"]=x["times"].clone()
    y=copy.deepcopy(x);y["pixels"]["rgb"][:,2]+=1000;y["patches"][:,2]+=1000
    a=m(**x,condition_only=True);b=m(**y,condition_only=True)
    for k in a:
        left=a[k].reshape(2,3,*a[k].shape[1:])[:,:2]
        right=b[k].reshape(2,3,*b[k].shape[1:])[:,:2]
        torch.testing.assert_close(left,right,rtol=0,atol=0)


def test_both_arms_identical_parameters_and_zero_residual_start():
    a,x=fixture("timeline");b,_=fixture("pooled_query")
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    for m in (a,b):
        nn.init.zeros_(m.timeline.output[-1].weight);nn.init.zeros_(m.timeline.output[-1].bias)
    for k,v in a(**x,condition_only=True).items():
        torch.testing.assert_close(v,b(**x,condition_only=True)[k],rtol=0,atol=0)


def test_pooled_targets_have_independent_queries_even_when_hidden():
    torch.manual_seed(42);c=AddressedConditioner(CFG,"pooled_query")
    with torch.no_grad():c.output[-1].weight.normal_(std=.1)
    h=torch.randn(1,3,3,12);z=torch.randn(1,3,1,20)
    v=torch.tensor([[True,False,False]]);t=torch.tensor([[0,4,11]])
    out=c(h,z,t,v)["state"]
    assert not torch.allclose(out[:,1],out[:,2])


def test_decoder_only_matches_full_velocity():
    m,x=fixture();m.eval()
    noisy,tau,_=noisy_labels(torch.ones(2,3,16,7),42,draws=1)
    a=m(**x,noisy_actions=noisy,tau=tau)
    e=m(**x,condition_only=True)
    b=m(noisy_actions=noisy,tau=tau,conditioning=e)
    torch.testing.assert_close(a["velocity"],b["velocity"],rtol=0,atol=0)
