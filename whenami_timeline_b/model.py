"""Causal current-decision student, same A parameter names for strict model warm-start."""

import torch
from torch import nn
import torch.nn.functional as F

from whenami_timeline_joint.model import JointTimeline
from whenami_timeline_transfer.contracts import Prefix,select_anchor
from stage3.behavior import flow_draw


class Student(JointTimeline):
    def __init__(self,vlm,local_encoder,reader,mode="timeline",config=None):
        kwargs={} if config is None else {"config":config}
        super().__init__(vlm,local_encoder,reader,mode,**kwargs)
        self.forecast_head=nn.Linear(self.timeline.config.condition,self.timeline.config.condition)
        nn.init.eye_(self.forecast_head.weight);nn.init.zeros_(self.forecast_head.bias)

    def forward(self,prefix=None,noisy_actions=None,tau=None,condition_only=False,conditioning=None):
        if conditioning is not None:
            if prefix is not None or condition_only:raise ValueError("decoder-only request")
            return {"velocity":self.reader(noisy_actions,tau,conditioning["reader_z"],
                                           per_token=conditioning["conditions"])}
        if not isinstance(prefix,Prefix):raise TypeError("only a physically sliced Prefix is accepted")
        b,k=prefix.times.shape
        encoded=super().forward(**prefix.kwargs(),condition_only=True)
        current=select_anchor(encoded,b,k,k-1)
        result={**current,"forecast":self.forecast_head(current["conditions"])}
        if condition_only:return result
        if noisy_actions is None or tau is None or len(noisy_actions)%b:
            raise ValueError("noisy action states must repeat current observations")
        draws=len(noisy_actions)//b
        result["velocity"]=self.reader(
            noisy_actions,tau,current["reader_z"].repeat_interleave(draws,0),
            per_token=current["conditions"].repeat_interleave(draws,0))
        return result


def build_student(seed=42):
    from whenami_timeline_joint.model import load_joint
    base,visual,transform,tokenizer,prompt=load_joint("timeline",seed)
    student=Student(base.vlm,base.local_encoder,base.reader,"timeline")
    student.timeline.load_state_dict(base.timeline.state_dict(),strict=True)
    return student,visual,transform,tokenizer,prompt


def draw(actions,seed,draws=2):
    if actions.ndim!=3 or actions.shape[1:]!=(16,7):raise ValueError("current action labels only")
    return flow_draw(actions.detach().repeat_interleave(draws,0),seed)


def objective(output,target,valid,teacher_condition,variant,weight=.1,draws=2):
    if variant not in ("action_only","condition_align") or weight<0:raise ValueError("loss configuration")
    if valid.ndim!=2 or valid.shape[-1]!=16 or not valid.any(-1).all():raise ValueError("valid actions mask")
    mask=valid.detach().repeat_interleave(draws,0)
    error=(output["velocity"].float()-target.detach().float()).square().mean(-1)
    action=((error*mask).sum(-1)/mask.sum(-1)).mean()
    cosine=action.new_zeros(())
    # Diagnostic target distance in BOTH arms; teacher always detached, never
    # used by policy forward. Only B1 retains a gradient from it.
    if teacher_condition is not None:
        if teacher_condition.shape!=output["forecast"].shape:raise ValueError("condition shape")
        cosine=(1-F.cosine_similarity(output["forecast"].float(),teacher_condition.detach().float(),
                                      dim=-1,eps=1e-6)).mean()
    loss=action+weight*cosine if variant=="condition_align" else action
    return loss,{"action_fm":action.detach(),"condition_cosine":cosine.detach(),
                 "weighted_alignment":(weight*cosine.detach() if variant=="condition_align" else action.new_zeros(())),
                 "total":loss.detach()}
