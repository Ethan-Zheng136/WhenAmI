"""Decoupled teacher-condition supervision; current action path stays strictly causal."""

import torch
from torch import nn
import torch.nn.functional as F

from whenami_timeline_transfer.contracts import Prefix,select_anchor
from stage3.behavior import flow_draw


class CausalStudent(nn.Module):
    def __init__(self,base,condition_dim=32):
        super().__init__()
        self.base=base
        # Identity initialization: first auxiliary prediction in the student's
        # existing condition coordinates; not a new uncalibrated target space.
        self.future_predictor=nn.Linear(condition_dim,condition_dim)
        nn.init.eye_(self.future_predictor.weight);nn.init.zeros_(self.future_predictor.bias)

    def forward(self,prefix,noisy_actions=None,tau=None):
        if not isinstance(prefix,Prefix):
            raise TypeError("student accepts a validated prefix, not a full future-bearing batch")
        b,k=prefix.times.shape
        encoded=self.base(**prefix.kwargs(),condition_only=True)
        current=select_anchor(encoded,b,k,k-1)
        result={**current,"forecast":self.future_predictor(current["conditions"])}
        if (noisy_actions is None)!=(tau is None):raise ValueError("both noisy actions and tau required")
        if noisy_actions is not None:
            if len(noisy_actions)%b:raise ValueError("action draws must repeat current batch")
            draws=len(noisy_actions)//b
            cond={key:value.repeat_interleave(draws,0) for key,value in current.items()}
            result["velocity"]=self.base(noisy_actions=noisy_actions,tau=tau,conditioning=cond)["velocity"]
        return result


@torch.no_grad()
def teacher_target(teacher,inputs,features,current_index):
    b,k=inputs["times"].shape
    teacher.eval()
    encoded=teacher(**inputs,**features,visible=torch.ones_like(inputs["times"],dtype=torch.bool),
                    condition_only=True)
    return select_anchor(encoded,b,k,current_index)["conditions"].detach()


def action_draw(actions,seed,draws=2):
    if actions.ndim!=3 or actions.shape[1:]!=(16,7):raise ValueError("current action labels [B,16,7]")
    return flow_draw(actions.detach().repeat_interleave(draws,0),seed)


def objective(output,velocity_target,action_mask,teacher_condition=None,weight=.1,draws=2):
    if weight<0 or action_mask.ndim!=2 or action_mask.shape[-1]!=16:
        raise ValueError("loss configuration")
    mask=action_mask.detach().repeat_interleave(draws,0)
    if not mask.any(-1).all():raise ValueError("empty action target")
    error=(output["velocity"].float()-velocity_target.detach().float()).square().mean(-1)
    action=((error*mask).sum(-1)/mask.sum(-1)).mean()
    alignment=action.new_zeros(())
    if teacher_condition is not None and weight:
        if output["forecast"].shape!=teacher_condition.shape:raise ValueError("teacher/student condition shape")
        alignment=(1-F.cosine_similarity(output["forecast"].float(),
                                        teacher_condition.detach().float(),dim=-1,eps=1e-6)).mean()
    return action+weight*alignment,{"action_fm":action.detach(),"condition_alignment":alignment.detach()}
