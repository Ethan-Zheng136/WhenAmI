"""Shared real DiT at every causal anchor, with identical FM draws across modes."""

import torch
from stage3.behavior import flow_draw


def timeline_objective(timeline_output, inputs, labels, reader, seed, draws=2,
                       transition_weight=.1, condition_scale=None):
    actions, mask = labels["actions"],labels["action_mask"]
    b,k,t,a = actions.shape
    if (t,a)!=(16,7) or mask.shape!=actions.shape[:-1] or mask.dtype!=torch.bool:
        raise ValueError("invalid action targets")
    if not (mask.sum(-1)>0).all():
        raise ValueError("empty anchor action target")
    # B,K then repeat draw axis: condition/cognition/noisy labels use same order.
    targets=actions.reshape(b*k,t,a).repeat_interleave(draws,0)
    x,tau,target_velocity=flow_draw(targets,seed)
    condition=timeline_output["conditions"].reshape(b*k,*timeline_output["conditions"].shape[2:]).repeat_interleave(draws,0)
    z=inputs["z"].reshape(b*k,*inputs["z"].shape[2:]).repeat_interleave(draws,0)
    velocity=reader(x,tau,z,per_token=condition)
    valid=mask.reshape(b*k,t).repeat_interleave(draws,0).to(velocity.dtype)
    per=(velocity.float()-target_velocity.float()).square().mean(-1)
    per_draw=(per*valid).sum(-1)/valid.sum(-1)
    per_anchor=per_draw.reshape(b,k,draws).mean(-1)
    action=per_anchor.mean()
    prediction=action.new_zeros(())
    if k>1 and transition_weight:
        scale=1. if condition_scale is None else condition_scale
        prediction=((timeline_output["predicted_conditions"][:,1:].float()-
                     inputs["base_condition"][:,1:].detach().float())/scale).square().mean()
    total=action+transition_weight*prediction
    return total,{"action_fm":action.detach(),"transition_mse":prediction.detach(),
                  "per_anchor_action_fm":per_anchor.detach(),"total":total.detach()}
