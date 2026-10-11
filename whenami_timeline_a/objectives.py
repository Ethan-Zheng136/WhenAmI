"""Actual per-anchor action FM; action labels cannot enter the condition encoder."""

import torch
from stage3.behavior import flow_draw


def action_loss(model, encoded, actions, action_mask, seed, draws=2):
    b,k,t,d=actions.shape
    if (t,d)!=(16,7) or action_mask.shape!=actions.shape[:-1] or action_mask.dtype!=torch.bool:
        raise ValueError("expected separate16x7 action labels and valid-action mask")
    if not action_mask.any(-1).all():
        raise ValueError("empty action target")
    target=actions.detach().flatten(0,1).repeat_interleave(draws,0)
    x,tau,velocity=flow_draw(target,seed)
    condition=encoded["conditions"].flatten(0,1).repeat_interleave(draws,0)
    z=encoded["reader_z"].flatten(0,1).repeat_interleave(draws,0)
    pred=model.reader(x,tau,z,per_token=condition)
    per=(pred.float()-velocity.float()).square().mean(-1)
    valid=action_mask.flatten(0,1).repeat_interleave(draws,0).to(per.dtype)
    anchor=((per*valid).sum(-1)/valid.sum(-1)).reshape(b,k,draws).mean(-1)
    return anchor.mean(), {"action_fm":anchor.mean().detach(), "per_anchor":anchor.detach()}


def noisy_labels(actions,seed,draws=2):
    return flow_draw(actions.detach().flatten(0,1).repeat_interleave(draws,0),seed)


def velocity_loss(output,velocity,action_mask,draws=2):
    b,k,t=action_mask.shape
    if t!=16 or not action_mask.any(-1).all():raise ValueError("invalid action mask")
    error=(output["velocity"].float()-velocity.detach().float()).square().mean(-1)
    mask=action_mask.flatten(0,1).repeat_interleave(draws,0).to(error.dtype)
    anchor=((error*mask).sum(-1)/mask.sum(-1)).reshape(b,k,draws).mean(-1)
    return anchor.mean(),{"action_fm":anchor.mean().detach(),"per_anchor":anchor.detach()}


@torch.no_grad()
def sample_actions(model, encoded, noise, intervals=9):
    if noise.ndim!=4 or noise.shape[-2:]!=(16,7) or noise.dtype!=torch.float32:
        raise ValueError("FP32 noise[B,K,16,7] required")
    b,k=noise.shape[:2];x=noise.flatten(0,1).clone()
    z=encoded["reader_z"].flatten(0,1)
    condition=encoded["conditions"].flatten(0,1)
    ts=torch.linspace(0,1,intervals+1,device=x.device,dtype=torch.float32)
    for i in range(intervals):
        v=model.reader(x,ts[i].expand(len(x)),z,per_token=condition)
        x=x+v.float()*(ts[i+1]-ts[i])
    if not torch.isfinite(x).all():raise FloatingPointError("nonfinite sampled action")
    return x.reshape(b,k,16,7)
