"""Frozen endpoint action readers with differentiable condition inputs."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from stage3.common import check_endpoint, reference_imports


def flow_draw(actions, seed):
    """Exact WoG distribution: epsilon ~ N(0,I), tau ~ Beta(1.5,1).

    Beta(a,1) inverse CDF is U**(1/a). Explicit generators keep forward draws
    identical in forward-only and bidirectional controls.
    """
    generator = torch.Generator(device=actions.device).manual_seed(int(seed))
    epsilon = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
    tau = torch.rand(len(actions), generator=generator, device=actions.device, dtype=actions.dtype).pow(2 / 3)
    tau = tau * (1 - 2e-3) + 1e-3
    x = (1 - tau[:, None, None]) * epsilon + tau[:, None, None] * actions
    return x, tau, actions - epsilon


class FrozenBehavior(nn.Module):
    def __init__(self, net, checkpoint_blocks=False):
        super().__init__()
        self.net = net.requires_grad_(False).eval()
        self.checkpoint_blocks = checkpoint_blocks
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()  # No label dropout in the fixed evaluator.
        return self

    def velocity(self, x, tau, cognition, condition):
        if self.checkpoint_blocks and torch.is_grad_enabled() and condition.requires_grad:
            return activation_checkpoint(
                lambda c: self.net(x, tau, cognition.detach(), per_token=c),
                condition, use_reentrant=False,
            )
        return self.net(x, tau, cognition.detach(), per_token=condition)

    def loss(self, actions, cognition, condition, seed):
        x, tau, target_velocity = flow_draw(actions.detach(), seed)
        # DO NOT wrap in no_grad: the net is frozen, but dL/dcondition is needed.
        prediction = self.velocity(x, tau, cognition, condition)
        return (prediction.float() - target_velocity.float()).square().mean()

    @torch.no_grad()
    def sample(self, cognition, condition, seed, num_steps=10):
        if num_steps < 2:
            raise ValueError("num_steps must be >=2")
        generator = torch.Generator(device=cognition.device).manual_seed(int(seed))
        x = torch.randn(
            (len(cognition), self.net.future_action_window_size + 1, self.net.in_channels),
            generator=generator, device=cognition.device, dtype=cognition.dtype,
        )
        ts = torch.linspace(0, 1, num_steps, device=x.device, dtype=x.dtype)
        for i in range(1, num_steps):
            tau = ts[i - 1].expand(len(cognition))
            x = x + self.net(x, tau, cognition, per_token=condition) * (ts[i] - ts[i - 1])
        return x


def load_behavior(record, device, checkpoint_blocks=False):
    check_endpoint(record)
    reference_imports()
    from action_model.action_model import ActionModel
    ck = torch.load(record["path"], map_location="cpu", mmap=True, weights_only=True)
    if not any("per_attn" in k for k in ck["model"]["action_model"]):
        raise ValueError("endpoint has no condition-consuming action reader")
    action = ActionModel(
        token_size=4096, model_type="DiT-L", in_channels=7, Ta=16, train_venc=True,
    )
    action.load_state_dict(ck["model"]["action_model"], strict=True)
    return FrozenBehavior(action.net.to(device), checkpoint_blocks)
