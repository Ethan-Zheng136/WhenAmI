"""R forecasts a condition; K corrects the causal Past condition before action.

Online inputs are strictly Past H/z/C. An explicitly separate oracle function
accepts true Future C for TRAINING/diagnostics only. No action enters R or K.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from stage3.connector import ConnectorConfig, SharedConnector


@dataclass(frozen=True)
class RefinementConfig:
    feature_dim: int = 1024
    cognition_dim: int = 4096
    tokens: int = 16
    condition_dim: int = 32
    r_hidden: int = 256
    r_layers: int = 3
    r_heads: int = 8
    k_hidden: int = 128
    k_layers: int = 2
    k_heads: int = 4
    scale_floor: float = .02

    def __post_init__(self):
        dimensions = (self.feature_dim, self.cognition_dim, self.tokens, self.condition_dim,
                      self.r_hidden, self.r_layers, self.r_heads, self.k_hidden, self.k_layers, self.k_heads)
        if min(dimensions) < 1 or self.r_hidden % self.r_heads or self.k_hidden % self.k_heads:
            raise ValueError("positive dimensions and divisible attention head sizes required")
        if self.scale_floor <= 0:
            raise ValueError("positive scale floor required")

    def as_dict(self):
        return asdict(self)


class RetrospectiveCorrector(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.past_input = nn.Sequential(nn.LayerNorm(c.feature_dim), nn.Linear(c.feature_dim, c.k_hidden))
        self.task_input = nn.Sequential(nn.LayerNorm(c.cognition_dim), nn.Linear(c.cognition_dim, c.k_hidden))
        self.future_input = nn.Linear(c.condition_dim, c.k_hidden)
        # Query identities, NOT time indices.
        self.past_position = nn.Parameter(torch.randn(1, c.tokens, c.k_hidden) * .02)
        self.future_position = nn.Parameter(torch.randn(1, c.tokens, c.k_hidden) * .02)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                c.k_hidden, c.k_heads, dim_feedforward=4 * c.k_hidden,
                dropout=0., activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(c.k_layers)
        ])
        self.norm = nn.LayerNorm(c.k_hidden)
        self.output = nn.Linear(c.k_hidden, c.condition_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, past_h, past_z, standardized_future):
        query = torch.cat([
            self.task_input(past_z), self.past_input(past_h) + self.past_position,
        ], dim=1)
        memory = self.future_input(standardized_future) + self.future_position
        for block in self.blocks:
            query = block(query, memory)
        return self.output(self.norm(query[:, 1:]))


class PastRefinement(nn.Module):
    def __init__(self, config=RefinementConfig(), target_statistics=None, mode="teacher"):
        super().__init__()
        if mode not in ("adapter", "predict", "teacher"):
            raise ValueError("mode must be adapter, predict, or teacher")
        self.config, self.mode = config, mode
        self.r = SharedConnector(ConnectorConfig(
            feature_dim=config.feature_dim, cognition_dim=config.cognition_dim,
            tokens=config.tokens, condition_dim=config.condition_dim,
            hidden=config.r_hidden, layers=config.r_layers, heads=config.r_heads,
            scale_floor=config.scale_floor,
        ), target_statistics)
        self.r.freeze_backward_only_adapters()
        self.k = RetrospectiveCorrector(config)
        if mode == "adapter":
            self.r.requires_grad_(False)

    def correct(self, past_h, past_z, past_c, future_c):
        if past_c.shape != future_c.shape:
            raise ValueError("Past and Future condition shapes must match")
        standardized = (future_c - self.r.future_mean) / self.r.future_scale
        delta = self.r.past_scale * self.k(past_h.detach(), past_z.detach(), standardized)
        return past_c.detach() + delta

    def forward(self, past_h, past_z, past_c):
        """Causal API: no future, actions, noisy actions, or target z accepted."""
        future = (
            self.r.future_mean[None].expand(len(past_h), -1, -1)
            if self.mode == "adapter"
            else self.r.predict(past_h.detach(), past_z.detach(), "forward")
        )
        return {
            "future_prediction": future,
            "online": self.correct(past_h, past_z, past_c, future),
        }

    def oracle(self, past_h, past_z, past_c, true_future_c):
        """Explicit privilege boundary. This output must never be deployed."""
        return self.correct(past_h, past_z, past_c, true_future_c.detach())


def online_inputs(batch):
    return [batch[f"past_{key}"].detach() for key in ("h", "z", "c")]


def differentiable_rollout(reader, cognition, condition, seed, steps=4):
    """Pure-noise Euler rollout; no GT interpolation, no intermediate detach.

steps counts integration intervals. Existing evaluation uses 10 grid points,
i.e. 9 intervals. This training auxiliary does not change evaluation sampling.
"""
    if steps < 1:
        raise ValueError("at least one integration interval required")
    generator = torch.Generator(device=cognition.device).manual_seed(int(seed))
    x = torch.randn(
        (len(cognition), reader.net.future_action_window_size + 1, reader.net.in_channels),
        generator=generator, device=cognition.device, dtype=cognition.dtype,
    )
    for i in range(steps):
        tau = x.new_full((len(x),), i / steps)
        x = x + reader.velocity(x, tau, cognition.detach(), condition) / steps
    return x


def repeated_fm_loss(reader, actions, cognition, condition, seed, draws=1):
    """Independent time/noise draws, averaged: NOT sequential Euler integration."""
    if draws < 1:
        raise ValueError("positive draw count required")
    return torch.stack([
        reader.loss(actions.detach(), cognition.detach(), condition, seed + j * 104729)
        for j in range(draws)
    ]).mean()


def refinement_loss(model, reader, batch, seed, alignment_weight=.1,
                    fm_draws=1, rollout_steps=0, rollout_weight=0.):
    if alignment_weight < 0 or rollout_weight < 0 or rollout_steps < 0:
        raise ValueError("nonnegative weights/steps required")
    if (rollout_weight > 0) != (rollout_steps > 0):
        raise ValueError("rollout weight and steps must both be zero or both positive")
    output = model(*online_inputs(batch))
    fm = repeated_fm_loss(reader, batch["past_actions"], batch["past_z"], output["online"], seed, fm_draws)
    values = {"online_action_fm": fm}
    action = fm
    if model.mode == "teacher":
        oracle = model.oracle(*online_inputs(batch), batch["future_c"])
        oracle_fm = repeated_fm_loss(reader, batch["past_actions"], batch["past_z"], oracle, seed, fm_draws)
        values["oracle_action_fm"] = oracle_fm
        # Keep overall action-gradient scale from trivially doubling.
        action = .5 * (fm + oracle_fm)
    if model.mode != "adapter":
        align = model.r.alignment_loss(output["future_prediction"], batch["future_c"], "future")
        values["future_alignment"] = align
        action = action + alignment_weight * align
    if rollout_steps:
        sampled = differentiable_rollout(
            reader, batch["past_z"], output["online"], seed + 2000003, rollout_steps,
        )
        mse = (sampled.float() - batch["past_actions"].detach().float()).square().mean()
        values["online_rollout_mse"] = mse
        action = action + rollout_weight * mse
    values["total"] = action
    values["delta_rms_standardized"] = (
        (output["online"] - batch["past_c"]) / model.r.past_scale
    ).square().mean().sqrt()
    return action, values
