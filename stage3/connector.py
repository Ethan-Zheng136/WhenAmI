"""Wide shared trunk, directional adapters, fixed target coordinates.

No actions, target conditions, or target-side features enter a directional
prediction. These are same-centre condition predictions, NOT inverse physics,
not per-timestep states, and not a claim of an established action timeline.
"""

from dataclasses import dataclass, asdict

import torch
from torch import nn


@dataclass(frozen=True)
class ConnectorConfig:
    feature_dim: int = 1024
    cognition_dim: int = 4096
    tokens: int = 16
    condition_dim: int = 32
    hidden: int = 256
    layers: int = 3
    heads: int = 8
    ffn_mult: int = 4
    scale_floor: float = 0.02

    def __post_init__(self):
        if min(self.feature_dim, self.cognition_dim, self.tokens, self.condition_dim,
               self.hidden, self.layers, self.heads, self.ffn_mult) < 1:
            raise ValueError("all model dimensions must be positive")
        if self.hidden % self.heads or self.scale_floor <= 0:
            raise ValueError("hidden must be divisible by heads; scale_floor must be positive")

    def as_dict(self):
        return asdict(self)


class SharedConnector(nn.Module):
    def __init__(self, config=ConnectorConfig(), target_statistics=None):
        super().__init__()
        self.config = config
        self.input_adapters = nn.ModuleDict({
            side: nn.Sequential(nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim, config.hidden))
            for side in ("past", "future")
        })
        self.task_adapters = nn.ModuleDict({
            side: nn.Sequential(nn.LayerNorm(config.cognition_dim), nn.Linear(config.cognition_dim, config.hidden))
            for side in ("past", "future")
        })
        self.direction = nn.Parameter(torch.randn(2, 1, config.hidden) * 0.02)
        # Query identity, NOT absolute episode progress/time labels.
        self.query_position = nn.Parameter(torch.randn(1, config.tokens, config.hidden) * 0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.hidden, config.heads, config.hidden * config.ffn_mult,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(config.layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden)
        self.readouts = nn.ModuleDict({
            side: nn.Linear(config.hidden, config.condition_dim) for side in ("past", "future")
        })
        for head in self.readouts.values():
            nn.init.normal_(head.weight, std=1e-3)  # Nonzero: upstream gradients are not cut.
            nn.init.zeros_(head.bias)
        for side in ("past", "future"):
            mean = torch.zeros(config.tokens, config.condition_dim)
            std = torch.ones_like(mean)
            if target_statistics is not None:
                mean = torch.as_tensor(target_statistics[side]["mean"]).float()
                std = torch.as_tensor(target_statistics[side]["std"]).float()
            expected = (config.tokens, config.condition_dim)
            if mean.shape != expected or std.shape != expected:
                raise ValueError(f"target statistics must have shape {expected}")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
                raise ValueError("invalid target statistics")
            self.register_buffer(f"{side}_mean", mean)
            self.register_buffer(f"{side}_scale", std.clamp_min(config.scale_floor))

    def predict(self, source_features, source_cognition, direction):
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be forward or backward")
        source, target, index = (
            ("past", "future", 0) if direction == "forward" else ("future", "past", 1)
        )
        c = self.config
        if source_features.ndim != 3 or source_features.shape[1:] != (c.tokens, c.feature_dim):
            raise ValueError("source features must be [B, tokens, feature_dim]")
        if source_cognition.shape != (len(source_features), 1, c.cognition_dim):
            raise ValueError("source cognition must be [B,1,cognition_dim]")
        h = self.input_adapters[source](source_features) + self.query_position + self.direction[index]
        z = self.task_adapters[source](source_cognition)
        hidden = torch.cat([z, h], dim=1)
        for block in self.blocks:
            hidden = block(hidden)
        decoded = self.readouts[target](self.final_norm(hidden[:, 1:]))
        return getattr(self, f"{target}_mean") + getattr(self, f"{target}_scale") * decoded

    def forward(self, past_features, past_cognition, future_features=None, future_cognition=None):
        result = {"forward": self.predict(past_features, past_cognition, "forward")}
        if (future_features is None) != (future_cognition is None):
            raise ValueError("backward prediction needs both future inputs")
        if future_features is not None:
            result["backward"] = self.predict(future_features, future_cognition, "backward")
        return result

    def alignment_loss(self, prediction, target, target_side):
        if prediction.shape != target.shape:
            raise ValueError("condition target shape mismatch")
        scale = getattr(self, f"{target_side}_scale")
        return ((prediction.float() - target.detach().float()) / scale).square().mean()

    def freeze_backward_only_adapters(self):
        # Forward-only control has the exact same shared architecture, but
        # unused directional adapters are excluded from the optimizer/DDP.
        for module in (self.input_adapters["future"], self.task_adapters["future"], self.readouts["past"]):
            module.requires_grad_(False)
