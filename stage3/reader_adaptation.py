"""Isolated low-rank residual adapters in a frozen Past DiT.

Never modifies checkpoint files or reference classes. Each wrapped block runs
its original computation, followed by a zero-initialized bottleneck residual.
"""

import torch
from torch import nn

from stage3.behavior import flow_draw
from stage3.oracle_evidence import EvidenceProbe
from stage3.refinement import RefinementConfig


class ResidualReadoutAdapter(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        if rank < 1:
            raise ValueError("positive adapter rank required")
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(torch.nn.functional.gelu(self.down(self.norm(x))))


class AdaptedBlock(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.adapter = ResidualReadoutAdapter(base.norm1.normalized_shape[0], rank)

    def forward(self, x, per_token):
        return self.adapter(self.base(x, per_token))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self


def attach_adapters(net, rank=32, enabled=True):
    net.requires_grad_(False).eval()
    if any(isinstance(block, AdaptedBlock) for block in net.blocks):
        raise ValueError("adapters already installed")
    # Construct on the existing network device, before optimizer creation.
    device = next(net.parameters()).device
    net.blocks = nn.ModuleList([AdaptedBlock(block, rank).to(device) for block in net.blocks])
    for block in net.blocks:
        block.adapter.requires_grad_(enabled)
    net.eval()
    return net


class ReaderStudyModel(nn.Module):
    def __init__(self, net, target_statistics, evidence, adaptive, adapter_rank=32, config=None):
        super().__init__()
        if evidence not in ("mean", "future"):
            raise ValueError("evidence must be mean or future")
        self.evidence, self.adaptive = evidence, adaptive
        self.config = config or RefinementConfig()
        self.probe = EvidenceProbe(
            self.config, target_statistics,
            target_statistics["future"]["mean"], target_statistics["future"]["std"],
        )
        self.net = attach_adapters(net, adapter_rank, adaptive)

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()  # Preserve original reader inference behavior/no dropout.
        self.probe.train(mode)
        return self

    def condition(self, batch):
        f = batch["future_c"] if self.evidence == "future" else self.probe.memory_mean[None].expand(len(batch["past_c"]), -1, -1)
        return self.probe(batch["past_h"], batch["past_z"], batch["past_c"], f)

    def forward(self, batch, seed):
        c = self.condition(batch)
        x, tau, target = flow_draw(batch["past_actions"].detach(), seed)
        pred = self.net(x, tau, batch["past_z"].detach(), per_token=c)
        return (pred.float() - target.float()).square().mean()

    def adapter_parameters(self):
        return [p for block in self.net.blocks for p in block.adapter.parameters()]

    def base_parameters(self):
        return [(name, p) for name, p in self.net.named_parameters() if ".adapter." not in name]

    def compact_state(self):
        return {
            "probe": {k: v.detach().cpu() for k, v in self.probe.state_dict().items()},
            "adapters": [
                {k: v.detach().cpu() for k, v in block.adapter.state_dict().items()}
                for block in self.net.blocks
            ],
        }

    def load_compact_state(self, state):
        self.probe.load_state_dict(state["probe"], strict=True)
        if len(state["adapters"]) != len(self.net.blocks):
            raise ValueError("adapter depth mismatch")
        for block, saved in zip(self.net.blocks, state["adapters"]):
            block.adapter.load_state_dict(saved, strict=True)

    @torch.no_grad()
    def sample(self, batch, seed, mean_intervention=False):
        c = self.condition(batch)
        if mean_intervention:
            c = self.probe(
                batch["past_h"], batch["past_z"], batch["past_c"],
                self.probe.memory_mean[None].expand(len(batch["past_c"]), -1, -1),
            )
        z = batch["past_z"]
        generator = torch.Generator(device=z.device).manual_seed(int(seed))
        x = torch.randn(
            (len(z), self.net.future_action_window_size + 1, self.net.in_channels),
            generator=generator, device=z.device, dtype=z.dtype,
        )
        ts = torch.linspace(0, 1, 10, device=z.device, dtype=z.dtype)
        for i in range(1, 10):
            x = x + self.net(x, ts[i - 1].expand(len(z)), z, per_token=c) * (ts[i] - ts[i - 1])
        return x
