"""Exercise the real trainer's update/save/eval/restart/progress on a tiny PPU model.

Synthetic tensors are explicitly test-only. Main entrypoint never imports this.
"""

from types import SimpleNamespace

import torch
from torch import nn

from stage3_perf import train
from stage3_scale.model import JointPolicy
from stage3.refinement import RefinementConfig
from stage3_scale.oxe_data import oxe_source_hashes


class VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = nn.Linear(6, 20)
        self.projector = nn.Linear(20, 20)
        self.llm_backbone = nn.Embedding(64, 20)

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        x = self.llm_backbone(input_ids) + self.projector(self.vision_backbone(pixel_values["rgb"]))[:, None]
        return SimpleNamespace(hidden_states=[x])


class Past(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.vae, self.v_proj = nn.Linear(12, 12), nn.Linear(5, 12), nn.Linear(12, 4)

    def encode_queries(self, features, vae):
        return self.encoder(features) + self.vae(vae)


class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.z, self.c = nn.Linear(20, 7), nn.Linear(4, 7)

    def forward(self, x, tau, z, per_token):
        return .1 * x + self.z(z) + self.c(per_token.mean(1))[:, None]


class Visual(nn.Module):
    def forward(self, current, past, future):
        x = past["rgb"]
        return {
            "history_features": x.repeat(1, 1, 2),
            "history_vae": x[:, :, :5],
            "future_condition": future["rgb"][:, :, :4],
        }


def tiny_load(seed, teacher):
    if teacher:
        raise ValueError("synthetic probe supports forecast only")
    torch.manual_seed(seed)
    config = RefinementConfig(feature_dim=12, cognition_dim=20, tokens=3, condition_dim=4,
                              r_hidden=16, r_layers=1, r_heads=4, k_hidden=16, k_layers=1, k_heads=4)
    stats = {key: {"mean": torch.zeros(3, 4), "std": torch.ones(3, 4)} for key in ("past", "future")}
    policy = JointPolicy(VLM(), Past(), DiT(), stats, config)
    with torch.no_grad():
        policy.k.output.weight.normal_(std=.02)
    return policy, Visual(), None, None, None, None


class Data:
    def __init__(self, *args):
        self.meta = {"oxe_source_sha256": oxe_source_hashes()}

    def batch(self, n, seed, step, micro, rank, device, augment):
        generator = torch.Generator().manual_seed(seed + step * 100 + micro + rank * 10000)
        def rand(*shape):
            return torch.randn(*shape, generator=generator)
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4]]).expand(n, -1).clone(),
            "attention_mask": torch.ones(n, 4, dtype=torch.bool),
            "current": {"rgb": rand(n, 6)}, "past": {"rgb": rand(n, 3, 6)},
            "future": {"rgb": rand(n, 3, 6)}, "actions": rand(n, 16, 7),
            "action_mask": torch.ones(n, 16, dtype=torch.bool),
            "identities": [f"synthetic/{rank}/{step}/{i}" for i in range(n)],
        }


def fake_eval(model, visual, root, output, *args):
    # Test trainer phase control/I/O only; raw-image action eval remains tested separately.
    import json
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(json.dumps({
        "complete": True, "synthetic_probe_only": True, "not_scientific_evaluation": True,
    }))


if __name__ == "__main__":
    import stage3_scale.evaluate_joint
    train.load_joint = tiny_load
    train.SingleParseWindows = Data
    stage3_scale.evaluate_joint.evaluate = fake_eval
    train.main()
