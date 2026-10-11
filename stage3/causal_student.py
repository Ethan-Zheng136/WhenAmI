"""Causal forecast-and-correct student with optional frozen-teacher supervision.

The policy API accepts Past H/z/C only. True Future C is confined to the target
loss / teacher branch; neither true nor noisy actions enter R or K.
"""

import torch
from torch import nn

from stage3.behavior import flow_draw
from stage3.connector import ConnectorConfig, SharedConnector
from stage3.reader_adaptation import ReaderStudyModel

VARIANTS = ("latent", "forecast", "distill_past", "distill_future")
SEEDS = (42, 43, 44, 45)


class CausalStudent(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        c = policy.config
        stats = {
            "past": {"mean": torch.zeros_like(policy.probe.past_scale), "std": policy.probe.past_scale},
            "future": {"mean": policy.probe.memory_mean, "std": policy.probe.memory_scale},
        }
        self.r = SharedConnector(ConnectorConfig(
            feature_dim=c.feature_dim, cognition_dim=c.cognition_dim, tokens=c.tokens,
            condition_dim=c.condition_dim, hidden=c.r_hidden, layers=c.r_layers,
            heads=c.r_heads, scale_floor=c.scale_floor,
        ), stats)
        self.r.freeze_backward_only_adapters()

    def train(self, mode=True):
        super().train(mode)
        self.policy.net.eval()
        return self

    def forward(self, past_h, past_z, past_c):
        """No future arguments, target actions, or noisy actions in this API."""
        future = self.r.predict(past_h.detach(), past_z.detach(), "forward")
        standardized = (future - self.policy.probe.memory_mean) / self.policy.probe.memory_scale
        # Do NOT call EvidenceProbe.forward: its oracle-only evidence detach
        # would cut the student R->K action gradient.
        delta = self.policy.probe.k(past_h.detach(), past_z.detach(), standardized)
        corrected = past_c.detach() + self.policy.probe.past_scale * delta
        return {"condition": corrected, "future_prediction": future}

    def compact_state(self):
        return {
            "r": {k: v.detach().cpu() for k, v in self.r.state_dict().items()},
            "policy": self.policy.compact_state(),
        }

    def load_compact_state(self, state):
        self.r.load_state_dict(state["r"], strict=True)
        self.policy.load_compact_state(state["policy"])

    @torch.no_grad()
    def sample(self, past_h, past_z, past_c, seed):
        c = self(past_h, past_z, past_c)["condition"]
        z = past_z
        generator = torch.Generator(device=z.device).manual_seed(int(seed))
        x = torch.randn(
            (len(z), self.policy.net.future_action_window_size + 1, self.policy.net.in_channels),
            generator=generator, device=z.device, dtype=z.dtype,
        )
        ts = torch.linspace(0, 1, 10, device=z.device, dtype=z.dtype)
        for i in range(1, 10):
            x = x + self.policy.net(x, ts[i - 1].expand(len(z)), z, per_token=c) * (ts[i] - ts[i - 1])
        return x


def past_inputs(batch):
    return [batch[f"past_{k}"].detach() for k in ("h", "z", "c")]


def student_loss(student, teacher, batch, variant, seed, draws=4, alignment_weight=.1, distill_weight=.3):
    if variant not in VARIANTS or draws < 1 or min(alignment_weight, distill_weight) < 0:
        raise ValueError("invalid variant/draw count/loss weights")
    needs_teacher = variant.startswith("distill_")
    if needs_teacher and teacher is None:
        raise ValueError("distillation variant requires its frozen teacher")
    if needs_teacher:
        expected = "future" if variant == "distill_future" else "mean"
        if teacher.evidence != expected or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError("teacher evidence or freezing mismatch")
    output = student(*past_inputs(batch))
    actions = batch["past_actions"].detach().repeat_interleave(draws, dim=0)
    z = batch["past_z"].detach().repeat_interleave(draws, dim=0)
    c = output["condition"].repeat_interleave(draws, dim=0)
    x, tau, truth = flow_draw(actions, seed)
    velocity = student.policy.net(x, tau, z, per_token=c)
    action_loss = (velocity.float() - truth.float()).square().mean()
    total = action_loss
    metrics = {"action_fm": action_loss}
    if variant != "latent":
        align = student.r.alignment_loss(output["future_prediction"], batch["future_c"], "future")
        total = total + alignment_weight * align
        metrics["future_alignment"] = align
    if needs_teacher:
        teacher.eval()
        with torch.no_grad():
            teacher_c = teacher.condition(batch).repeat_interleave(draws, dim=0)
            teacher_velocity = teacher.net(x, tau, z, per_token=teacher_c)
        distill = (velocity.float() - teacher_velocity.detach().float()).square().mean()
        total = total + distill_weight * distill
        metrics["behavior_distillation"] = distill
        metrics["teacher_action_fm"] = (teacher_velocity.float() - truth.float()).square().mean()
    metrics["total"] = total
    return total, metrics


def cosine_learning_rate(step, max_steps, peak=1e-4, warmup=500, floor_ratio=.1):
    """Absolute-step schedule allows matched runs and unchanged-budget resumes."""
    import math
    if not (1 <= step <= max_steps) or not (0 <= warmup < max_steps):
        raise ValueError("invalid schedule step/budget")
    if warmup and step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return peak * (floor_ratio + (1 - floor_ratio) * .5 * (1 + math.cos(math.pi * progress)))


def load_reader_checkpoint(path, cache, device, frozen=False, expected_evidence=None):
    """Strictly match teacher/base to the same endpoints and train/dev cache."""
    from pathlib import Path
    from stage3.behavior import load_behavior
    from stage3.common import sha256
    from stage3.refinement import RefinementConfig
    path = Path(path)
    cp = torch.load(path, map_location="cpu", weights_only=True)
    meta = cp["metadata"]
    if meta["schema"] != "whenami_reader_study_v1":
        raise ValueError("not a Reader Study checkpoint")
    if meta["cache_manifest_sha256"] != cache.manifest_sha256 or cp["step"] != 5000:
        raise ValueError("expected same-cache final5k Reader Study checkpoint")
    settings = meta["settings"]
    if expected_evidence is not None and settings["evidence"] != expected_evidence:
        raise ValueError("wrong teacher/base evidence")
    if not settings["adaptive"]:
        raise ValueError("student initialization and teachers use adaptive Reader Study checkpoints")
    for side in ("past", "future"):
        for key in ("path", "bytes", "mtime_ns", "config_sha256", "statistics_sha256"):
            if meta["endpoints"][side][key] != cache.manifest["endpoints"][side][key]:
                raise ValueError("checkpoint/cache endpoint mismatch")
    reader = load_behavior(cache.manifest["endpoints"]["past"], device)
    model = ReaderStudyModel(
        reader.net, cache.target_statistics(), settings["evidence"], True,
        settings["adapter_rank"], RefinementConfig(**meta["model_config"]),
    ).to(device)
    model.load_compact_state(cp["state"])
    if frozen:
        model.requires_grad_(False).eval()
    return model, {
        "path": str(path.resolve()), "sha256": sha256(path), "step": cp["step"],
        "seed": settings["seed"], "evidence": settings["evidence"],
    }
