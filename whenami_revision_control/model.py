"""Three observed nodes; real teacher revisions feed the current action residual.

This is a finite-window prototype. It rebuilds each <=3-node window and is NOT an
unbounded cross-episode memory, physical inverse model, or future-action planner.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from whenami_timeline_a.model import Config, ReadEvidence
from whenami_timeline_b.model import Student, draw
from whenami_timeline_joint.model import load_joint
from whenami_revision_control.contracts import RevisionInput
from stage3_scale.model import cognition

VARIANTS = ("direct_history", "revised_nodes")


@dataclass
class NodeState:
    times: torch.Tensor
    originals: torch.Tensor
    bank: torch.Tensor
    local_conditions: torch.Tensor


class NodeMemory(nn.Module):
    def __init__(self, config=Config(), hidden=128, heads=4, mode="revised_nodes"):
        super().__init__()
        if mode not in VARIANTS:
            raise ValueError("unknown revision mode")
        self.mode, self.hidden, self.config = mode, hidden, config
        self.observe = nn.Sequential(nn.LayerNorm(config.feature), nn.Linear(config.feature, hidden))
        self.cognition = nn.Sequential(nn.LayerNorm(config.cognition), nn.Linear(config.cognition, hidden))
        self.position = nn.Parameter(torch.randn(1, config.queries, hidden) * .02)
        self.actions = nn.GRUCell(7, hidden)
        self.time = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())
        c = Config(hidden=hidden, heads=heads)
        self.blocks = nn.ModuleList([ReadEvidence(c) for _ in range(2)])
        self.node_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, config.condition))

    def age(self, gap):
        x = gap.to(self.position.dtype)
        return self.time(torch.stack((x / 32., torch.log1p(x) / 4., torch.sqrt(x + 1.) - 1.), -1))

    def encode_actions(self, actions, mask):
        value = actions.new_zeros(len(actions), self.hidden, dtype=self.position.dtype)
        for j in range(actions.shape[1]):
            x = torch.where(mask[:, j, None], actions[:, j].to(value.dtype), torch.zeros_like(actions[:, j], dtype=value.dtype))
            update = self.actions(x, value)
            value = torch.where(mask[:, j, None], update, value)
        return value

    def step(self, h, local_condition, time, actions, mask, state=None, z=None):
        if time.dtype not in (torch.int32, torch.int64):
            raise ValueError("integer event identity")
        if state is None:
            if mask.any():
                raise ValueError("initial event has no observed predecessor interval")
        else:
            if state.times.shape[1] >= 3:
                raise ValueError("rebuild the latest three-node window; no silent truncation of state")
            gap = time - state.times[:, -1]
            if (gap <= 0).any() or not torch.equal(mask.sum(-1), gap):
                raise ValueError("duplicate evidence or incomplete executed interval")
        action = self.encode_actions(actions, mask)
        observation = self.observe(h) + self.position + action[:, None]
        if z is not None:
            observation = observation + self.cognition(z)
        if state is None:
            originals, source = observation[:, None], observation[:, None]
            times, local = time[:, None], local_condition[:, None]
        else:
            originals = torch.cat((state.originals, observation[:, None]), 1)
            # Strong baseline: same complete observed memory/commands at every step,
            # but old queries are reconstructed rather than recurrently revised.
            old = state.bank if self.mode == "revised_nodes" else state.originals
            source = torch.cat((old, observation[:, None]), 1)
            times = torch.cat((state.times, time[:, None]), 1)
            local = torch.cat((state.local_conditions, local_condition[:, None]), 1)
        b, k, q, _ = source.shape
        age = self.age(time[:, None] - times)[:, :, None]
        query = (source + age).reshape(b, k*q, self.hidden)
        evidence = (originals + age).reshape(b, k*q, self.hidden)
        prohibited = torch.zeros((b, k*q), dtype=torch.bool, device=query.device)
        for block in self.blocks:
            query = block(query, evidence, prohibited)
        bank = query.reshape(b, k, q, self.hidden)
        conditions = local + self.node_head(bank)
        return NodeState(times, originals, bank, local), conditions

    def forward(self, h, local, window, z=None):
        window.validate()
        state, versions = None, []
        for j in range(window.times.shape[1]):
            state, value = self.step(h[:, j], local[:, j], window.times[:, j],
                                     window.executed[:, j], window.executed_mask[:, j], state,
                                     None if z is None else z[:, j])
            versions.append(value)
        return {"state": state, "versions": tuple(versions), "nodes": versions[-1]}


class ActionBridge(nn.Module):
    def __init__(self, config=Config(), hidden=128, heads=4):
        super().__init__()
        self.query = nn.Linear(config.condition, hidden)
        self.cognition = nn.Linear(config.cognition, hidden)
        self.evidence = nn.Linear(config.condition, hidden)
        self.time = nn.Sequential(nn.Linear(2, hidden), nn.Tanh())
        self.attend = ReadEvidence(Config(hidden=hidden, heads=heads))
        self.output = nn.Linear(hidden, config.condition)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base, cognition, old_nodes, ages):
        if old_nodes.shape[1] == 0:
            return base, torch.zeros_like(base)
        x = ages.to(base.dtype)
        temporal = self.time(torch.stack((x/32., torch.log1p(x)/4.), -1))[:, :, None]
        memory = (self.evidence(old_nodes) + temporal).flatten(1, 2)
        query = self.query(base) + self.cognition(cognition)
        mask = torch.zeros(memory.shape[:2], device=memory.device, dtype=torch.bool)
        hidden = self.attend(query, memory, mask)
        delta = self.output(hidden)
        return base + delta, delta


class RevisionPolicy(Student):
    def __init__(self, vlm, local_encoder, reader, variant="revised_nodes",
                 config=Config(), hidden=128, heads=4):
        super().__init__(vlm, local_encoder, reader, "timeline", config)
        self.variant = variant
        self.memory = NodeMemory(config, hidden, heads, variant)
        self.bridge = ActionBridge(config, hidden, heads)
        # Retained for exact B checkpoint coverage, but not part of this objective.
        self.forecast_head.requires_grad_(False)

    def forward(self, inputs=None, noisy_actions=None, tau=None, condition_only=False,
                conditioning=None, unrevised_action=False):
        if conditioning is not None:
            if inputs is not None or condition_only:
                raise ValueError("decoder-only request")
            return {"velocity": self.reader(noisy_actions, tau, conditioning["reader_z"],
                                             per_token=conditioning["conditions"])}
        if not isinstance(inputs, RevisionInput):
            raise TypeError("only an observed RevisionInput is accepted")
        inputs.validate()
        # Preserve original B forward path exactly, including independent token padding.
        base = super().forward(prefix=inputs.base, condition_only=True)
        w = inputs.window
        b, k = w.times.shape
        h = self.local_encoder.encode_queries(w.patches.flatten(0, 1), w.vae.flatten(0, 1))
        local = self.local_encoder.v_proj(h).reshape(b, k, *base["conditions"].shape[1:])
        h = h.reshape(b, k, *h.shape[1:])
        # Each node receives its own visual-language cognition. Never broadcast
        # the latest frame's cognition into earlier versions of old event states.
        z = torch.stack([cognition(
            self.vlm, frame.input_ids, frame.attention_mask,
            {name: value[:, 0] for name, value in frame.pixels.items()})
            for frame in inputs.frames], 1)
        encoded = self.memory(h, local, w, z)
        old = encoded["nodes"][:, :-1]
        if unrevised_action and k > 1:
            old = encoded["versions"][-2]
        conditions, delta = self.bridge(
            base["conditions"], base["reader_z"], old, w.times[:, -1:] - w.times[:, :-1])
        result = {"conditions": conditions, "reader_z": base["reader_z"],
                  "base_conditions": base["conditions"], "delta": delta,
                  "node_conditions": encoded["nodes"], "node_versions": encoded["versions"]}
        if condition_only:
            return result
        if noisy_actions is None or tau is None or len(noisy_actions) % b:
            raise ValueError("current noisy actions must repeat observations")
        repeats = len(noisy_actions) // b
        result["velocity"] = self.reader(
            noisy_actions, tau, base["reader_z"].repeat_interleave(repeats, 0),
            per_token=conditions.repeat_interleave(repeats, 0))
        return result


def build_student(variant="revised_nodes", seed=42):
    base, visual, transform, tokenizer, prompt = load_joint("timeline", seed)
    model = RevisionPolicy(base.vlm, base.local_encoder, base.reader, variant)
    model.timeline.load_state_dict(base.timeline.state_dict(), strict=True)
    return model, visual, transform, tokenizer, prompt


def objective(output, target, valid, teacher_before, teacher_after,
              node_weight=.1, revision_weight=.1, draws=2):
    if min(node_weight, revision_weight) < 0 or valid.ndim != 2 or not valid.any(-1).all():
        raise ValueError("loss configuration")
    mask = valid.detach().repeat_interleave(draws, 0)
    error = (output["velocity"].float() - target.detach().float()).square().mean(-1)
    action = ((error * mask).sum(-1) / mask.sum(-1)).mean()
    before = output["node_versions"][-2]
    after = output["node_conditions"][:, :-1]
    if before.shape != after.shape or after.shape != teacher_after.shape or before.shape != teacher_before.shape:
        raise ValueError("same old-node identities before/after new evidence required")
    normalize = lambda x: F.normalize(x.float(), dim=-1, eps=1e-6)
    sb, sa = normalize(before), normalize(after)
    tb, ta = normalize(teacher_before.detach()), normalize(teacher_after.detach())
    node = (1. - (sa * ta).sum(-1)).mean()
    revision = ((sa - sb) - (ta - tb)).square().sum(-1).mean()
    weighted = node_weight * node + revision_weight * revision
    loss = action + weighted
    return loss, {"action_fm": action.detach(), "node_alignment": node.detach(),
                  "revision_alignment": revision.detach(), "weighted_aux": weighted.detach(),
                  "delta_rms": output["delta"].float().square().mean().sqrt().detach(),
                  "total": loss.detach()}
