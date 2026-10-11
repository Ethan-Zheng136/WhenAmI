"""Small evidence-time revision prototype, not a replacement for the running VLA.

F predicts an outcome from a causal filtered state + executed action sequence.
After a REAL observation arrives, B revises old event nodes; D reads the bank.
B is an asymmetric evidence updater, never a claimed inverse of F.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class RevisionState:
    times: torch.Tensor             # [B,K], actual event times
    filtered: torch.Tensor          # [B,K,H], b[k|k] retained for auditing
    revised: torch.Tensor           # [B,K,H], current b[k|t]


class EvidenceTimePolicy(nn.Module):
    def __init__(self, observation_dim=12, action_dim=7, hidden=64, state_classes=2, decisions=4,
                 mode="incremental"):
        super().__init__()
        if mode not in ("incremental", "direct_history"):
            raise ValueError("unknown research mode")
        self.mode = mode
        self.observation_dim, self.action_dim, self.hidden = observation_dim, action_dim, hidden
        self.observe = nn.Sequential(nn.Linear(observation_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.action_encoder = nn.GRUCell(action_dim, hidden)
        self.time_encoder = nn.Sequential(nn.Linear(3, hidden), nn.Tanh())
        self.initial = nn.Parameter(torch.zeros(1, hidden))
        self.transition = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.predict_observation = nn.Linear(hidden, observation_dim)
        self.filter_update = nn.GRUCell(3 * hidden, hidden)
        self.revise = nn.Sequential(nn.Linear(5 * hidden, hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden), nn.Tanh())
        # Nonzero output: action gradients can reach the revision trunk from the outset.
        nn.init.normal_(self.revise[-2].weight, std=.005)
        nn.init.zeros_(self.revise[-2].bias)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.state_head = nn.Linear(hidden, state_classes)
        self.decision_head = nn.Linear(hidden, decisions)

    def time_features(self, gap):
        # Actual interval, not normalized episode progress or an absolute clock.
        x = gap.to(self.initial.dtype)
        return self.time_encoder(torch.stack((torch.log1p(x), x / 16., torch.sqrt(x + 1) - 1), -1))

    def encode_executed(self, actions, valid):
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim or valid.shape != actions.shape[:2]:
            raise ValueError("invalid executed action interval")
        if valid.dtype != torch.bool:
            raise ValueError("executed mask must be boolean")
        if valid.shape[1] > 1 and ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("interval padding must be trailing")
        if not torch.isfinite(actions[valid]).all():
            raise ValueError("nonfinite observed action")
        h = actions.new_zeros(len(actions), self.hidden)
        # Mask before encoding; unavailable padding can never alter a prediction.
        for j in range(actions.shape[1]):
            x = torch.where(valid[:, j, None], actions[:, j], torch.zeros_like(actions[:, j]))
            update = self.action_encoder(x, h)
            h = torch.where(valid[:, j, None], update, h)
        return h

    def step(self, observation, time, executed_actions, executed_mask, state=None, revise=True):
        """Process one new evidence arrival. No future labels/observations accepted."""
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError("one feature vector per current observation required")
        if time.shape != observation.shape[:1] or time.dtype not in (torch.int32, torch.int64):
            raise ValueError("time must be an integer frame index per batch element")
        if not torch.isfinite(observation).all():
            raise ValueError("invalid observation")
        u = self.encode_executed(executed_actions, executed_mask)
        if state is None:
            if executed_mask.any():
                raise ValueError("first anchor has no represented predecessor action interval")
            gap = torch.zeros_like(time)
            old = self.initial.expand(len(observation), -1)
        else:
            gap = time - state.times[:, -1]
            if (gap <= 0).any():
                raise ValueError("evidence arrival must advance time; no duplicate evidence updates")
            if not torch.equal(executed_mask.sum(1), gap):
                raise ValueError("sparse anchors require the complete intervening action sequence")
            old = state.filtered[:, -1]
        dt = self.time_features(gap)
        prior = old + self.transition(torch.cat((old, u, dt), -1))
        predicted_observation = self.predict_observation(prior)
        evidence = self.observe(observation)
        predicted_evidence = self.observe(predicted_observation)
        innovation = evidence - predicted_evidence
        now = self.filter_update(torch.cat((evidence, u, dt), -1), prior)
        if state is None:
            bank, filtered, times = now[:, None], now[:, None], time[:, None]
        else:
            n = state.revised.shape[1]
            age = self.time_features(time[:, None] - state.times)
            # Strong matched-capacity control: direct_history can use the FULL
            # current filtered state, not only an innovation or an old snapshot.
            source = state.revised if self.mode == "incremental" else state.filtered
            message = innovation if self.mode == "incremental" else now
            update = self.revise(torch.cat((
                source, state.filtered, message[:, None].expand(-1, n, -1),
                u[:, None].expand(-1, n, -1), age,
            ), -1))
            bank = torch.cat((source + update if revise else state.filtered, now[:, None]), 1)
            filtered = torch.cat((state.filtered, now[:, None]), 1)
            times = torch.cat((state.times, time[:, None]), 1)
        # Only observed nodes enter memory readout. Current action depends on the
        # revised bank, not a teacher state or an expert future action.
        age = self.time_features(time[:, None] - times)
        score = (self.query(now)[:, None] * self.key(bank + age)).sum(-1) / math.sqrt(self.hidden)
        context = (score.softmax(1)[..., None] * self.value(bank)).sum(1)
        current = self.fuse(torch.cat((now, context), -1))
        new_state = RevisionState(times, filtered, bank)
        return new_state, {
            "decision_logits": self.decision_head(current),
            "event_logits": self.state_head(bank),
            "filtered_event_logits": self.state_head(filtered),
            "predicted_observation": predicted_observation,
            "current_state": current,
        }

    def forward(self, observations, times, executed_actions, executed_mask, revise=True):
        if observations.ndim != 3 or observations.shape[:2] != times.shape or observations.shape[1] < 1:
            raise ValueError("nonempty aligned [B,T,feature] prefix required")
        if executed_actions.ndim != 4 or executed_actions.shape[:2] != times.shape:
            raise ValueError("aligned action intervals required")
        if executed_mask.shape != executed_actions.shape[:-1]:
            raise ValueError("action mask mismatch")
        state, outputs = None, []
        for i in range(observations.shape[1]):
            state, output = self.step(observations[:, i], times[:, i],
                                      executed_actions[:, i], executed_mask[:, i], state, revise)
            outputs.append(output)
        return {
            **outputs[-1], "bank": state.revised, "state": state,
            "decision_history": torch.stack([o["decision_logits"] for o in outputs], 1),
            "belief_versions": tuple(o["event_logits"] for o in outputs),
            "observation_predictions": torch.stack([o["predicted_observation"] for o in outputs], 1),
        }


class RevisionConditionBridge(nn.Module):
    """Optional zero-initialized bridge to existing [B,16,32] action conditions.

    Unit-testable interface only. It is NOT wired into the production 8B policy
    by this module; that needs a separate training recipe.
    """
    def __init__(self, hidden=64, tokens=16, condition_dim=32, heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, tokens, hidden) * .02)
        self.attend = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0)
        self.output = nn.Linear(hidden, condition_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base_condition, observed_bank):
        query = self.queries.expand(len(observed_bank), -1, -1)
        delta, _ = self.attend(query, observed_bank, observed_bank, need_weights=False)
        if base_condition.shape != (len(observed_bank), self.queries.shape[1], self.output.out_features):
            raise ValueError("base condition must match bridge token coordinates")
        return base_condition + self.output(delta)
