"""WhenAmI Stage 2: contracts for a causal, WoG-style history condition.

Stage 1 = Future (the existing WoG oracle); Stage 2 = Past (this module).
Stage 3/4 are NOT implemented here. No transition, uncertainty or distillation
is implied by this first history baseline.

The QFormer architecture is unchanged. DINO reads four historical frames; Wan
reads those frames followed by the current frame, in chronological order.
Actions are supervision only in this baseline, not an input to the conditioner.
"""

from dataclasses import dataclass
from typing import Sequence

import torch


DEFAULT_PAST_OFFSETS = (-15, -11, -7, -3, 0)


@dataclass(frozen=True)
class PastSpec:
    offsets: tuple = DEFAULT_PAST_OFFSETS
    horizon: int = 16

    def __post_init__(self):
        if self.horizon <= 0:
            raise ValueError("action horizon must be positive")
        if len(self.offsets) < 2 or any(
            not isinstance(o, int) or isinstance(o, bool) for o in self.offsets
        ):
            raise ValueError("past offsets must be integer raw-trajectory timesteps")
        if tuple(sorted(set(self.offsets))) != tuple(self.offsets) or self.offsets[-1] != 0:
            raise ValueError("past offsets must be strictly increasing and end at 0; no future frames")
        # Wan's causal encoder consumes one initial frame plus groups of four.
        if (len(self.offsets) - 1) % 4:
            raise ValueError("Wan VAE requires 1 + 4*k history/current frames (default: 5)")

    @classmethod
    def from_offsets(cls, offsets=None, horizon=16):
        return cls(tuple(DEFAULT_PAST_OFFSETS if offsets is None else offsets), horizon)

    @property
    def lookback(self):
        # Number of raw action steps, NOT number of sampled historical frames.
        return -self.offsets[0]

    @property
    def n_past(self):
        return len(self.offsets) - 1

    def metadata(self):
        return {
            "schema_version": 1,
            "stage": 2,
            "conditioner": "past_qformer",
            "obs_offsets": list(self.offsets),
            "action_horizon": self.horizon,
            "history_action_input": False,
            "video_order": "past_then_current",
        }

    def action_targets(self, actions, masks=None):
        """Select a_t ... a_{t+Ta-1}, irrespective of sparse image sampling."""
        start, end = self.lookback, self.lookback + self.horizon
        if actions is None or actions.ndim != 3 or actions.shape[1] < end:
            raise ValueError(
                f"Stage 2 needs action coverage [-{start}, {self.horizon - 1}], "
                f"i.e. at least {end} action rows; got {getattr(actions, 'shape', None)}"
            )
        target = actions[:, start:end]
        if masks is not None:
            if masks.ndim != 2 or masks.shape != actions.shape[:2]:
                raise ValueError("action masks must have shape [batch, raw action window]")
            masks = masks[:, start:end]
        return target, masks

    def visual_inputs(self, current, past, future=None):
        """Return DINO history and chronological VAE video; reject privilege."""
        if future and any(v.shape[1] != 0 for v in future.values()):
            raise ValueError("Stage 2 must not receive future observations")
        if not current or not past or set(current) != set(past):
            raise ValueError("Stage 2 requires current pixels and matching historical modalities")
        video = {}
        for key, now in current.items():
            hist = past[key]
            if hist.shape != (now.shape[0], self.n_past, *now.shape[1:]):
                raise ValueError(
                    f"{key}: expected {self.n_past} historical frames matching current pixels, "
                    f"got history={tuple(hist.shape)}, current={tuple(now.shape)}"
                )
            video[key] = torch.cat([hist, now.unsqueeze(1)], dim=1)
        return past, video


def checkpoint_spec(checkpoint, requested_stage=None, requested_offsets=None, horizon=16):
    """Infer Stage 2 from checkpoint payload, never a mutable run-directory config.

    Loading a Future checkpoint with requested_stage=2 is an explicit warm start.
    Loading a Past checkpoint as an old Future/flow model is rejected.
    """
    model = checkpoint.get("model", {})
    meta = checkpoint.get("whenami")
    saved_stage = meta.get("stage") if meta else None
    has_past = "past_qformer" in model
    if has_past and (not meta or saved_stage != 2):
        raise ValueError("past_qformer checkpoint is missing its Stage-2 metadata")
    if saved_stage == 2 and not has_past:
        raise ValueError("Stage-2 checkpoint has no past_qformer weights")
    if saved_stage == 2 and (
        meta.get("schema_version") != 1
        or meta.get("conditioner") != "past_qformer"
        or meta.get("history_action_input") is not False
        or meta.get("video_order") != "past_then_current"
    ):
        raise ValueError("Unsupported Stage-2 checkpoint contract; refusing to guess its input semantics")
    stage = saved_stage if requested_stage is None and saved_stage is not None else requested_stage
    stage = 0 if stage is None else stage
    if stage not in (0, 1, 2):
        raise ValueError("Only legacy (0), Future (1), and Past (2) are implemented")
    offsets = requested_offsets
    if has_past:
        if stage != 2:
            raise ValueError("Cannot load a Past checkpoint as Future or a legacy temporal model")
        if horizon != meta["action_horizon"]:
            raise ValueError("action horizon differs from the saved Stage-2 checkpoint")
        if offsets is not None and list(offsets) != meta["obs_offsets"]:
            raise ValueError("obs_offsets differ from the saved Stage-2 checkpoint")
        offsets = meta["obs_offsets"]
    if stage == 2:
        offsets = list(PastSpec.from_offsets(offsets, horizon).offsets)
    return stage, offsets


def history_frames(observations: Sequence, offsets=DEFAULT_PAST_OFFSETS):
    """Sample past-only PIL frames from an episode-local list ending at now.

    Callers append one observation per environment step and reset per episode.
    At episode start repeat the first observation, as RLDS boundary clipping does.
    The returned list excludes now, for predict_action(past_frames=...).
    """
    spec = PastSpec.from_offsets(offsets)
    if len(observations) == 0:
        raise ValueError("history must contain at least the current observation")
    t = len(observations) - 1
    return [observations[max(0, t + offset)] for offset in spec.offsets[:-1]]
