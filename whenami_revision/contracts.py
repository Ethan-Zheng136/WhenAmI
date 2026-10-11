"""Explicit event/evidence-time separation, without CUDA/TF/model dependencies."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def canonical_anchors(anchors, cutoff, length):
    """Use actual observed frames only. Repeated boundary indices are ONE event."""
    if any(not isinstance(v, Integral) or isinstance(v, bool) for v in (cutoff, length)):
        raise ValueError("frame indices must be integers, not rounded timestamps")
    if length < 2 or not 0 <= cutoff < length:
        raise ValueError("cutoff outside episode")
    anchors = tuple(anchors)
    if any(not isinstance(k, Integral) or isinstance(k, bool) for k in anchors):
        raise ValueError("frame indices must be integers")
    anchors = tuple(int(k) for k in anchors)
    if not anchors or any(k < 0 or k >= length for k in anchors):
        raise ValueError("invalid event frame index")
    if tuple(sorted(anchors)) != anchors:
        raise ValueError("event times must be ordered, not silently sorted")
    # Unavailable anchors are physically removed from the policy-facing view.
    visible = tuple(dict.fromkeys(k for k in anchors if k <= cutoff))
    if not visible or visible[-1] != cutoff:
        raise ValueError("policy view must end at its current evidence cutoff")
    return visible


@dataclass(frozen=True)
class EvidencePrefix:
    """A policy-facing raw prefix. No target actions, future images or phase labels."""
    dataset: str
    episode_key: str
    source_times: tuple[int, ...]  # Metadata for audit; never fed as absolute progress.
    cutoff: int
    images: tuple[bytes, ...]
    instructions: tuple[str, ...]
    interval_actions: tuple[np.ndarray, ...]  # interval[i] = actions[t[i-1]:t[i]]
    interval_action_times: tuple[tuple[int, ...], ...]

    def validate(self):
        n = len(self.source_times)
        if n == 0 or not (n == len(self.images) == len(self.instructions) ==
                           len(self.interval_actions) == len(self.interval_action_times)):
            raise ValueError("unaligned prefix")
        if self.source_times[-1] != self.cutoff or any(
                a >= b for a, b in zip(self.source_times, self.source_times[1:])):
            raise ValueError("duplicate/unordered anchors or wrong cutoff")
        if any(not isinstance(k, Integral) or isinstance(k, bool) or k < 0 for k in self.source_times):
            raise ValueError("event times must be nonnegative integer frame indices")
        if not all(self.images):
            raise ValueError("missing encoded image")
        for i, (actions, times) in enumerate(zip(self.interval_actions, self.interval_action_times)):
            expected = () if i == 0 else tuple(range(self.source_times[i - 1], self.source_times[i]))
            if times != expected:
                raise ValueError("action interval must include exactly the PRECEDING executed actions")
            if actions.shape != (len(times), 7) or not np.isfinite(actions).all():
                raise ValueError("invalid action sequence")
            if times and max(times) >= self.source_times[i]:
                raise ValueError("action at/after the observation is not yet executed")
        return self

    @property
    def relative_times(self):
        # Removes global episode clock; deltas remain real frame-index differences.
        return np.asarray(self.source_times, dtype=np.int64) - self.source_times[0]

    @property
    def event_ids(self):
        return tuple(f"{self.dataset}/{self.episode_key}/{k}" for k in self.source_times)


@dataclass(frozen=True)
class ActionSupervision:
    """Labels kept OUTSIDE EvidencePrefix and model.forward kwargs."""
    target_time: int
    evidence_cutoff: int
    raw_actions: np.ndarray
    valid: np.ndarray

    def validate(self):
        if self.target_time != self.evidence_cutoff:
            raise ValueError("current action supervision cannot use a retrospective state with future actions")
        if self.raw_actions.shape != (16, 7) or self.valid.shape != (16,) or self.valid.dtype != np.bool_:
            raise ValueError("wrong action-label shape/mask")
        if not self.valid.any() or not np.isfinite(self.raw_actions).all():
            raise ValueError("invalid current action label")
        return self


def current_action_supervision(actions, target_time, evidence_cutoff):
    if any(not isinstance(k, Integral) or isinstance(k, bool) for k in (target_time, evidence_cutoff)):
        raise ValueError("action times must be integer frame indices")
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError("actions must be [length,7]")
    if target_time != evidence_cutoff:
        raise ValueError("do not predict old actions from an action-informed retrospective state")
    if not 0 <= target_time < len(actions):
        raise ValueError("target outside episode")
    count = min(16, len(actions) - target_time)
    value = np.zeros((16, 7), dtype=np.float32)
    value[:count] = actions[target_time:target_time + count]
    value[count:, 6] = actions[-1, 6]
    return ActionSupervision(target_time, evidence_cutoff, value, np.arange(16) < count).validate()


def revision_pair(earlier, later, event_time):
    """Same-event versions, never mixing episodes or changing already seen evidence."""
    earlier.validate(); later.validate()
    if (earlier.dataset, earlier.episode_key) != (later.dataset, later.episode_key):
        raise ValueError("revision must stay in one episode")
    if not event_time <= earlier.cutoff < later.cutoff:
        raise ValueError("revision needs later real evidence")
    if event_time not in earlier.source_times or event_time not in later.source_times:
        raise ValueError("event absent from one version")
    if later.source_times[:len(earlier.source_times)] != earlier.source_times:
        raise ValueError("revision cannot change the observed anchor prefix")
    n = len(earlier.source_times)
    if earlier.images != later.images[:n] or earlier.instructions != later.instructions[:n]:
        raise ValueError("old evidence changed")
    for lhs, rhs in zip(earlier.interval_actions, later.interval_actions):
        if not np.array_equal(lhs, rhs):
            raise ValueError("old executed action changed")
    return {"event_id": f"{earlier.dataset}/{earlier.episode_key}/{event_time}",
            "earlier_cutoff": earlier.cutoff, "later_cutoff": later.cutoff}
