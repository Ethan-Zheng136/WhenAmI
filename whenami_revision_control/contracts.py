from dataclasses import dataclass

import torch

from whenami_timeline_transfer.contracts import Prefix


@dataclass(frozen=True)
class ObservedWindow:
    """Observed visual features and EXECUTED commands only; no future targets."""
    patches: torch.Tensor
    vae: torch.Tensor
    times: torch.Tensor
    executed: torch.Tensor
    executed_mask: torch.Tensor

    def validate(self):
        if self.times.ndim != 2 or self.times.dtype not in (torch.int32, torch.int64):
            raise ValueError("integer event times [B,K] required")
        b, k = self.times.shape
        if not 1 <= k <= 3 or self.patches.shape[:2] != (b, k) or self.vae.shape[:2] != (b, k):
            raise ValueError("one to three observed anchors, not an unbounded memory")
        if self.executed.ndim != 4 or self.executed.shape[:2] != (b, k) or self.executed.shape[-1] != 7:
            raise ValueError("executed command intervals [B,K,L,7]")
        if self.executed_mask.shape != self.executed.shape[:-1] or self.executed_mask.dtype != torch.bool:
            raise ValueError("boolean executed interval mask")
        if self.executed_mask[:, 0].any():
            raise ValueError("first anchor cannot have an unspecified predecessor")
        if ((~self.executed_mask[:, :, :-1]) & self.executed_mask[:, :, 1:]).any():
            raise ValueError("executed interval padding must be trailing")
        if not torch.isfinite(self.executed[self.executed_mask]).all():
            raise ValueError("nonfinite executed command")
        if k > 1:
            gap = self.times.diff(dim=1)
            if (gap <= 0).any() or not torch.equal(gap, self.executed_mask[:, 1:].sum(-1)):
                raise ValueError("unique chronological events and complete executed intervals required")
        return self

    def prefix(self, count):
        if not 1 <= count <= self.times.shape[1]:
            raise ValueError("invalid observed prefix")
        return ObservedWindow(*(getattr(self, n)[:, :count].clone() for n in (
            "patches", "vae", "times", "executed", "executed_mask"))).validate()


@dataclass(frozen=True)
class RevisionInput:
    base: Prefix                 # Original B's latest one/two images, independently encoded.
    window: ObservedWindow
    frames: tuple[Prefix, ...]   # Independent per-frame VLM inputs; no future-dependent padding.

    def validate(self):
        self.base.validate()
        self.window.validate()
        if len(self.frames) != self.window.times.shape[1]:
            raise ValueError("one independently tokenized VLM input per observed event")
        for frame in self.frames:
            frame.validate()
            if frame.times.shape != (self.window.times.shape[0], 1):
                raise ValueError("memory VLM inputs must be independent single frames")
        k = min(2, self.window.times.shape[1])
        expected = self.window.times[:, -k:]
        expected = expected - expected[:, :1]
        if not torch.equal(self.base.times, expected):
            raise ValueError("base B must read the latest observed pair with its original time origin")
        return self
