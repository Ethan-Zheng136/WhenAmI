"""Full OXE27 training, fixed future horizon; no new dataset preprocessing."""
import numpy as np
import torch

from whenami_timeline_joint.data import Inputs as OriginalInputs, DEV
from whenami_timeline_b.data import batch as original_batch


def sample_times(length, rng, future_gap=4, max_history_gap=12):
    if future_gap < 1 or length < future_gap + 3:
        raise ValueError("episode too short for one past/current/future triplet")
    past_gap = int(rng.integers(1, min(max_history_gap, length - future_gap - 2) + 1))
    start = int(rng.integers(0, length - 1 - past_gap - future_gap))
    return np.asarray([start, start + past_gap, start + past_gap + future_gap], np.int64)


class Inputs(OriginalInputs):
    def __init__(self, future_gap=4, dev_root=DEV, dev_limit=0):
        super().__init__(nodes=3, dev_root=dev_root, dev_limit=dev_limit)
        self.future_gap = int(future_gap)
        self.raw.episodes = {
            d: [e for e in values if e["length"] >= self.future_gap + 3]
            for d, values in self.raw.episodes.items()
        }
        if any(not values for values in self.raw.episodes.values()):
            raise ValueError("fixed horizon removed an entire domain")
        self.eligible_counts = {d: len(values) for d, values in self.raw.episodes.items()}

    def batch(self, count, seed, step, rank, micro=0):
        cases = []
        for ordinal in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([seed, step, rank, micro * count + ordinal]))
            name = self.raw.domains[int(rng.choice(len(self.raw.domains), p=self.raw.probabilities))]
            candidates = self.raw.episodes[name]
            for _ in range(512):
                entry = candidates[int(rng.integers(len(candidates)))]
                _, episode = self.raw._episode(entry)
                times = sample_times(episode["length"], rng, self.future_gap)
                texts = episode["instructions"][int(times[0]):int(times[-1]) + 1]
                if not texts[0].strip() or any(t != texts[0] for t in texts):
                    continue
                if any(int(episode["frame_spans"][int(t), 1]) <= 0 for t in times):
                    continue
                observations, labels, provenance = self.raw.materialize(entry, times, self.stats)
                cases.append({"observations": observations, "labels": labels, "provenance": provenance})
                break
            else:
                raise RuntimeError("no valid fixed-horizon training sample in " + name)
        return cases


def batch(cases, visual, transform, tokenizer, prompt, device, future_gap=4):
    full, features, prefix, label = original_batch(cases, visual, transform, tokenizer, prompt, device)
    if not torch.equal(full["times"][:, 2] - full["times"][:, 1],
                       torch.full_like(full["times"][:, 1], future_gap)):
        raise ValueError("teacher future time differs from student prediction horizon")
    return full, features, prefix, label
