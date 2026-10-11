"""Reuse exact OXE sampling/augmentation, overlap bounded CPU preparation only."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from stage3_scale.oxe_data import OXERawWindows


class SingleParseWindows(OXERawWindows):
    """Avoid a second 313MB JSON parse and 27 full episode-list scans per rank."""
    def __init__(self, index, transform, tokenizer, prompt_builder, statistics, allow_partial=False):
        self.index_path = Path(index)
        self.meta = json.loads(self.index_path.read_text())
        if self.meta["schema"] != "whenami_raw_full_train_index_v1":
            raise ValueError("wrong raw index")
        if not self.meta["complete"] and not allow_partial:
            raise ValueError("partial data index is smoke-only")
        if self.meta.get("data_mix") != "oxe_magic_soup_plus":
            raise ValueError("require OXE27 index")
        self.transform, self.tokenizer, self.prompt_builder = transform, tokenizer, prompt_builder
        self.statistics = statistics
        self.domains = sorted(self.meta["window_counts"])
        self.episodes = {d: [] for d in self.domains}
        for episode in self.meta["episodes"]:
            self.episodes[episode["dataset"]].append(episode)
        self.ends = {d: np.cumsum([e["windows"] for e in self.episodes[d]]) for d in self.domains}
        if any(not len(self.ends[d]) or self.ends[d][-1] != self.meta["window_counts"][d] for d in self.domains):
            raise ValueError("invalid domain windows")
        self.loaded, self.verified = {}, set()
        self.file_records = {r["file"]: r for r in self.meta["shard_records"]}
        probs = self.meta["sampling_probabilities"]
        if self.meta["complete"] and set(probs) != set(self.domains):
            raise ValueError("full index lacks required domains")
        p = np.asarray([probs[d] for d in self.domains], dtype=np.float64)
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError("invalid domain weights")
        self.probabilities = p / p.sum()


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value  # identity strings stay on host


class BatchPrefetch:
    """One CPU worker owns the loader. No global RNG or accelerator operations.

    Dataset.batch is a pure function of seed/step/micro/rank plus deterministic
    cache state. Resume discards queued work and regenerates those same batches.
    """
    def __init__(self, dataset, count, seed, start, stop, accum, rank, augment, depth=2):
        if depth < 0:
            raise ValueError("prefetch depth must be >=0")
        self.dataset, self.count, self.seed = dataset, count, seed
        self.rank, self.augment, self.depth = rank, augment, depth
        self.keys = iter((step, micro) for step in range(start + 1, stop + 1) for micro in range(accum))
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oxe-cpu") if depth else None
        self.queue = deque()
        self.closed = False
        if self.executor:
            self._fill()

    def _build(self, key):
        step, micro = key
        return self.dataset.batch(self.count, self.seed, step, micro, self.rank,
                                  torch.device("cpu"), augment=self.augment)

    def _fill(self):
        while len(self.queue) < self.depth:
            key = next(self.keys, None)
            if key is None:
                break
            self.queue.append((key, self.executor.submit(self._build, key)))

    def get(self, step, micro):
        if self.closed:
            raise RuntimeError("prefetcher closed")
        if not self.executor:
            key = next(self.keys)
            if key != (step, micro):
                raise ValueError("batch order changed")
            return self._build(key)
        key, future = self.queue.popleft()
        if key != (step, micro):
            raise ValueError("prefetch order changed")
        value = future.result()  # Propagate decoder errors, never silently skip.
        self._fill()
        return value

    def close(self):
        self.closed = True
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.queue.clear()
