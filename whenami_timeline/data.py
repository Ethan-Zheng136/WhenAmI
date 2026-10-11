"""Raw training-only multi-anchor samples; targets separated from policy inputs."""

import hashlib
import io
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from whenami_revision.raw_data import RevisionEpisodes
from stage3_scale.oxe_data import action_window
from evaluation.past_protocol import normalize_actions


PAST_OFFSETS = (-15, -11, -7, -3)


def research_role(key):
    # A new INTERNAL split from original TRAIN ONLY. No main dev/sealed access.
    return "internal_dev" if int(hashlib.sha256(("timeline01/" + key).encode()).hexdigest()[:8], 16) % 20 == 0 else "train"


def choose_times(length, rng, nodes=3):
    if length < nodes + 1:
        raise ValueError("not enough actual frames for distinct anchors")
    maximum_gap = min(4, (length - 2) // (nodes - 1))
    gaps = rng.integers(1, maximum_gap + 1, size=nodes - 1)
    span = int(gaps.sum())
    start = int(rng.integers(0, length - 1 - span))
    return np.r_[start, start + np.cumsum(gaps)].astype(np.int64)


class RawTimelines(RevisionEpisodes):
    def __init__(self, index, role="train", nodes=3):
        super().__init__(index)
        if role not in ("train", "internal_dev"):
            raise ValueError(role)
        self.role, self.nodes = role, nodes
        self.episodes = {d: [e for e in entries if e["length"] >= nodes + 1 and
                             research_role(e["episode_key"]) == role] for d, entries in self.episodes.items()}
        if any(not entries for entries in self.episodes.values()):
            raise ValueError("research split omitted a domain")
        self.domains = sorted(self.episodes)
        weights = np.asarray([self.index["sampling_probabilities"][d] for d in self.domains])
        self.probabilities = weights / weights.sum()

    def select(self, seed, step, rank=0, ordinal=0, domain=None):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, rank, ordinal]))
        d = domain or self.domains[int(rng.choice(len(self.domains), p=self.probabilities))]
        entries = self.episodes[d]
        # No silent invalid sample fallback into another domain.
        for _ in range(100):
            entry = entries[int(rng.integers(len(entries)))]
            payload, e = self._episode(entry)
            times = choose_times(e["length"], rng, self.nodes)
            instructions = e["instructions"][int(times[0]):int(times[-1]) + 1]
            if not instructions[0].strip() or any(s != instructions[0] for s in instructions):
                continue
            selected = []
            for k in times:
                selected.extend([max(0, int(k + off)) for off in PAST_OFFSETS] + [int(k)])
            if any(int(e["frame_spans"][k, 1]) <= 0 for k in selected):
                continue
            return entry, times
        raise RuntimeError(f"no constant-instruction timeline sampled in {d}")

    def materialize(self, entry, times, transform, statistics):
        data, e = self._episode(entry)
        if len(times) != self.nodes or np.any(np.diff(times) <= 0) or min(times) < 0 or max(times) >= e["length"] - 1:
            raise ValueError("invalid physical anchor times")
        def decode(index):
            start, size = e["frame_spans"][int(index)].tolist()
            with Image.open(io.BytesIO(data["frames"][start:start + size].numpy().tobytes())) as image:
                return transform(image.convert("RGB").resize((224, 224)))
        now, past = [], []
        for k in times:
            now.append(decode(k))
            past.append([decode(max(0, int(k + off))) for off in PAST_OFFSETS])
        maximum = max(1, int(np.diff(times).max()))
        actions = torch.zeros(self.nodes, maximum, 7)
        masks = torch.zeros(self.nodes, maximum, dtype=torch.bool)
        labels, label_masks = [], []
        stats = statistics[e["dataset"]]["action"]
        for i, k in enumerate(times):
            if i:
                seq = e["raw_actions"][int(times[i-1]):int(k)].numpy()
                actions[i, :len(seq)] = torch.from_numpy(normalize_actions(seq, stats))
                masks[i, :len(seq)] = True
            value, mask = action_window(e["raw_actions"].numpy(), int(k))
            labels.append(torch.from_numpy(normalize_actions(value, stats)))
            label_masks.append(torch.from_numpy(mask))
        policy = {
            "current": {key: torch.stack([x[key] for x in now]) for key in now[0]},
            "past": {key: torch.stack([torch.stack([x[key] for x in history]) for history in past]) for key in now[0]},
            "instructions": [e["instructions"][int(k)] for k in times],
            "times": torch.from_numpy(np.asarray(times) - times[0]),
            "executed_actions": actions, "executed_mask": masks,
        }
        supervision = {"actions": torch.stack(labels), "action_mask": torch.stack(label_masks)}
        provenance = {
            "dataset": e["dataset"], "episode_key": e["episode_key"], "source": entry["shard"],
            "source_times": [int(k) for k in times], "actual_gaps": np.diff(times).tolist(),
            "role": research_role(e["episode_key"]), "history_padding": [max(0, 15-int(k)) for k in times],
            "instruction": policy["instructions"][0],
        }
        return policy, supervision, provenance


def feature_batch(cases, indices, device):
    """Build [B,K,...] tensors; no target values accepted by Timeline.forward."""
    selected = [cases[int(i)] for i in indices]
    max_actions = max(c["policy"]["executed_actions"].shape[1] for c in selected)
    groups, nodes = len(selected), selected[0]["policy"]["times"].shape[0]
    acts = torch.zeros(groups, nodes, max_actions, 7)
    mask = torch.zeros(groups, nodes, max_actions, dtype=torch.bool)
    for i, c in enumerate(selected):
        n = c["policy"]["executed_actions"].shape[1]
        acts[i, :, :n] = c["policy"]["executed_actions"]
        mask[i, :, :n] = c["policy"]["executed_mask"]
    inputs = {k: torch.stack([c["policy"][k] for c in selected]).to(device)
              for k in ("h", "z", "base_condition", "times")}
    inputs.update(executed_actions=acts.to(device), executed_mask=mask.to(device))
    labels = {k: torch.stack([c["labels"][k] for c in selected]).to(device)
              for k in ("actions", "action_mask")}
    return inputs, labels
