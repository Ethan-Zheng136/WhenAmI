"""Read existing prepared OXE train episodes, expose only legal multi-anchor prefixes."""

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from whenami_revision.contracts import (
    EvidencePrefix, canonical_anchors, current_action_supervision,
)


class RevisionEpisodes:
    def __init__(self, index):
        self.index_path = Path(index)
        self.index = json.loads(self.index_path.read_text())
        if (not self.index.get("complete") or self.index.get("data_mix") != "oxe_magic_soup_plus" or
                self.index.get("domain_count") != 27 or
                self.index.get("scope") != "full_source_train_filtered"):
            raise ValueError("require complete train-only OXE27 preparation, not dev/sealed data")
        self.episodes = {}
        for entry in self.index["episodes"]:
            self.episodes.setdefault(entry["dataset"], []).append(entry)
        self.records = {r["file"]: r for r in self.index["shard_records"]}
        self.active = None

    def _episode(self, entry):
        path = entry["shard"]
        if path not in self.records or Path(path).stat().st_size != self.records[path]["bytes"]:
            raise ValueError("shard missing/modified")
        if self.active is None or self.active[0] != path:
            self.active = (path, torch.load(path, weights_only=True, mmap=True, map_location="cpu"))
        data = self.active[1]
        episode = data["episodes"][entry["episode_index"]]
        if (episode["dataset"], episode["episode_key"]) != (entry["dataset"], entry["episode_key"]):
            raise ValueError("wrong episode lookup")
        return data, episode

    def prefix(self, entry, anchors, cutoff, same_instruction=True):
        data, e = self._episode(entry)
        visible = canonical_anchors(anchors, cutoff, e["length"])
        if same_instruction:
            # Reject rather than accidentally infer a task transition from a later instruction.
            texts = e["instructions"][visible[0]:cutoff + 1]
            if any(x != texts[0] for x in texts):
                raise ValueError("instruction changed inside this research window")
        images, intervals, action_times = [], [], []
        for i, k in enumerate(visible):
            start, size = e["frame_spans"][k].tolist()
            images.append(data["frames"][start:start + size].numpy().tobytes())
            lo = visible[i - 1] if i else k
            intervals.append(e["raw_actions"][lo:k].numpy().copy())
            action_times.append(tuple(range(lo, k)))
        return EvidencePrefix(
            e["dataset"], e["episode_key"], visible, cutoff, tuple(images),
            tuple(e["instructions"][k] for k in visible), tuple(intervals), tuple(action_times),
        ).validate()

    def action_labels(self, entry, prefix):
        prefix.validate()
        if (entry["dataset"], entry["episode_key"]) != (prefix.dataset, prefix.episode_key):
            raise ValueError("labels from a different episode")
        _, e = self._episode(entry)
        return current_action_supervision(e["raw_actions"].numpy(), prefix.cutoff, prefix.cutoff)


def tensorize(prefix, transform, action_statistics=None):
    """No access to the full episode, its length, future instructions or action labels.

    transform must operate on ONE image, not encode future video before slicing.
    Output is a single-sample feature-view skeleton; production VLM integration
    remains separate from this first data-contract implementation.
    """
    prefix.validate()
    images = []
    for encoded in prefix.images:
        with Image.open(io.BytesIO(encoded)) as image:
            images.append(transform(image.convert("RGB")))
    if not all(torch.is_tensor(x) for x in images):
        raise ValueError("prototype transform must return a tensor")
    lengths = [len(a) for a in prefix.interval_actions]
    width = max(1, max(lengths))
    actions = torch.zeros(len(images), width, 7)
    mask = torch.zeros(len(images), width, dtype=torch.bool)
    for i, value in enumerate(prefix.interval_actions):
        if action_statistics is not None:
            from evaluation.past_protocol import normalize_actions
            value = normalize_actions(value, action_statistics)
        actions[i, :lengths[i]] = torch.as_tensor(value)
        mask[i, :lengths[i]] = True
    return {
        "observations": torch.stack(images)[None],
        "times": torch.as_tensor(prefix.relative_times)[None],
        "executed_actions": actions[None],
        "executed_mask": mask[None],
    }
