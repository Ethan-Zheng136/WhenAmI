"""Endpoint cache contracts and episode-disjoint connector train/validation."""

import json
from pathlib import Path

import numpy as np
import torch

from stage3.common import check_endpoint, check_sources, sha256

FIELDS = {
    "past_h": (16, 1024), "future_h": (16, 1024),
    "past_z": (1, 4096), "future_z": (1, 4096),
    "past_c": (16, 32), "future_c": (16, 32),
    "past_actions": (16, 7), "future_actions": (16, 7),
}


def episode_split(datasets, episodes, seed=42, validation_fraction=0.2):
    """Split complete episodes within each dataset; no random window split."""
    datasets, episodes = np.asarray(datasets), np.asarray(episodes)
    valid = np.zeros(len(datasets), dtype=bool)
    rng = np.random.default_rng(seed)
    for name in sorted(set(datasets.tolist())):
        unique = np.unique(episodes[datasets == name])
        if len(unique) < 2:
            raise ValueError(f"{name}: at least two episodes needed for a nonleaking split")
        n = max(1, min(len(unique) - 1, round(len(unique) * validation_fraction)))
        chosen = rng.permutation(unique)[:n]
        valid |= (datasets == name) & np.isin(episodes, chosen)
    return np.flatnonzero(~valid), np.flatnonzero(valid)


def assert_episode_disjoint(datasets, episodes, train, valid):
    if set(train) & set(valid) or len(train) == 0 or len(valid) == 0:
        raise ValueError("empty/overlapping train/validation indices")
    if sorted(list(train) + list(valid)) != list(range(len(datasets))):
        raise ValueError("train/validation split must cover every sample exactly once")
    keys = list(zip(datasets.tolist(), episodes.tolist()))
    if {keys[i] for i in train} & {keys[i] for i in valid}:
        raise ValueError("an episode appears in both connector train and validation")


class EndpointCache:
    def __init__(self, path, check_provenance=True):
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        self.manifest_sha256 = sha256(manifest_path)
        if self.manifest.get("protocol") != "whenami_stage3_frozen_endpoints_v1" or not self.manifest.get("complete"):
            raise ValueError("cache is incomplete or wrong protocol")
        if check_provenance:
            for side in ("past", "future"):
                check_endpoint(self.manifest["endpoints"][side])
            check_sources(self.manifest["reference_source_sha256"])
            for name, expected in self.manifest["array_sha256"].items():
                if sha256(self.path / name) != expected:
                    raise ValueError(f"cached tensor changed after extraction: {name}")
        self.arrays = {}
        n = self.manifest["n"]
        for field, shape in FIELDS.items():
            arr = np.load(self.path / f"{field}.npy", mmap_mode="r", allow_pickle=False)
            if arr.shape != (n, *shape) or arr.dtype != np.float32:
                raise ValueError(f"bad cache tensor: {field} {arr.shape} {arr.dtype}")
            self.arrays[field] = arr
        self.datasets = np.load(self.path / "dataset.npy", allow_pickle=False)
        self.episodes = np.load(self.path / "episode.npy", allow_pickle=False)
        self.timesteps = np.load(self.path / "timestep.npy", allow_pickle=False)
        self.train_indices = np.load(self.path / "train_indices.npy", allow_pickle=False)
        self.valid_indices = np.load(self.path / "valid_indices.npy", allow_pickle=False)
        if any(len(a) != n for a in (self.datasets, self.episodes, self.timesteps)):
            raise ValueError("sample metadata length mismatch")
        assert_episode_disjoint(self.datasets, self.episodes, self.train_indices, self.valid_indices)

    def batch(self, indices, device):
        return {
            name: torch.from_numpy(np.array(arr[indices], copy=True)).to(device)
            for name, arr in self.arrays.items()
        }

    def target_statistics(self):
        # Never fit a scaler to the validation set.
        return {
            side: {
                "mean": torch.from_numpy(np.asarray(self.arrays[f"{side}_c"][self.train_indices]).mean(0)),
                "std": torch.from_numpy(np.asarray(self.arrays[f"{side}_c"][self.train_indices]).std(0)),
            } for side in ("past", "future")
        }
