"""No-GPU correctness tests for new execution utilities."""

import io
import json
import os
from pathlib import Path
import random
import tempfile
import time
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pytest
import torch

from stage3_perf.data import BatchPrefetch, SingleParseWindows, to_device
from stage3_perf.progress import Progress
from stage3_perf.runtime import group_ranks, PhaseTimer
from stage3_scale.oxe_data import OXERawWindows


class FakeDataset:
    def batch(self, count, seed, step, micro, rank, device, augment):
        assert device.type == "cpu"
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, micro, rank]))
        return {"x": torch.from_numpy(rng.normal(size=(count, 3))),
                "identity": (step, micro, rank), "aug": augment}


@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_prefetch_preserves_batches_and_global_rng(depth):
    ds = FakeDataset()
    random.seed(22); np.random.seed(22); torch.manual_seed(22)
    py = random.getstate(); np_state = np.random.get_state(); cpu = torch.get_rng_state()
    loader = BatchPrefetch(ds, 4, 42, 2, 5, 2, 3, True, depth)
    try:
        for step in range(3, 6):
            for micro in range(2):
                got = loader.get(step, micro)
                expected = ds.batch(4, 42, step, micro, 3, torch.device("cpu"), True)
                torch.testing.assert_close(got["x"], expected["x"], rtol=0, atol=0)
                assert got["identity"] == expected["identity"]
    finally:
        loader.close()
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), cpu)


def test_prefetch_propagates_error():
    ds = FakeDataset()
    ds.batch = mock.Mock(side_effect=ValueError("bad shard"))
    loader = BatchPrefetch(ds, 4, 42, 0, 2, 1, 0, True, 2)
    try:
        with pytest.raises(ValueError, match="bad shard"):
            loader.get(1, 0)
    finally:
        loader.close()


def test_prefetch_resume_same_future_batch():
    ds = FakeDataset()
    first = BatchPrefetch(ds, 4, 42, 0, 3, 1, 0, True, 2)
    first.get(1, 0)
    expected = first.get(2, 0)
    first.close()
    resumed = BatchPrefetch(ds, 4, 42, 1, 3, 1, 0, True, 2)
    try:
        torch.testing.assert_close(expected["x"], resumed.get(2, 0)["x"], rtol=0, atol=0)
    finally:
        resumed.close()


def test_group_topology_320():
    shards, replicas = group_ranks(320, 16)
    assert len(shards) == 20 and len(replicas) == 16
    assert shards[1] == list(range(16, 32))
    assert replicas[7] == list(range(7, 320, 16))
    assert sorted(x for group in shards for x in group) == list(range(320))
    assert sorted(x for group in replicas for x in group) == list(range(320))
    with pytest.raises(ValueError):
        group_ranks(33, 16)


def test_single_parse_preserves_old_index_order(tmp_path):
    content = {
        "schema": "whenami_raw_full_train_index_v1", "data_mix": "oxe_magic_soup_plus",
        "complete": True, "window_counts": {"b": 3, "a": 5},
        "episodes": [{"dataset": "b", "windows": 3}, {"dataset": "a", "windows": 2},
                     {"dataset": "a", "windows": 3}], "shard_records": [],
        "sampling_probabilities": {"a": .2, "b": .8},
    }
    p = tmp_path / "index.json"; p.write_text(json.dumps(content))
    old = OXERawWindows(p, None, None, None, {})
    new = SingleParseWindows(p, None, None, None, {})
    assert old.episodes == new.episodes and old.domains == new.domains
    assert np.array_equal(old.probabilities, new.probabilities)
    for d in old.domains:
        assert np.array_equal(old.ends[d], new.ends[d])


def test_progress_eta_plain_output_and_resume(tmp_path):
    stream = io.StringIO()
    with redirect_stdout(stream):
        p = Progress(tmp_path, 100, 20, 1280, 25, 25, True, 10)
        p.update({"step": 21, "seconds": 2., "lr": 1e-6, "new_lr": 5e-6,
                  "total": .4, "action_fm": .18, "peak_allocated_gib_max": 20.})
        p.close()
    assert "21/100" in stream.getvalue() and "ETA_train" in stream.getvalue()
    r = json.loads((tmp_path / "progress.jsonl").read_text())
    assert r["eta_training_seconds"] == 158
    assert r["eta_total_seconds"] is None
    assert r["next_checkpoint_step"] == 25


def test_to_device_and_timer():
    value = {"x": torch.ones(2), "nested": {"a": torch.zeros(1)}, "identities": ["a", "b"]}
    result = to_device(value, torch.device("cpu"))
    assert result["identities"] == ["a", "b"] and result["x"].device.type == "cpu"
    timer = PhaseTimer(True, torch.device("cpu"))
    with timer.phase("cpu"):
        time.sleep(.001)
    assert timer.global_max()["cpu"] > 0


def test_progress_early_checkpoint(tmp_path):
    p = Progress(tmp_path, 50000, 0, 1280, 2500, 2500, True, 10, first_save=100)
    p.update({"step": 1, "seconds": 2., "lr": 1e-6, "new_lr": 5e-6,
              "total": .4, "action_fm": .18, "peak_allocated_gib_max": 20.})
    p.close()
    r = json.loads((tmp_path / "progress.jsonl").read_text())
    assert r["next_checkpoint_step"] == 100


def test_resume_eta_preserves_measured_overheads(tmp_path):
    (tmp_path / "train.jsonl").write_text(json.dumps({"step": 100, "seconds": 3.}) + "\n")
    (tmp_path / "phase_events.jsonl").write_text(
        json.dumps({"phase": "save", "step": 100, "seconds": 20.}) + "\n" +
        json.dumps({"phase": "eval", "step": 100, "seconds": 40.}) + "\n")
    p = Progress(tmp_path, 50000, 100, 1280, 2500, 2500, True, 10, first_save=100)
    p.update({"step": 101, "seconds": 3., "lr": 1e-6, "new_lr": 5e-6,
              "total": .4, "action_fm": .18, "peak_allocated_gib_max": 20.})
    p.close()
    r = json.loads((tmp_path / "progress.jsonl").read_text())
    assert r["eta_total_seconds"] == r["eta_training_seconds"] + 20 * 60
