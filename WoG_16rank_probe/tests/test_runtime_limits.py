import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from prismatic.vla.datasets.rlds.runtime_limits import PipelineLimits, pipeline_limits, pinned_statistics
from prismatic.util.memory_diagnostics import cgroup_memory, process_memory, emit_memory


def test_default_does_not_enable_runtime_profile(monkeypatch):
    monkeypatch.delenv("WAI_BOUNDED_INPUT", raising=False)
    assert pipeline_limits() is None


def test_tf_options_are_explicit_and_output_values_preserved():
    source = tf.data.Dataset.range(20).map(lambda x: x * 2, num_parallel_calls=4)
    bounded = PipelineLimits(private_threads=2, cpu_budget=2).apply(source)
    opts = bounded.options()
    assert opts.autotune.enabled is False
    assert opts.experimental_warm_start is False
    assert opts.threading.private_threadpool_size == 2
    assert opts.threading.max_intra_op_parallelism == 1
    assert opts.experimental_optimization.inject_prefetch is False
    # Legacy dlimp is nondeterministic too; scheduling may change order, not
    # the values/cardinality. Do not incorrectly require a fixed arrival order.
    assert sorted(bounded.as_numpy_iterator()) == list(range(0, 40, 2))


def test_invalid_limits_fail_closed(monkeypatch):
    monkeypatch.setenv("WAI_BOUNDED_INPUT", "1")
    monkeypatch.setenv("WAI_TF_PRIVATE_THREADS", "0")
    with pytest.raises(ValueError):
        pipeline_limits()


def test_read_limits_apply_to_episode_source_and_preserve_metadata(monkeypatch):
    from prismatic.vla.datasets.rlds.runtime_limits import read_rlds
    monkeypatch.setenv("WAI_BOUNDED_INPUT", "1")
    calls = []
    class Builder:
        def as_dataset(self, **kwargs):
            calls.append(kwargs)
            return tf.data.Dataset.from_tensors({
                "steps": {"action": tf.reshape(tf.range(14), (2, 7)),
                          "observation": {"image": tf.constant(["jpeg-a", "jpeg-b"])}},
                "episode_metadata": {"success": tf.constant(True)},
            })
    dataset = read_rlds(Builder(), num_parallel_reads=16)
    cfg = calls[0]["read_config"]
    assert cfg.interleave_block_length == cfg.interleave_cycle_length == 1
    assert cfg.num_parallel_calls_for_decode == 1
    assert cfg.try_autocache is False and cfg.skip_prefetch is True
    batch = next(dataset.as_numpy_iterator())
    np.testing.assert_equal(batch["action"], np.arange(14).reshape(2, 7))
    assert batch["observation"]["image"].tolist() == [b"jpeg-a", b"jpeg-b"]
    assert dataset.is_flattened is False


def test_pinned_stats_require_all_requested_datasets(monkeypatch, tmp_path):
    stats = {
        "a": {"action": {"q01": [-1] * 7, "q99": [1] * 7},
              "num_transitions": 40, "num_trajectories": 2}
    }
    path = tmp_path / "statistics.json"
    path.write_text(json.dumps(stats))
    monkeypatch.setenv("WAI_DATASET_STATISTICS", str(path))
    assert pinned_statistics(["a"]) == stats
    with pytest.raises(ValueError, match="missing datasets"):
        pinned_statistics(["a", "b"])


def test_memory_logging_is_opt_in_and_reads_local_cgroup(monkeypatch, tmp_path):
    monkeypatch.setenv("WAI_MEMORY_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("RANK", "7")
    emit_memory("test")
    record = json.loads((tmp_path / "rank-7.jsonl").read_text())
    assert record["phase"] == "test" and record["rank"] == "7"
    assert record["process"]["VmRSS_kib"] > 0
    assert process_memory()["threads"] >= 1
    assert isinstance(cgroup_memory(), dict)


def test_node_monitor_preserves_child_failure_and_labels_safety_stop(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/monitor_node.py"
    output = tmp_path / "failure.jsonl"
    result = subprocess.run([
        sys.executable, str(script), "--out", str(output), "--interval", "0.02", "--",
        sys.executable, "-c", "raise SystemExit(7)",
    ], timeout=15)
    assert result.returncode == 7
    final = json.loads(output.read_text().splitlines()[-1])
    assert final["returncode"] == 7 and final["stop_reason"] is None
    output2 = tmp_path / "guard.jsonl"
    result = subprocess.run([
        sys.executable, str(script), "--out", str(output2), "--interval", "0.02",
        "--timeout", "0.1", "--", sys.executable, "-c", "import time; time.sleep(30)",
    ], timeout=15)
    assert result.returncode == 124
    final = json.loads(output2.read_text().splitlines()[-1])
    assert final["stop_reason"] == "profiling_wall_time_limit"


def test_pinned_build_skips_statistics_graph_and_keeps_mixture(monkeypatch):
    """Test actual mixture factory with tiny TF records, no OXE downloads."""
    import prismatic.vla.datasets.rlds.dataset as module
    import dlimp as dl

    stats = {
        name: {"action": {"q01": [-1] * 7, "q99": [1] * 7},
               "num_transitions": count, "num_trajectories": 1}
        for name, count in [("a", 10), ("b", 20)]
    }
    monkeypatch.setattr(module, "pinned_statistics", lambda names: stats)
    monkeypatch.setattr(module, "pipeline_limits", lambda: PipelineLimits(private_threads=2))
    calls = []
    def make(name, **kwargs):
        calls.append(name)
        assert isinstance(kwargs["dataset_statistics"]["action"]["q01"], np.ndarray)
        source = tf.data.Dataset.from_tensors({
            "action": tf.zeros((2, 7)),
            "observation": {"timestep": tf.constant([0, 1])},
        })
        source.__class__ = type("DLataset", (dl.DLataset, type(source)), dl.DLataset.__dict__.copy())
        source.is_flattened = False
        return source, kwargs["dataset_statistics"]
    monkeypatch.setattr(module, "make_dataset_from_rlds", make)
    monkeypatch.setattr(module, "apply_trajectory_transforms", lambda dataset, **kw: dataset)
    monkeypatch.setattr(module, "apply_frame_transforms", lambda dataset, **kw: dataset)
    dataset, length, got = module.make_interleaved_dataset(
        [{"name": "a"}, {"name": "b"}], [1.0, 0.5], train=True, shuffle_buffer_size=2,
        balance_weights=True, traj_transform_threads=2, traj_read_threads=2,
        traj_transform_kwargs={}, frame_transform_kwargs={},
    )
    assert calls == ["a", "b"]  # No extra discarded graph for statistics.
    assert set(got) == {"a", "b"}
    np.testing.assert_allclose(dataset.sample_weights, [0.5, 0.5])
    assert length == 20
    assert next(dataset.as_numpy_iterator())["action"].shape == (7,)
