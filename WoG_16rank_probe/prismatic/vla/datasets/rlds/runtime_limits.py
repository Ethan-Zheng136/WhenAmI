"""Opt-in input-pipeline limits for the isolated 16-rank experiment.

This module is NOT installed in the production WoG tree or shared site-packages.
An autotune RAM budget is NOT a hard process-memory cap; explicitly sized
shuffle buffers, decoded trajectories, model loading and allocators are separate.
"""

import json
import os
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineLimits:
    private_threads: int = 8
    frame_parallel_calls: int = 2
    autotune: bool = False
    warm_start: bool = False
    ram_budget_gib: int = 1
    cpu_budget: int = 8

    def __post_init__(self):
        for name in ("private_threads", "frame_parallel_calls", "ram_budget_gib", "cpu_budget"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive; zero would silently restore automatic tuning")

    def options(self):
        import tensorflow as tf
        options = tf.data.Options()
        options.threading.private_threadpool_size = self.private_threads
        options.threading.max_intra_op_parallelism = 1
        options.autotune.enabled = self.autotune
        options.autotune.cpu_budget = self.cpu_budget
        options.autotune.ram_budget = self.ram_budget_gib * 2**30
        options.experimental_warm_start = self.warm_start
        options.experimental_optimization.inject_prefetch = False
        options.deterministic = False  # Preserve the legacy nondeterministic sampling policy.
        return options

    def apply(self, dataset):
        return dataset.with_options(self.options())

    def as_dict(self):
        return asdict(self)


def pipeline_limits():
    if os.environ.get("WAI_BOUNDED_INPUT", "0") != "1":
        return None
    return PipelineLimits(
        private_threads=int(os.environ.get("WAI_TF_PRIVATE_THREADS", "8")),
        frame_parallel_calls=int(os.environ.get("WAI_FRAME_PARALLEL_CALLS", "2")),
        cpu_budget=int(os.environ.get("WAI_TF_CPU_BUDGET", "8")),
    )


def read_rlds(builder, split="train", shuffle=True, num_parallel_reads=None):
    """Bound buffers at their SOURCE, not just at the terminal mixed dataset.

    Installed dlimp leaves TFDS interleave_block_length=16 and decode=AUTOTUNE.
    RLDS TFRecords hold entire episodes, so these are episode-sized buffers.
    Reuse dlimp's exact skip-decoding and metadata mapping; change only read
    scheduling/buffering. No public dependency files or dataset bytes are edited.
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import dlimp as dl
    limits = pipeline_limits()
    reads = tf.data.AUTOTUNE if num_parallel_reads is None else num_parallel_reads
    if limits is None:
        return dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle, num_parallel_reads=reads)
    # These two helpers are the exact ones in the pinned dlimp DLataset.from_rlds.
    from dlimp.dataset import _wrap, _broadcast_metadata_rlds
    read_config = tfds.ReadConfig(
        options=limits.options(),
        skip_prefetch=True,
        try_autocache=False,
        num_parallel_calls_for_interleave_files=1,
        interleave_cycle_length=1,
        interleave_block_length=1,
        num_parallel_calls_for_decode=1,
        override_buffer_size=1024 * 1024,
    )
    dataset = _wrap(builder.as_dataset, False)(
        split=split, shuffle_files=shuffle,
        decoders={"steps": tfds.decode.SkipDecoding()}, read_config=read_config,
    )
    return limits.apply(dataset.enumerate().traj_map(_broadcast_metadata_rlds, num_parallel_calls=1))


def pinned_statistics(dataset_names):
    """Avoid every rank recomputing corpus statistics when a cache hash changes.

    Caller must supply statistics from the matching data/standardization recipe.
    Missing names fail closed: never quietly scan a multi-TB dataset on 640 ranks.
    """
    path = os.environ.get("WAI_DATASET_STATISTICS")
    if not path:
        return None
    stats = json.loads(Path(path).read_text())
    missing = sorted(set(dataset_names) - set(stats))
    if missing:
        raise ValueError(f"Pinned statistics missing datasets: {missing}")
    for name in dataset_names:
        item = stats[name]
        if int(item["num_transitions"]) <= 0 or int(item["num_trajectories"]) <= 0:
            raise ValueError(f"Invalid dataset size for {name}")
        action = item["action"]
        if len(action["q01"]) != 7 or len(action["q99"]) != 7:
            raise ValueError(f"Expected 7D action statistics for {name}")
    return stats


def runtime_summary():
    limits = pipeline_limits()
    path = os.environ.get("WAI_DATASET_STATISTICS")
    return {
        "bounded": limits is not None,
        "limits": limits.as_dict() if limits else None,
        "source_read_limits": ({
            "interleave_cycle_length": 1, "interleave_block_length": 1,
            "num_parallel_calls_for_decode": 1, "override_buffer_size": 1024 * 1024,
            "skip_prefetch": True, "try_autocache": False,
        } if limits else None),
        "statistics_path": path,
        "statistics_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None,
    }
