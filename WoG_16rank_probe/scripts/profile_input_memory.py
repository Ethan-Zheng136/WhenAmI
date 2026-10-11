"""CPU-only real OXE pipe benchmark, WITHOUT a VLM, optimizer, or GPU tensors.

Use monitor_node.py to bound walltime/RSS. This is intentionally not a claim
that a 16-GPU full-training job has passed.
"""

import argparse
import json
import os
from pathlib import Path
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("legacy", "bounded"), required=True)
ap.add_argument("--mix", default="oxe_magic_soup_plus")
ap.add_argument("--shuffle", type=int, default=128)
ap.add_argument("--samples", type=int, default=32)
ap.add_argument("--image-aug", action="store_true")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--statistics", default=(
    "/mnt/pfs/zhengguantian/WoG_v2/outputs/stage1/baseline_v2--image_aug/dataset_statistics.json"
))
args = ap.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
    os.environ.pop(key, None)
os.environ["WAI_BOUNDED_INPUT"] = "1" if args.mode == "bounded" else "0"
os.environ["WAI_DATASET_STATISTICS"] = args.statistics
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if args.mode == "bounded":
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch  # Pinned environment import ordering; no device allocation.
from prismatic.util.memory_diagnostics import process_memory, cgroup_memory
from prismatic.vla.datasets.rlds import make_interleaved_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from prismatic.vla.datasets.rlds.runtime_limits import pipeline_limits


start = time.monotonic()
records = []
def snapshot(phase):
    item = {
        "phase": phase, "elapsed_seconds": time.monotonic() - start,
        "process": process_memory(include_pss=True), "cgroup": cgroup_memory(),
    }
    records.append(item)
    print(json.dumps(item), flush=True)


snapshot("imports_complete")
mixture = OXE_NAMED_MIXTURES.get(args.mix, [(args.mix, 1.0)])
kwargs, weights = get_oxe_dataset_kwargs_and_weights(
    Path("/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment"),
    mixture, load_camera_views=("primary",), load_depth=False, load_proprio=False,
    load_language=True, action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
)
limits = pipeline_limits()
frame_kwargs = {
    "resize_size": (224, 224),
    "num_parallel_calls": limits.frame_parallel_calls if limits else 16,
}
if args.image_aug:
    frame_kwargs["image_augment_kwargs"] = {
        "random_resized_crop": {"scale": [0.9, 0.9], "ratio": [1.0, 1.0]},
        "random_brightness": [0.2], "random_contrast": [0.8, 1.2],
        "random_saturation": [0.8, 1.2], "random_hue": [0.05],
        "augment_order": [
            "random_resized_crop", "random_brightness", "random_contrast",
            "random_saturation", "random_hue",
        ],
    }
dataset, length, statistics = make_interleaved_dataset(
    kwargs, weights, train=True, shuffle_buffer_size=args.shuffle,
    balance_weights=True, traj_transform_threads=len(mixture), traj_read_threads=len(mixture),
    traj_transform_kwargs={
        "window_size": 16, "future_action_window_size": 15,
        "obs_offsets": [-15, -11, -7, -3, 0], "skip_unlabeled": True,
    },
    frame_transform_kwargs=frame_kwargs,
)
snapshot("dataset_graph_built")
options = dataset.options()
settings = {
    "autotune": options.autotune.enabled,
    "private_threadpool_size": options.threading.private_threadpool_size,
    "max_intra_op_parallelism": options.threading.max_intra_op_parallelism,
    "warm_start": options.experimental_warm_start,
    "ram_budget": options.autotune.ram_budget,
}
print("SETTINGS", settings, flush=True)
iterator = dataset.as_numpy_iterator()
snapshot("iterator_created")
seen = {}
for i in range(args.samples):
    batch = next(iterator)
    assert batch["action"].shape == (31, 7)
    assert batch["observation"]["image_primary"].shape == (5, 224, 224, 3)
    ts = batch["observation"]["timestep"]
    assert max(ts) == ts[-1]
    name = batch["dataset_name"].decode()
    seen[name] = seen.get(name, 0) + 1
    if i in (0, 7, 31) or i == args.samples - 1:
        snapshot(f"sample_{i + 1}")
result = {
    "mode": args.mode, "mix": args.mix, "shuffle": args.shuffle,
    "sample_count": args.samples, "dataset_count": len(statistics), "image_aug": args.image_aug,
    "sampled_datasets": seen, "settings": settings, "memory": records,
    "note": "Single process, data-only; NOT 16-rank model training",
}
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("x") as f:
    json.dump(result, f, indent=2)
