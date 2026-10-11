"""Provenance, safe I/O and reference-code paths; no heavyweight model imports."""

import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
REFERENCE = BASE / "WoG_16rank_probe"
STORE = Path("/mnt/pfs/zhengguantian/WoG_v2")
DEFAULT_PAST = STORE / (
    "outputs/stage2_16rank_probe/past_s2_16rank_20node_from5k_01--image_aug/"
    "checkpoints/step-015000-epoch-01-loss=0.2112.pt"
)
DEFAULT_FUTURE = STORE / (
    "outputs/stage1/baseline_v2--image_aug/checkpoints/step-020000-epoch-02-loss=0.1076.pt"
)
SOURCE_FILES = [
    "vla/vla.py", "vla/qformer.py", "vla/load.py", "vla/past.py",
    "action_model/action_model.py", "action_model/models.py",
    "prismatic/vla/datasets/rlds/oxe/transforms.py",
    "prismatic/vla/datasets/rlds/utils/data_utils.py",
    "prismatic/vla/datasets/rlds/dataset.py",
    "prismatic/vla/datasets/rlds/runtime_limits.py",
]


def reference_imports():
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def endpoint_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    config = path.parents[1] / "config.json"
    statistics = path.parents[1] / "dataset_statistics.json"
    return {
        "path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "config": json.loads(config.read_text()), "config_sha256": sha256(config),
        "statistics": json.loads(statistics.read_text()), "statistics_sha256": sha256(statistics),
    }


def check_endpoint(record):
    path = Path(record["path"])
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
        raise ValueError(f"Frozen endpoint checkpoint changed: {path}")
    for name in ("config", "statistics"):
        filename = "config.json" if name == "config" else "dataset_statistics.json"
        if sha256(path.parents[1] / filename) != record[f"{name}_sha256"]:
            raise ValueError(f"Endpoint {filename} changed: {path}")


def source_fingerprint():
    return {name: sha256(REFERENCE / name) for name in SOURCE_FILES}


def check_sources(fingerprint):
    for name, expected in fingerprint.items():
        if sha256(REFERENCE / name) != expected:
            raise ValueError(f"Reference source changed since extraction: {name}")
