"""Pinned A50k model-only transfer using the unchanged original-B strict loader."""

import json
from pathlib import Path

from stage3.common import BASE, REFERENCE, STORE
from stage3_oxe_eval.prepare import sha
from whenami_timeline_b.loading import load_model_only as _load_model_only

TEACHER = STORE / "outputs/whenami_timeline_joint/training/timeline_joint_timeline_320_50k_01/main/checkpoints/step-050000"
TEACHER_MANIFEST_SHA256 = "f785848d08a55bc5aee4e821b3feac61b3543665e6117982b7eeb0cbbb301714"


def teacher_record():
    manifest = TEACHER / "COMMITTED.json"
    digest = sha(manifest)
    if digest != TEACHER_MANIFEST_SHA256:
        raise ValueError("B50 requires the explicitly pinned A50k checkpoint")
    meta = json.loads(manifest.read_text())
    if (not meta["complete"] or meta["step"] != 50000
            or meta["recipe"]["settings"]["mode"] != "timeline"):
        raise ValueError("B50 requires completed timeline A50k, not pooled A or old B")
    for field, root in (("source", BASE), ("reference_source", REFERENCE)):
        for name, value in meta["recipe"][field].items():
            if sha(root / name) != value:
                raise ValueError("A50k source contract drift: " + name)
    model_dir = TEACHER / "model_dcp"
    if (not (model_dir / ".metadata").is_file()
            or len(list(model_dir.glob("*.distcp"))) != meta["shard_width"]):
        raise ValueError("A50k model checkpoint shards missing")
    return {
        "path": str(TEACHER), "step": 50000, "committed_sha256": digest,
        "source_recipe": meta["recipe"],
        "scope": "A50k model-only warm start; fresh B optimizer, not B20k resume",
    }


def load_model_only(model, path=TEACHER, group=None, sharded=False):
    # Explicit paths remain supported for small strict-loading/restart tests.
    # Production calls pin TEACHER and validate teacher_record before loading.
    return _load_model_only(model, Path(path), group, sharded)
