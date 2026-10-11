"""Pinned A50k model-only transfer using the unchanged original-B strict loader."""

import json
from pathlib import Path

from stage3.common import BASE, REFERENCE, STORE
from stage3_oxe_eval.prepare import sha
from whenami_timeline_b.loading import load_model_only as _load_model_only

TEACHER = STORE / "outputs/whenami_timeline_joint/training/timeline_joint_pooled_query_320_50k_01/main/checkpoints/step-050000"
TEACHER_MANIFEST_SHA256 = "50c056353588009f0d18a8b30161c6e89c819ef91e120826aacdde1bad4cdf54"


def teacher_record():
    manifest = TEACHER / "COMMITTED.json"
    digest = sha(manifest)
    if digest != TEACHER_MANIFEST_SHA256:
        raise ValueError("B50 requires the explicitly pinned A50k checkpoint")
    meta = json.loads(manifest.read_text())
    if (not meta["complete"] or meta["step"] != 50000
            or meta["recipe"]["settings"]["mode"] != "pooled_query"):
        raise ValueError("B50 requires completed pooled A50k, not timeline A or old B")
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
        "mode": "pooled_query", "scope": "Pooled-A50k model-only warm start; fresh B optimizer",
    }


def load_model_only(model, path=TEACHER, group=None, sharded=False):
    # Explicit paths remain supported for small strict-loading/restart tests.
    # Production calls pin TEACHER and validate teacher_record before loading.
    if model.timeline.mode != "pooled_query":
        raise ValueError("pooled checkpoint cannot be used with timeline architecture")
    report = _load_model_only(model, Path(path), group, sharded)
    return {**report, "architecture_mode": model.timeline.mode}
