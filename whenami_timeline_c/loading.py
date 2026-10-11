"""Fixed A50k teacher and A50k student initialization; only new read_scale allowed."""
import json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict, StateDictOptions,
)

from stage3.common import STORE, BASE, REFERENCE
from stage3_oxe_eval.prepare import sha

TEACHER = STORE / "outputs/whenami_timeline_joint/training/timeline_joint_timeline_320_50k_01/main/checkpoints/step-050000"


def teacher_record():
    meta = json.loads((TEACHER / "COMMITTED.json").read_text())
    if not meta["complete"] or meta["step"] != 50000 or meta["recipe"]["settings"]["mode"] != "timeline":
        raise ValueError("C requires the fixed completed timeline A50k")
    for field, root in (("source", BASE), ("reference_source", REFERENCE)):
        for name, value in meta["recipe"][field].items():
            if sha(root / name) != value:
                raise ValueError("A source contract changed")
    return {"path": str(TEACHER), "step": 50000,
            "committed_sha256": sha(TEACHER / "COMMITTED.json"),
            "scope": "Model-only A50k warm start, not old B20k continuation."}


def load_model(model, checkpoint=TEACHER, group=None, sharded=False):
    checkpoint = Path(checkpoint)
    metadata = dcp.FileSystemReader(checkpoint / "model_dcp").read_metadata().state_dict_metadata
    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    state = get_model_state_dict(model, options=options) if sharded else model.state_dict()
    extra = {k for k in state if k == "read_scale"}
    core = {k: v for k, v in state.items() if k not in extra}
    if set(metadata) != {"model." + k for k in core}:
        raise ValueError("A50k/C state keys mismatch")
    for key, value in core.items():
        if tuple(value.shape) != tuple(metadata["model." + key].size):
            raise ValueError("A50k/C shape mismatch")
        if not sharded and value.device.type != "cuda":
            raise ValueError("unsharded destinations must be private accelerator tensors")
    dcp.load({"model": core}, checkpoint_id=checkpoint / "model_dcp",
             process_group=group, no_dist=not torch.distributed.is_initialized())
    if sharded:
        result = set_model_state_dict(model, core, options=StateDictOptions(
            full_state_dict=False, cpu_offload=False, strict=False))
        if set(result.missing_keys) != extra or result.unexpected_keys:
            raise ValueError("unexpected C warm-start omissions")
    return {"A_state_entries": len(core), "new_entries": sorted(extra)}


def build_student(variant="future_aux", future_gap=4, seed=42):
    from whenami_timeline_joint.model import load_joint
    from whenami_timeline_c.model import TimelineStudent
    base, visual, transform, tokenizer, prompt = load_joint("timeline", seed)
    student = TimelineStudent(base.vlm, base.local_encoder, base.reader, variant, future_gap)
    student.timeline.load_state_dict(base.timeline.state_dict(), strict=True)
    return student, visual, transform, tokenizer, prompt
