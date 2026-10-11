"""Two independently pinned sources: B1 policy warmstart and A50k hindsight teacher."""
from pathlib import Path
import json

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from stage3.common import BASE, REFERENCE, STORE
from stage3_oxe_eval.prepare import sha

STUDENT = STORE / "outputs/whenami_timeline_b/training/timeline_b_condition_align_A10k_320_20k_01/main/checkpoints/step-020000"
TEACHER = STORE / "outputs/whenami_timeline_joint/training/timeline_joint_timeline_320_50k_01/main/checkpoints/step-050000"
PINS = {
    "student": (STUDENT, "4d9f3b3e0722a570a8f8e17744652a8239509e65f7db4f9c25156d40f92805bb", 20000),
    "teacher": (TEACHER, "f785848d08a55bc5aee4e821b3feac61b3543665e6117982b7eeb0cbbb301714", 50000),
}
NEW_PREFIXES = ("memory.", "bridge.")


def source_record():
    result = {}
    for role, (path, digest, step) in PINS.items():
        if sha(path / "COMMITTED.json") != digest:
            raise ValueError("pinned source changed: " + role)
        meta = json.loads((path / "COMMITTED.json").read_text())
        if not meta["complete"] or meta["step"] != step:
            raise ValueError("uncommitted source")
        setting = meta["recipe"]["settings"]
        if role == "teacher" and setting["mode"] != "timeline":
            raise ValueError("wrong A teacher")
        if role == "student" and (setting["variant"] != "condition_align" or meta["recipe"]["teacher"]["step"] != 10000):
            raise ValueError("wrong B1 initialization")
        for field, root in (("source", BASE), ("reference_source", REFERENCE)):
            for name, expected in meta["recipe"][field].items():
                if sha(root/name) != expected:
                    raise ValueError("source contract drift: " + name)
        if not (path/"model_dcp/.metadata").is_file() or len(list((path/"model_dcp").glob("*.distcp"))) != 16:
            raise ValueError("source model shards missing")
        result[role] = {"path": str(path), "step": step, "committed_sha256": digest}
    return result


def load_model(model, checkpoint, group=None, sharded=False, student=False):
    checkpoint = Path(checkpoint)
    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    state = get_model_state_dict(model, options=options) if sharded else model.state_dict()
    extra = {k for k in state if k.startswith(NEW_PREFIXES)} if student else set()
    core = {k: v for k, v in state.items() if k not in extra}
    metadata = dcp.FileSystemReader(checkpoint/"model_dcp").read_metadata().state_dict_metadata
    if set(metadata) != {"model."+k for k in core}:
        raise ValueError("source-to-live model key coverage differs")
    for k, v in core.items():
        if tuple(metadata["model."+k].size) != tuple(v.shape):
            raise ValueError("source shape differs: " + k)
        if not sharded and v.device.type != "cuda":
            raise ValueError("unsharded destinations must be private accelerator allocations")
    dcp.load({"model": core}, checkpoint_id=checkpoint/"model_dcp", process_group=group,
             no_dist=not torch.distributed.is_initialized())
    if sharded:
        loaded = set_model_state_dict(model, core, options=StateDictOptions(
            full_state_dict=False, cpu_offload=False, strict=False))
        if set(loaded.missing_keys) != extra or loaded.unexpected_keys:
            raise ValueError("unexpected warmstart omissions")
    return {"loaded_entries": len(core), "new_entries": sorted(extra)}
