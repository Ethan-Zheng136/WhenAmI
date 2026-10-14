"""Warm start: Stage I from audited A50k; Stage II from a completed present-token Stage I.

New updaters (present.bwd, present.imaginer.updater) are INITIALISED from the source
`timeline.*` tensors (reuse A50k updater design/init); everything else new keeps its init.
"""
import json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
from whenami_unified.loading import A50, checkpoint_record

SCHEMA = "whenami_present_token_v1"
OPT = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=False)


def source_record(path, role, engineering=False):
    path = Path(path)
    if role == "a50":
        if path.resolve() != A50.resolve():
            raise ValueError("Stage I must warm-start the audited A50k")
        return checkpoint_record(path)
    meta = json.loads((path/"COMMITTED.json").read_text())
    recipe = meta["recipe"]
    if recipe.get("schema") != SCHEMA or recipe["settings"]["stage"] != "teacher" or not meta["complete"]:
        raise ValueError("Stage II needs a committed present-token Stage I checkpoint")
    if not engineering and (meta["step"] != recipe["settings"]["steps"] or recipe["settings"]["engineering"]):
        raise ValueError("formal Stage II needs the COMPLETED formal Stage I")
    return {"path": str(path), "step": meta["step"], "sha256": None, "recipe": recipe}


def new_prefixes(student, from_a50):
    if from_a50:
        return ("node_cognition.", "present.", "executed_adapter.")
    return ("present.imaginer.", "executed_adapter.") if student else ()


COPY_FROM_TIMELINE = ("present.bwd.", "present.imaginer.updater.")


def load_weights(model, path, student, from_a50, group=None):
    state = get_model_state_dict(model, options=OPT)
    meta = dcp.FileSystemReader(Path(path)/"model_dcp").read_metadata().state_dict_metadata
    keys = {k.removeprefix("model.") for k in meta}
    new = set(state)-keys
    allowed = new_prefixes(student, from_a50)
    if any(not k.startswith(allowed) for k in new):
        raise ValueError("unexpected missing keys: "+str(sorted(k for k in new if not k.startswith(allowed))[:5]))
    extra = keys-set(state)
    if extra:
        raise ValueError("source has keys the model lacks: "+str(sorted(extra)[:5]))
    core = {k: v for k, v in state.items() if k in keys}
    for k, v in core.items():
        if tuple(v.shape) != tuple(meta["model."+k].size):
            raise ValueError("shape "+k)
    nd = not torch.distributed.is_initialized()
    dcp.load({"model": core}, checkpoint_id=Path(path)/"model_dcp", process_group=group, no_dist=nd)
    copied = {}
    for prefix in COPY_FROM_TIMELINE:
        dest = {k: v for k, v in state.items() if k.startswith(prefix) and k in new}
        if not dest:
            continue
        src = {"timeline."+k[len(prefix):]: v for k, v in dest.items()}
        if set(src)-keys:
            raise ValueError("timeline source missing for "+prefix)
        dcp.load({"model": src}, checkpoint_id=Path(path)/"model_dcp", process_group=group, no_dist=nd)
        core.update(dest)
        copied[prefix] = len(dest)
    set_model_state_dict(model, core, options=OPT)
    fresh = sorted(k for k in new if not k.startswith(tuple(copied)))
    return {"loaded_entries": len(core), "copied_from_timeline": copied, "fresh_init": fresh}


def load_frozen_teacher(model, path, group=None):
    state = get_model_state_dict(model, options=OPT)
    meta = dcp.FileSystemReader(Path(path)/"model_dcp").read_metadata().state_dict_metadata
    keys = {k.removeprefix("model.") for k in meta}
    if set(state) != keys:
        raise ValueError("teacher coverage mismatch")
    dcp.load({"model": state}, checkpoint_id=Path(path)/"model_dcp", process_group=group,
             no_dist=not torch.distributed.is_initialized())
    set_model_state_dict(model, state, options=OPT)
    return {"loaded_entries": len(state)}
