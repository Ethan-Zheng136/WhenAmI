"""Pinned warmstart with strict NEW-module allowlist and stage identity checks."""
import json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions
from whenami_unified.loading import A50, checkpoint_record as legacy_record

SCHEMA = "whenami_action_shaped_timeline_v1"


def source_record(path, role="a50", engineering=False):
    path = Path(path)
    record = legacy_record(path)
    if role == "a50":
        if path.resolve() != A50.resolve():
            raise ValueError("Stage1 must warmstart audited A50")
    elif role == "teacher":
        recipe = record["recipe"]
        if recipe.get("schema") != SCHEMA or recipe["settings"]["stage"] != "teacher":
            raise ValueError("Stage2 requires NEW A+ timeline teacher, not old A or U")
        if not engineering and record["step"] != recipe["settings"]["steps"]:
            raise ValueError("teacher not completed; no silent switch to intermediate or engineering checkpoint")
        if not engineering and recipe["settings"]["engineering"]:
            raise ValueError("engineering teacher is not a formal trained A+")
    else:
        raise ValueError("source role")
    return record


def load_weights(model, path, sharded=False, group=None):
    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    state = get_model_state_dict(model, options=options) if sharded else model.state_dict()
    metadata = dcp.FileSystemReader(Path(path)/"model_dcp").read_metadata().state_dict_metadata
    keys = {k.removeprefix("model.") for k in metadata}
    new = set(state)-keys
    allowed = ("node_cognition.", "executed_adapter.", "forecast_head.")
    if keys-set(state) or any(not k.startswith(allowed) for k in new):
        raise ValueError("source coverage mismatch")
    if Path(path).resolve() != A50.resolve() and any(k.startswith("node_cognition.") for k in new):
        raise ValueError("new teacher must carry trained node cognition")
    core = {k: v for k, v in state.items() if k in keys}
    for k, value in core.items():
        if tuple(value.shape) != tuple(metadata["model."+k].size):
            raise ValueError("source shape: "+k)
        if not sharded and value.device.type != "cuda":
            raise ValueError("private accelerator destination required")
    dcp.load({"model": core}, checkpoint_id=Path(path)/"model_dcp",
             process_group=group, no_dist=not torch.distributed.is_initialized())
    if sharded:
        mismatch = set_model_state_dict(model, core, options=StateDictOptions(
            full_state_dict=False, cpu_offload=False, strict=False))
        if set(mismatch.missing_keys) != new or mismatch.unexpected_keys:
            raise ValueError("FSDP post-load omissions")
    return {"loaded_entries": len(core), "new_entries": sorted(new),
            "legacy_numeric_forward_parity_claimed": False}
