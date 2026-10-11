"""Keep the exact C architecture, fixed teacher, read scale and Adam/RNG history."""
import json
import math
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from stage3.common import STORE
from stage3_oxe_eval.prepare import sha
from stage3_perf.checkpoint import SCHEMA
from whenami_b_continue.contracts import (
    START, END, SCHEDULE, TEACHER_HASHES, check_sources, file_inventory,
    continuation_lr, restore as _restore,
)

PARENTS = {
    "c_aux": ("timeline_c_future_aux_A50k_h4_320_20k_01",
              "78fe858652809457835cc7fb5087daf13d69bcc1376c0d7aa307f7469e5dc4a7", "future_aux", 0.),
    "c_read": ("timeline_c_future_read_A50k_h4_320_20k_01",
               "e00707e3d3331ecf9accfde8494a3a74af31f4eedfd3508f6bf0c5a9f503f071", "future_read", 1.),
}


def parent_path(case):
    return STORE/"outputs/whenami_timeline_c/training"/PARENTS[case][0]/"main/checkpoints/step-020000"


def source_record(case):
    run, digest, variant, read_scale = PARENTS[case]
    path = parent_path(case)
    if sha(path/"COMMITTED.json") != digest:
        raise ValueError("wrong pinned C20k parent: "+case)
    meta = json.loads((path/"COMMITTED.json").read_text())
    if (not meta["complete"] or meta["schema"] != SCHEMA or meta["step"] != START
            or meta["world_size"] != 320 or meta["shard_width"] != 16
            or meta["sharding"] != "hybrid_zero2"):
        raise ValueError("C parent topology/step mismatch")
    recipe = meta["recipe"]
    expected = {"variant": variant, "steps": START, "micro": 1, "global_batch": 320,
                "seed": 42, "lr": 1e-5, "time_lr": 1e-5, "future_gap": 4,
                "read_warmup": 1000, "alignment_weight": .1, "warmup": 1000,
                "sharding": "hybrid_zero2", "activation": "llm",
                "save_every": 2500, "eval_every": 1000, "first_check": 100, "dev_limit": 0}
    if recipe["settings"] != expected or recipe["action_condition_tokens"] != 32:
        raise ValueError("C parent architecture or recipe changed")
    check_sources(recipe)
    main = path.parent.parent
    done = json.loads((main/"COMPLETE.json").read_text())
    if not done["complete"] or done["steps"] != START:
        raise ValueError("C parent run did not complete20k")
    with (main/"train.jsonl").open("rb") as stream:
        stream.seek(max(0, (main/"train.jsonl").stat().st_size-8192))
        last = json.loads(stream.read().splitlines()[-1])
    rates = last["lr"]
    if (last["step"] != START or last["read_scale"] != read_scale or len(rates) != 2
            or any(not math.isclose(x,1e-6,rel_tol=1e-12,abs_tol=0) for x in rates)):
        raise ValueError("C parent terminal LR/read_scale mismatch")
    teacher = Path(recipe["teacher"]["path"])
    if (recipe["teacher"]["step"] != 50000
            or sha(teacher/"COMMITTED.json") != TEACHER_HASHES[50000]
            or recipe["teacher"]["committed_sha256"] != TEACHER_HASHES[50000]):
        raise ValueError("C teacher identity changed")
    tmeta = json.loads((teacher/"COMMITTED.json").read_text())
    if not tmeta["complete"] or tmeta["step"] != 50000 or tmeta["recipe"]["settings"]["mode"] != "timeline":
        raise ValueError("C teacher is not timeline A50k")
    check_sources(tmeta["recipe"])
    inventory = file_inventory(path)
    file_inventory(teacher, optimizer=False)
    return {
        "case": case, "variant": variant, "path": str(path), "step": START,
        "committed_sha256": digest, "metadata_sha256": sha(path/"model_dcp/.metadata"),
        "teacher": {"path": str(teacher), "step": 50000, "committed_sha256": TEACHER_HASHES[50000]},
        "parent_terminal_lr": rates, "inventory": inventory,
        "final_dev": str(main/"development/step-020000"),
        "future_gap": 4, "read_warmup": 1000, "read_scale": read_scale,
        "action_condition_tokens": 32,
    }


def restore(io, model, optimizer, parent, recipe, resume_from=None):
    step, proof = _restore(io, model, optimizer, parent, recipe, resume_from)
    if model.variant != parent["variant"] or model.future_gap != parent["future_gap"]:
        raise ValueError("C live architecture differs from parent")
    actual = float(model.read_scale)
    if actual != parent["read_scale"]:
        raise ValueError("C read_scale not restored; cannot restart the read-in warmup")
    return step, {**proof, "read_scale": actual, "future_gap": model.future_gap,
                  "variant": model.variant}


def load_complete_model(model, checkpoint):
    """Local smoke only; includes read_scale, unlike the original A->C loader."""
    checkpoint = Path(checkpoint)
    state = model.state_dict()
    metadata = dcp.FileSystemReader(checkpoint/"model_dcp").read_metadata().state_dict_metadata
    if set(metadata) != {"model."+k for k in state} or len(state) != 1549:
        raise ValueError("complete C20k model coverage mismatch")
    for k, value in state.items():
        if value.device.type != "cuda" or tuple(value.shape) != tuple(metadata["model."+k].size):
            raise ValueError("C smoke requires private accelerator tensors of exact shape")
    dcp.load({"model": state}, checkpoint_id=checkpoint/"model_dcp", no_dist=True)
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("nonfinite C parent model tensor")
    return {"loaded_entries": len(state), "read_scale": float(model.read_scale)}
