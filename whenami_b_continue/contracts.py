"""Pinned completed parents and a low-LR continuation contract, NOT a warmstart."""
import json
import math
from pathlib import Path

from stage3.common import BASE, REFERENCE, STORE
from stage3_oxe_eval.prepare import sha
from stage3_perf.checkpoint import SCHEMA

START = 20000
END = 50000
SCHEDULE = "hold_parent_terminal_lr_no_warmup_v1"
PARENTS = {
    "old_b0": ("whenami_timeline_b", "timeline_b_action_only_A10k_320_20k_01",
               "9d8332411b82435218c18c458df0d8289cb58c7f1b884b2ccc8ef7d8c0869fa1",
               "action_only", 10000),
    "old_b1": ("whenami_timeline_b", "timeline_b_condition_align_A10k_320_20k_01",
               "4d9f3b3e0722a570a8f8e17744652a8239509e65f7db4f9c25156d40f92805bb",
               "condition_align", 10000),
    "new_b0": ("whenami_timeline_b50", "timeline_b50_action_only_A50k_320_20k_01",
               "77a155d22321e9c35783d914d2ad0fd5115397b46bd7e9e111c3d3f01867a811",
               "action_only", 50000),
    "new_b1": ("whenami_timeline_b50", "timeline_b50_condition_align_A50k_320_20k_01",
               "5dcf38d7565332bac3b6e5e752198b9a92fc5492ef4c362997ee3828d17f13c2",
               "condition_align", 50000),
}
TEACHER_HASHES = {
    10000: "f143022cd460718dc9a4121ec605ddb4fbab89fe39b71049bb39e46b4d145de3",
    50000: "f785848d08a55bc5aee4e821b3feac61b3543665e6117982b7eeb0cbbb301714",
}


def parent_path(case):
    family, run, *_ = PARENTS[case]
    return STORE/"outputs"/family/"training"/run/"main/checkpoints/step-020000"


def file_inventory(path, width=16, world=320, optimizer=True):
    path = Path(path)
    required = [path/"model_dcp/.metadata", *[
        path/"model_dcp"/f"__{i}_0.distcp" for i in range(width)]]
    if optimizer:
        required += [path/f"optimizer-{i:05d}.pt" for i in range(width)]
        required += [path/f"rng-{i:05d}.pt" for i in range(world)]
    if any(not p.is_file() or p.stat().st_size == 0 for p in required):
        raise ValueError("incomplete model/Adam/RNG inventory: "+str(path))
    return {"model_shards": width, "optimizer_shards": width if optimizer else None,
            "rng_shards": world if optimizer else None}


def check_sources(recipe):
    for field, root in (("source", BASE), ("reference_source", REFERENCE)):
        for relative, digest in recipe[field].items():
            if sha(root/relative) != digest:
                raise ValueError("parent source contract drift: "+relative)


def source_record(case):
    family, run, digest, variant, teacher_step = PARENTS[case]
    path = parent_path(case)
    if sha(path/"COMMITTED.json") != digest:
        raise ValueError("wrong pinned B20k parent: "+case)
    meta = json.loads((path/"COMMITTED.json").read_text())
    if (not meta["complete"] or meta["schema"] != SCHEMA or meta["step"] != START
            or meta["world_size"] != 320 or meta["shard_width"] != 16
            or meta["sharding"] != "hybrid_zero2"):
        raise ValueError("parent topology/step mismatch")
    recipe = meta["recipe"]
    expected = {"variant": variant, "steps": START, "micro": 1, "global_batch": 320,
                "seed": 42, "lr": 1e-5, "aux_lr": 1e-4, "alignment_weight": .1,
                "warmup": 1000, "sharding": "hybrid_zero2", "activation": "llm",
                "save_every": 2500, "eval_every": 1000, "first_check": 100, "dev_limit": 0}
    if recipe["settings"] != expected:
        raise ValueError("unexpected parent recipe; do not silently reinterpret it")
    check_sources(recipe)
    main = path.parent.parent
    done = json.loads((main/"COMPLETE.json").read_text())
    if not done["complete"] or done["steps"] != START:
        raise ValueError("parent run did not complete20k")
    with (main/"train.jsonl").open("rb") as stream:
        stream.seek(max(0, (main/"train.jsonl").stat().st_size-8192))
        last = json.loads(stream.read().splitlines()[-1])
    if last["step"] != START or len(last["lr"]) != 2:
        raise ValueError("parent terminal schedule missing")
    rates = last["lr"]
    if any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=0)
           for a,b in zip(rates,(1e-6,1e-5))):
        raise ValueError("parent terminal learning rates unexpected")
    teacher = Path(recipe["teacher"]["path"])
    if (recipe["teacher"]["step"] != teacher_step
            or sha(teacher/"COMMITTED.json") != TEACHER_HASHES[teacher_step]
            or recipe["teacher"]["committed_sha256"] != TEACHER_HASHES[teacher_step]):
        raise ValueError("teacher identity was changed")
    tmeta = json.loads((teacher/"COMMITTED.json").read_text())
    if not tmeta["complete"] or tmeta["step"] != teacher_step or tmeta["recipe"]["settings"]["mode"] != "timeline":
        raise ValueError("wrong teacher checkpoint")
    check_sources(tmeta["recipe"])
    inventory = file_inventory(path)
    file_inventory(teacher, optimizer=False)
    return {
        "case": case, "variant": variant, "path": str(path), "step": START,
        "committed_sha256": digest, "metadata_sha256": sha(path/"model_dcp/.metadata"),
        "teacher": {"path": str(teacher), "step": teacher_step,
                    "committed_sha256": TEACHER_HASHES[teacher_step]},
        "parent_terminal_lr": rates, "inventory": inventory,
        "final_dev": str(main/"development/step-020000"),
    }


def continuation_lr(step, terminal_lr):
    if not START <= step <= END or terminal_lr <= 0 or not math.isfinite(terminal_lr):
        raise ValueError("invalid continuation step/rate")
    # Intentional minimal extension: retain exact last optimizer rates, with no
    # new warmup, no backward reinterpretation of the original 20k cosine.
    return terminal_lr


def restore(io, model, optimizer, parent, recipe, resume_from=None):
    """Validate under ORIGINAL recipe, then write new continuation provenance.

    CheckpointIO itself is unchanged. Later continuation resumes require the new
    recipe exactly; only the first 20k parent load validates the parent recipe.
    """
    path = Path(resume_from) if resume_from else Path(parent["path"])
    meta = json.loads((path/"COMMITTED.json").read_text())
    if resume_from:
        if meta["recipe"] != recipe or not START < meta["step"] < END:
            raise ValueError("wrong continuation recipe/step; cannot reset optimizer")
        expected_recipe = recipe
    else:
        if sha(path/"COMMITTED.json") != parent["committed_sha256"] or meta["step"] != START:
            raise ValueError("wrong original parent")
        expected_recipe = meta["recipe"]
    step = io.load(path, model, optimizer, expected_recipe)
    rates = [g["lr"] for g in optimizer.param_groups]
    if rates != parent["parent_terminal_lr"]:
        raise ValueError("Adam restored but optimizer LR differs from pinned continuation schedule")
    steps = [float(s["step"]) for s in optimizer.state.values() if "step" in s]
    if not steps or max(steps) != step or any(not math.isfinite(x) or x <= 0 or x > step for x in steps):
        raise ValueError("Adam step counters missing/reset/inconsistent")
    return step, {
        "complete": True, "checkpoint": str(path), "restored_step": step,
        "original_parent_recipe_used": resume_from is None,
        "model_Adam_rank_RNG_restored": True, "lr": rates,
        "local_Adam_states": len(steps), "local_Adam_step_min": min(steps),
        "local_Adam_step_max": max(steps), "parent_manifest_sha256": parent["committed_sha256"],
    }
