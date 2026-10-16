"""Explicit, recorded relaxation of ONE contract: Stage II from an INTERMEDIATE A+ teacher.

Agreed recipe (WHENAMI_TIMELINE_FORCE_320_SUBMIT_20261012.md §2/§5) initialises and
distils the student from the COMPLETED Stage1 step-050000. On 2026-10-16 the user
decided to start Stage II now from the committed A+ step-040000 (dev curve flat:
35k 0.04078 / 40k 0.04075 causal_last mse6). That is the ONLY change; every other
teacher check of whenami_timeline_force.loading.source_record is kept, and only the
pinned checkpoint(s) below are accepted -- no silent switch to any other step.
"""
import json
from pathlib import Path
from whenami_timeline_force import loading as base

APLUS = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/training/"
             "timeline_force_Aplus_160_resume5k_01/main/checkpoints")
APPROVED = {APLUS/"step-040000": 40000}
DEVIATION = ("Stage2 init+distillation teacher = committed INTERMEDIATE A+ step-040000 "
             "(agreed recipe: completed step-050000); user decision 2026-10-16")


def teacher_record(path, role="a50", engineering=False):
    if role != "teacher":
        return base.source_record(path, role, engineering)
    path = Path(path)
    expected = APPROVED.get(path.resolve())
    if expected is None:
        raise ValueError("teacher not in explicit approved intermediate list: "+str(path))
    meta = json.loads((path/"COMMITTED.json").read_text())
    record = base.legacy_record(path)  # COMMITTED complete, model_dcp metadata, source-drift check
    recipe = record["recipe"]
    s = recipe["settings"]
    if recipe.get("schema") != base.SCHEMA or s["stage"] != "teacher" or s["variant"] != "revise":
        raise ValueError("Stage2 requires NEW A+ revise teacher, not old A or U")
    if s["engineering"] or s["bf16_storage"] or s["steps"] != 50000 or s["global_batch"] != 320:
        raise ValueError("teacher must be the formal A+ 50k schedule")
    if record["step"] != expected or meta["step"] != expected or not meta["complete"]:
        raise ValueError("approved intermediate teacher step mismatch")
    record["intermediate_teacher"] = {"approved_step": expected, "schedule_steps": s["steps"],
                                      "deviation": DEVIATION}
    return record
