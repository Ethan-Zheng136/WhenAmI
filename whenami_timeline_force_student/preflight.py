"""Stage II (intermediate A+ teacher) preflight + same-topology admission gate.

Identical checks to whenami_timeline_force.preflight, except (a) teacher via gate.teacher_record
and (b) admission topology/world is the launched one (160 or 320) instead of hardcoded 320.
"""
import argparse
import json
import math
from pathlib import Path
from stage3_perf.checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from whenami_timeline_force.preflight import local_preflight
from whenami_timeline_force.runtime import fingerprints
from whenami_timeline_force_student.gate import teacher_record


def check_admission(root, variant, world, teacher):
    root = Path(root)
    probe = json.loads((root/"topology_probe/RESULT.json").read_text())
    if (not probe["complete"] or probe["world_size"] != world or probe["shard_width"] != 16 or
            not probe["next_update_bitwise_equal"] or probe["cpu_emulation"]):
        raise ValueError(f"actual {world}/16 topology admission missing")
    metas = []
    for directory, end, start in ((root/"admission1", 100, 1), (root/"admission2", 120, 101)):
        meta = json.loads((directory/"checkpoints"/f"step-{end:06d}"/"COMMITTED.json").read_text())
        if (meta["schema"] != CHECKPOINT_SCHEMA or not meta["complete"] or meta["step"] != end or
                meta["world_size"] != world or meta["shard_width"] != 16):
            raise ValueError("full model/optimizer/RNG admission checkpoint missing")
        recipe = meta["recipe"]
        s = recipe["settings"]
        if (s["stage"] != "student" or s["variant"] != variant or s["steps"] != 50000 or
                Path(s["teacher"]).resolve() != Path(teacher).resolve()):
            raise ValueError("admission phase/method/teacher mismatch")
        if recipe["warmstart"]["step"] != 40000 or Path(recipe["warmstart"]["path"]).resolve() != Path(teacher).resolve():
            raise ValueError("student warmstart is not the approved teacher")
        if recipe["source"] != fingerprints() or s["bf16_storage"] or s["engineering"]:
            raise ValueError("unverified sources or engineering precision in formal admission")
        rows = [json.loads(line) for line in (directory/"train.jsonl").read_text().splitlines()]
        if [r["step"] for r in rows] != list(range(start, end+1)):
            raise ValueError("incomplete update log")
        keys = ("total", "action_fm", "current_alignment", "cognition_alignment", "revision", "grad_norm")
        if any(not math.isfinite(r[k]) for r in rows for k in keys):
            raise ValueError("nonfinite admission")
        audits = [r["gradients"] for r in rows if r["gradients"]]
        if not audits or any(not all(math.isfinite(v) and v > 0 for v in row.values()) for row in audits):
            raise ValueError("missing full-method gradients")
        ev = json.loads((directory/"development"/f"step-{end:06d}"/"RESULT.json").read_text())
        if not ev["complete"] or len(ev["domains"]) != 27 or not ev["all_reader_conditions_from_node"]:
            raise ValueError("DEV full interface admission")
        metas.append(meta)
    if metas[0]["recipe"] != metas[1]["recipe"]:
        raise ValueError("resume recipe differs")
    res = json.loads((root/"admission2/RESUME_VERIFIED.json").read_text())
    if not res["complete"] or res["restored_step"] != 100 or res["first_update"] != 101:
        raise ValueError("new-process Adam/RNG resume missing")
    return root/"admission2/checkpoints/step-000120"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--variant", choices=("direct", "revise"), required=True)
    p.add_argument("--phase", choices=("admit", "run", "full", "resume"), required=True)
    p.add_argument("--world", type=int, choices=(160, 320), required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--resume-from", type=Path)
    a = p.parse_args()
    local_preflight()
    record = teacher_record(a.teacher, "teacher")
    if a.phase in ("admit", "run"):
        if any((a.root/n).exists() for n in ("topology_probe", "admission1", "admission2", "main")):
            raise ValueError("fresh RUN_ID required; never overwrite existing training")
    elif a.phase == "full":
        check_admission(a.root, a.variant, a.world, a.teacher)
        if (a.root/"main").exists():
            raise ValueError("main exists; use explicit PHASE=resume and new RUN_ID")
    else:
        if a.resume_from is None or (a.root/"main").exists():
            raise ValueError("explicit RESUME_FROM and fresh RUN_ID required")
        meta = json.loads((a.resume_from/"COMMITTED.json").read_text())
        s = meta["recipe"]["settings"]
        if (not meta["complete"] or meta["world_size"] not in (160, 320) or meta["shard_width"] != 16 or
                s["stage"] != "student" or s["variant"] != a.variant or s["engineering"] or
                Path(s["teacher"]).resolve() != a.teacher.resolve()):
            raise ValueError("resume needs complete formal student checkpoint with the same teacher")
    print("TIMELINE_FORCE_STUDENT_PREFLIGHT_OK", a.variant, a.phase, a.world,
          "teacher_step", record["step"], flush=True)


if __name__ == "__main__":
    main()
