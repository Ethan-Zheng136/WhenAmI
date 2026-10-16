"""Gate formal tasks on real local proof and same-topology admission."""
import argparse
import json
import math
from pathlib import Path
from stage3.common import BASE, STORE
from stage3_oxe_eval.prepare import sha
from stage3_perf.checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from whenami_unified.data import DEV, INDEX
from whenami_timeline_force.runtime import fingerprints
from whenami_timeline_force.loading import source_record


def local_preflight():
    path = BASE/"whenami_timeline_force/LOCAL_VERIFICATION.json"
    proof = json.loads(path.read_text())
    if not proof.get("local_passed") or proof["sources"] != fingerprints():
        raise ValueError("new timeline code/launcher drift or no real local verification")
    for name, digest in proof["artifacts"].items():
        if sha(STORE/name) != digest:
            raise ValueError("verification artifact drift: "+name)
    dev = json.loads((DEV/"MANIFEST.json").read_text())
    if not dev["complete"] or dev["domains"] != 27 or dev["train_overlap"]:
        raise ValueError("held-out DEV contract")
    if dev["train_index_sha256"] != sha(INDEX):
        raise ValueError("train index changed")
    return proof


def check_admission(root, stage, variant):
    root = Path(root)
    probe = json.loads((root/"topology_probe/RESULT.json").read_text())
    if (not probe["complete"] or probe["world_size"] != 320 or probe["shard_width"] != 16 or
        not probe["next_update_bitwise_equal"] or probe["cpu_emulation"]):
        raise ValueError("actual320/16 topology admission missing")
    first, second = root/"admission1", root/"admission2"
    metas = []
    for directory, end, start in ((first, 100, 1), (second, 120, 101)):
        meta = json.loads((directory/"checkpoints"/f"step-{end:06d}"/"COMMITTED.json").read_text())
        if (meta["schema"] != CHECKPOINT_SCHEMA or not meta["complete"] or meta["step"] != end or
            meta["world_size"] != 320 or meta["shard_width"] != 16):
            raise ValueError("full model/optimizer/RNG admission checkpoint missing")
        recipe = meta["recipe"]
        settings = recipe["settings"]
        if settings["stage"] != stage or settings["variant"] != variant or settings["steps"] != 50000:
            raise ValueError("admission phase/method mismatch")
        if recipe["source"] != fingerprints() or settings["bf16_storage"] or settings["engineering"]:
            raise ValueError("unverified sources or engineering precision in formal admission")
        rows = [json.loads(line) for line in (directory/"train.jsonl").read_text().splitlines()]
        if [r["step"] for r in rows] != list(range(start, end+1)):
            raise ValueError("incomplete update log")
        if any(not math.isfinite(r[k]) for r in rows for k in ("total", "action_fm", "grad_norm")):
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
    res = json.loads((second/"RESUME_VERIFIED.json").read_text())
    if not res["complete"] or res["restored_step"] != 100 or res["first_update"] != 101:
        raise ValueError("new-process Adam/RNG resume missing")
    return second/"checkpoints/step-000120"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("teacher", "student"), required=True)
    p.add_argument("--variant", choices=("direct", "revise"), required=True)
    p.add_argument("--phase", choices=("admit", "run", "full", "resume"), required=True)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--resume-from", type=Path)
    a = p.parse_args()
    local_preflight()
    if a.stage == "student":
        if a.teacher is None:
            raise ValueError("explicit A+ teacher")
        source_record(a.teacher, "teacher")
    if a.phase in ("admit", "run"):
        if any((a.root/name).exists() for name in ("topology_probe", "admission1", "admission2", "main")):
            raise ValueError("fresh RUN_ID required; never overwrite existing training")
    elif a.phase == "full":
        check_admission(a.root, a.stage, a.variant)
        if (a.root/"main").exists():
            raise ValueError("main exists; use explicit PHASE=resume and new RUN_ID")
    else:
        if a.resume_from is None:
            raise ValueError("explicit RESUME_FROM required")
        if (a.root/"main").exists():
            raise ValueError("resume output needs fresh RUN_ID")
        meta = json.loads((a.resume_from/"COMMITTED.json").read_text())
        if not meta["complete"] or meta["world_size"] != 320 or meta["shard_width"] != 16:
            raise ValueError("resume320/16 checkpoint required")
    print("TIMELINE_FORCE_PREFLIGHT_OK", a.stage, a.variant, a.phase, flush=True)


if __name__ == "__main__":
    main()
