"""Topology-agnostic preflight: sources, data contracts, warm-start/teacher, resume shards."""
import argparse, json, os
from pathlib import Path
from stage3_oxe_eval.prepare import sha
from whenami_unified.data import DEV, INDEX
from whenami_unified.loading import A50
from whenami_present.loading import source_record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("teacher", "student"), required=True)
    p.add_argument("--teacher", type=Path); p.add_argument("--resume-from", type=Path)
    p.add_argument("--world", type=int, required=True); p.add_argument("--global-batch", type=int, required=True)
    p.add_argument("--micro", type=int, default=1); p.add_argument("--engineering", action="store_true")
    a = p.parse_args()
    if a.global_batch % (a.world*a.micro):
        raise ValueError(f"global batch {a.global_batch} not divisible by world {a.world} x micro {a.micro}")
    dev = json.loads((DEV/"MANIFEST.json").read_text())
    if not dev["complete"] or dev["train_overlap"] or dev["train_index_sha256"] != sha(INDEX):
        raise ValueError("DEV contract")
    if a.stage == "teacher":
        source_record(A50, "a50")
    else:
        if a.teacher is None:
            raise ValueError("Stage II needs TEACHER_CKPT")
        source_record(a.teacher, "teacher", a.engineering)
    if a.resume_from:
        meta = json.loads((a.resume_from/"COMMITTED.json").read_text())
        if not meta["complete"] or meta["world_size"] != a.world:
            raise ValueError(f"resume needs same world size: ckpt {meta['world_size']} vs {a.world}")
        s = meta["recipe"]["settings"]
        if s["stage"] != a.stage or s["global_batch"] != a.global_batch:
            raise ValueError("resume stage/global batch mismatch")
        missing = [i for i in range(a.world) if not (a.resume_from/f"rng-{i:05d}.pt").exists()]
        if missing:
            raise ValueError("missing rng shards "+str(missing[:5]))
    print("PRESENT_PREFLIGHT_OK", a.stage, "world", a.world, "accum", a.global_batch//(a.world*a.micro), flush=True)


if __name__ == "__main__":
    main()
