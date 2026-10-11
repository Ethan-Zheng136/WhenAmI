"""Gate actual320-PPU trials on current-source local checks and prior32-PPU success."""

import argparse
import json
from pathlib import Path

from stage3.common import BASE,STORE
from stage3_oxe_eval.prepare import sha
from stage3_perf.checkpoint import SCHEMA
from whenami_timeline_joint.train import fingerprints
from whenami_timeline_a_training.preflight import check_admission as frozen_admission
from whenami_timeline_joint.data import DEV,INDEX


def local_preflight():
    proof=json.loads((BASE/"whenami_timeline_joint/LOCAL_VERIFICATION.json").read_text())
    if not proof["local_passed"] or proof["sources"]!=fingerprints():
        raise ValueError("full-VLM code/dependencies changed or unverified")
    if sha(BASE/"run_timeline_joint_320.sh")!=proof["launcher_sha256"]:
        raise ValueError("launcher changed after verification")
    dev=json.loads((DEV/"MANIFEST.json").read_text())
    if (not dev["complete"] or dev["domains"]!=27 or dev["train_overlap"]
            or dev["train_index_sha256"]!=sha(INDEX)):
        raise ValueError("raw train/dev partition contract changed")
    if sha(DEV/"MANIFEST.json")!=proof["dev_manifest_sha256"]:
        raise ValueError("development cohort changed after local verification")
    frozen_admission(STORE/"outputs/whenami_timeline_a/training/timeline_a_timeline_32_10k_01","timeline")
    return proof


def check_large_admission(root,mode):
    root=Path(root)
    tiny=json.loads((root/"topology_probe/RESULT.json").read_text())
    if not tiny["complete"] or tiny["world_size"]!=320 or not tiny["next_update_bitwise_equal"]:
        raise ValueError("actual320-rank topology probe not passed")
    first=root/"admission1";second=root/"admission2"
    metas=[]
    for run,step,expected in ((first,100,list(range(1,101))),(second,120,list(range(101,121)))):
        meta=json.loads((run/"checkpoints"/f"step-{step:06d}"/"COMMITTED.json").read_text())
        if not meta["complete"] or meta["schema"]!=SCHEMA or meta["step"]!=step:
            raise ValueError("large checkpoint incomplete")
        if meta["world_size"]!=320 or meta["shard_width"]!=16:
            raise ValueError("wrong large admission topology")
        recipe=meta["recipe"]
        if recipe["settings"]["mode"]!=mode or recipe["source"]!=fingerprints():
            raise ValueError("large admission mode/source drift")
        rows=[json.loads(line) for line in (run/"train.jsonl").read_text().splitlines()]
        if [r["step"] for r in rows]!=expected:raise ValueError("missing admission updates")
        audits=[r["gradients"] for r in rows if r["gradients"]]
        if not audits or not all(max(a[k] for a in audits)>0 for k in audits[0]):
            raise ValueError("missing learned-module gradients")
        evaluation=json.loads((run/"development"/f"step-{step:06d}"/"RESULT.json").read_text())
        if not evaluation["complete"] or len(evaluation["domains"])!=27 or not evaluation["raw_vlm_recomputed"]:
            raise ValueError("development re-encoding not verified")
        metas.append(meta)
    if metas[0]["recipe"]!=metas[1]["recipe"]:raise ValueError("resume recipe mismatch")
    restored=json.loads((second/"RESUME_VERIFIED.json").read_text())
    if not restored["complete"] or restored["restored_step"]!=100 or restored["first_update"]!=101:
        raise ValueError("full optimizer new-process resume not established")
    return second/"checkpoints/step-000120"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=("timeline","pooled_query"),required=True)
    p.add_argument("--phase",choices=("admit","run","full"),required=True)
    a=p.parse_args()
    local_preflight()
    if a.phase in ("admit","run"):
        # Workers may create the shared log parent before node0 finishes local
        # checks. Actual training/probe directories must still be absent.
        if any((a.root/name).exists() for name in ("topology_probe","admission1","admission2","main")):
            raise ValueError("fresh RUN_ID required; do not overwrite prior work")
    else:
        check_large_admission(a.root,a.mode)
        if (a.root/"main").exists():raise ValueError("main already exists; recovery must be explicit")
    print("FULL_TIMELINE_PREFLIGHT_OK",a.mode,a.phase,flush=True)


if __name__=="__main__":main()
