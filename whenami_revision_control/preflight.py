"""Strict admission gates for two additional320PPU Phase-B pilot jobs."""

import argparse
import json
import math
from pathlib import Path

from stage3.common import BASE,STORE
from stage3_oxe_eval.prepare import sha
from stage3_perf.checkpoint import SCHEMA
from whenami_revision_control.loading import source_record
from whenami_revision_control.train import fingerprints
from whenami_timeline_joint.data import DEV,INDEX


def local_preflight():
    proof=json.loads((BASE/"whenami_revision_control/LOCAL_VERIFICATION.json").read_text())
    if not proof["local_passed"] or proof["source"]!=fingerprints():
        raise ValueError("B source/dependency drift or incomplete local verification")
    if sha(BASE/"run_revision_control_320.sh")!=proof["launcher_sha256"]:
        raise ValueError("B launcher changed")
    source=source_record()
    if source!=proof["sources"]:
        raise ValueError("pinned teacher changed")
    m=json.loads((DEV/"MANIFEST.json").read_text())
    if not m["complete"] or m["domains"]!=27 or m["train_overlap"] or m["train_index_sha256"]!=sha(INDEX):
        raise ValueError("B dev/train partition mismatch")
    return proof


def check_admission(root,variant):
    root=Path(root)
    tiny=json.loads((root/"topology_probe/RESULT.json").read_text())
    if not tiny["complete"] or tiny["world_size"]!=320 or not tiny["next_update_bitwise_equal"]:
        raise ValueError("B320 topology proof missing")
    recipes=[]
    for segment,step,start in (("admission1",100,1),("admission2",120,101)):
        d=root/segment
        meta=json.loads((d/"checkpoints"/f"step-{step:06d}/COMMITTED.json").read_text())
        if not meta["complete"] or meta["schema"]!=SCHEMA or meta["world_size"]!=320 or meta["shard_width"]!=16:
            raise ValueError("B checkpoint topology invalid")
        recipe=meta["recipe"];recipes.append(recipe)
        if recipe["settings"]["variant"]!=variant or recipe["source"]!=fingerprints():
            raise ValueError("B experiment identity changed")
        if recipe["sources"]!=source_record():
            raise ValueError("B used wrong teacher")
        rows=[json.loads(s) for s in (d/"train.jsonl").read_text().splitlines()]
        if [r["step"] for r in rows]!=list(range(start,step+1)):
            raise ValueError("B admission updates missing")
        if not all(math.isfinite(r[k]) for r in rows for k in ("action_fm","node_alignment","revision_alignment","weighted_aux","delta_rms","total","grad_norm")):
            raise ValueError("nonfinite B training")
        gradients=[r["gradients"] for r in rows if r["gradients"]]
        for k in ("vlm_vision","projector","llm","qformer","time","dit","memory","bridge"):
            if not gradients or max(r[k] for r in gradients)<=0:raise ValueError("B body gradient missing "+k)
        if start==1 and rows[0]["delta_rms"]!=0:
            raise ValueError("initial action residual must be zero")
        if not any(r["delta_rms"]>0 for r in rows):
            raise ValueError("action bridge did not begin learning")
        dev=json.loads((d/"development"/f"step-{step:06d}/RESULT.json").read_text())
        if not dev["complete"] or not dev["causal_student_only"] or len(dev["domains"])!=27 or dev.get("protocol")!="revision_last_anchor_executed_history_v1":
            raise ValueError("B causal evaluation missing")
    if recipes[0]!=recipes[1]:raise ValueError("B resume recipe differs")
    restore=json.loads((root/"admission2/RESUME_VERIFIED.json").read_text())
    if not restore["complete"] or restore["restored_step"]!=100 or restore["first_update"]!=101:
        raise ValueError("B new-process optimizer restore not verified")
    return root/"admission2/checkpoints/step-000120"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--variant",choices=("direct_history","revised_nodes"),required=True)
    p.add_argument("--phase",choices=("admit","run","full"),required=True)
    a=p.parse_args()
    local_preflight()
    if a.phase in ("admit","run"):
        if any((a.root/n).exists() for n in ("topology_probe","admission1","admission2","main")):
            raise ValueError("fresh B RUN_ID required")
    else:
        check_admission(a.root,a.variant)
        if (a.root/"main").exists():raise ValueError("B main exists; explicit recovery required")
    print("REVISION_CONTROL_PREFLIGHT_OK",a.variant,a.phase,flush=True)


if __name__=="__main__":main()
