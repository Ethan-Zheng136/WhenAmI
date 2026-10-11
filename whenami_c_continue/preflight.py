"""Four completed B parents, immutable provenance, and actual320 continuation gates."""
import argparse
import json
import math
from pathlib import Path

from stage3.common import BASE
from stage3_oxe_eval.prepare import sha
from stage3_perf.checkpoint import SCHEMA
from whenami_timeline_joint.data import DEV, INDEX
from whenami_c_continue.contracts import PARENTS, START, END, SCHEDULE, source_record, file_inventory
from whenami_c_continue.train import fingerprints


def local_preflight(case):
    proof=json.loads((BASE/"whenami_c_continue/LOCAL_VERIFICATION.json").read_text())
    if not proof["local_passed"] or proof["source"]!=fingerprints():
        raise ValueError("continuation local proof/source contract missing or changed")
    if sha(BASE/"run_c_continue_320.sh")!=proof["launcher_sha256"]:
        raise ValueError("continuation launcher changed")
    source=source_record(case)
    if source!=proof["parents"][case]:
        raise ValueError("continuation parent changed")
    manifest=json.loads((DEV/"MANIFEST.json").read_text())
    if not manifest["complete"] or manifest["domains"]!=27 or manifest["train_overlap"] or manifest["train_index_sha256"]!=sha(INDEX):
        raise ValueError("B DEV/TRAIN partition changed")
    return proof


def check_admission(root,case):
    root=Path(root);source=source_record(case)
    tiny=json.loads((root/"topology_probe/RESULT.json").read_text())
    if not tiny["complete"] or tiny["world_size"]!=320 or not tiny["next_update_bitwise_equal"]:
        raise ValueError("actual320 topology probe incomplete")
    recipes=[]
    for segment,start,end in (("admission1",20001,20100),("admission2",20101,20120)):
        d=root/segment
        cp=d/"checkpoints"/f"step-{end:06d}"
        meta=json.loads((cp/"COMMITTED.json").read_text())
        if (not meta["complete"] or meta["schema"]!=SCHEMA or meta["step"]!=end
                or meta["world_size"]!=320 or meta["shard_width"]!=16):
            raise ValueError("continuation checkpoint topology/step mismatch")
        file_inventory(cp)
        recipe=meta["recipe"];recipes.append(recipe)
        if recipe["parent"]!=source or recipe["source"]!=fingerprints() or recipe["settings"]["steps"]!=END:
            raise ValueError("continuation identity changed")
        if recipe["continuation"]["schedule"]!=SCHEDULE or recipe["action_condition_tokens"]!=32:
            raise ValueError("wrong continuation schedule")
        rows=[json.loads(x) for x in (d/"train.jsonl").read_text().splitlines()]
        if [r["step"] for r in rows]!=list(range(start,end+1)):
            raise ValueError("continuation skipped/repeated updates")
        for row in rows:
            if not all(math.isfinite(row[k]) for k in ("action_fm","current_alignment","future_alignment","weighted_alignment","total","grad_norm")):
                raise ValueError("nonfinite continuation training")
            if row["lr"]!=source["parent_terminal_lr"]:
                raise ValueError("learning-rate warmup/reset detected")
            if row["read_scale"]!=source["read_scale"]:
                raise ValueError("C future read-in scale changed/reset")
        grads=[r["gradients"] for r in rows if r["gradients"]]
        for key in ("vlm_vision","projector","llm","qformer","time","dit"):
            if not grads or not all(math.isfinite(g[key]) for g in grads) or max(g[key] for g in grads)<=0:
                raise ValueError("missing B body gradient "+key)
        for rank in range(320):
            restore=json.loads((d/f"restore-rank-{rank:05d}.json").read_text())
            if (not restore["complete"] or not restore["model_Adam_rank_RNG_restored"]
                    or restore["restored_step"]!=start-1 or restore["local_Adam_step_max"]!=start-1
                    or restore["rank"]!=rank or restore["world"]!=320
                    or restore["read_scale"]!=source["read_scale"] or restore["future_gap"]!=4
                    or restore["original_parent_recipe_used"]!=(segment=="admission1")):
                raise ValueError("model/Adam/RNG parent restoration not verified on every rank")
        first=json.loads((d/"RESUME_VERIFIED.json").read_text())
        if not first["complete"] or first["restored_step"]!=start-1 or first["first_update"]!=start:
            raise ValueError("first resumed update not verified")
        replay=json.loads((d/"CHECKPOINT_REPLAY_VERIFIED.json").read_text())
        if not replay["complete"] or replay["max_prediction_abs_error"]>1e-6 or replay["episodes"]!=410:
            raise ValueError("checkpoint replay differs")
        dev=json.loads((d/"development"/f"step-{end:06d}/RESULT.json").read_text())
        if not dev["complete"] or not dev["causal_student_only"] or len(dev["domains"])!=27:
            raise ValueError("continuation causal evaluation incomplete")
    if recipes[0]!=recipes[1]:
        raise ValueError("continuation admission recipe differs across segments")
    return root/"admission2/checkpoints/step-020120"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",choices=tuple(PARENTS),required=True)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--phase",choices=("run","admit","full"),required=True)
    a=p.parse_args();local_preflight(a.case)
    if a.phase in ("run","admit"):
        if any((a.root/n).exists() for n in ("topology_probe","admission1","admission2","main")):
            raise ValueError("fresh continuation RUN_ID required; never overwrite existing segments")
    else:
        check_admission(a.root,a.case)
        if (a.root/"main").exists():
            raise ValueError("main already exists; explicit checkpoint recovery required")
    print("C_CONTINUATION_PREFLIGHT_OK",a.case,a.phase,flush=True)


if __name__=="__main__":main()
