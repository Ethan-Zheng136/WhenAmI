"""Compare matched B0/B1 causal DEV scores, validating all saved predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from stage3_oxe_eval.prepare import sha,write_json
from stage3_oxe_eval.summarize import paired
from stage3.common import DEFAULT_FUTURE
from whenami_timeline_joint.data import DEV


def load_run(root,step):
    root=Path(root);recipe=json.loads((root/"recipe.json").read_text())
    d=root/"development"/f"step-{step:06d}"
    report=json.loads((d/"RESULT.json").read_text())
    if not report["complete"] or not report["causal_student_only"]:
        raise ValueError("not a completed causal evaluation")
    manifest=json.loads((DEV/"MANIFEST.json").read_text())
    if recipe["dev_manifest_sha256"]!=sha(DEV/"MANIFEST.json"):
        raise ValueError("dev changed")
    stats=json.loads((DEFAULT_FUTURE.parents[1]/"dataset_statistics.json").read_text())
    scored={}
    for p in sorted(d.glob("rank-*.json")):
        for row in json.loads(p.read_text()):
            key=(row["dataset"],row["episode_key"])
            if key in scored:raise ValueError("duplicate episode")
            scored[key]=row
    predictions={}
    for p in d.glob("rank-*.npz"):
        with np.load(p) as a:
            for j,index in enumerate(a["indices"]):
                row=manifest["clips"][int(index)]
                key=(row["dataset"],row["episode_key"])
                prediction=a["predictions"][j];target=a["targets"][j];mask=a["masks"][j]
                if key in predictions or not np.isfinite(prediction).all():raise ValueError("invalid prediction")
                st=stats[row["dataset"]]["action"]
                unused=np.asarray(st["min"])==np.asarray(st["max"])
                clipped=np.where(unused,0.,np.clip(prediction,-1,1))
                error=np.square(clipped[...,:6]-target[None,...,:6]).mean(-1)
                mse=float(((error*mask).sum(-1)/mask.sum()).mean())
                if abs(mse-scored[key]["mse6"])>1e-6:raise ValueError("score recheck differs")
                predictions[key]={"target":target.copy(),"mask":mask.copy()}
    if set(predictions)!=set(scored):raise ValueError("missing predictions")
    return recipe,report,scored,predictions


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--action-only",type=Path,required=True)
    p.add_argument("--aligned",type=Path,required=True)
    p.add_argument("--step",type=int,default=20000)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    a,ar,ascore,ap=load_run(args.action_only,args.step)
    b,br,bscore,bp=load_run(args.aligned,args.step)
    if a["settings"]["variant"]!="action_only" or b["settings"]["variant"]!="condition_align":
        raise ValueError("wrong variants")
    aa=json.loads(json.dumps(a));bb=json.loads(json.dumps(b))
    aa["settings"].pop("variant");bb["settings"].pop("variant")
    if aa!=bb or ar["noise_seeds"]!=br["noise_seeds"]:
        raise ValueError("unmatched B training or evaluation")
    keys=sorted(ascore)
    if keys!=sorted(bscore):raise ValueError("unpaired cohort")
    for key in keys:
        for field in ("target","mask"):
            if not np.array_equal(ap[key][field],bp[key][field]):raise ValueError("target/mask mismatch")
    names=np.asarray([k[0] for k in keys])
    comparison=paired(np.asarray([ascore[k]["mse6"] for k in keys]),
                      np.asarray([bscore[k]["mse6"] for k in keys]),names)
    result={"complete":True,"step":args.step,"action_only":ar,"condition_align":br,"comparison":comparison,
        "teacher_step":a["teacher"]["step"],"matched_recipe_except_alignment_variant":True,
        "scope":"Fixed-teacher pilot, fixed training seed, causal DEV only; no closed-loop result",
        "next_requirement":"Repeat with further training seeds, compare full A->B pooled/timeline paths, and fixed-protocol closed-loop tasks."}
    write_json(args.output/"RESULT.json",result)
    print(json.dumps(comparison,indent=2))


if __name__=="__main__":main()
