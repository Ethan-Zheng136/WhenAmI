"""Fixed A5k: measure teacher advantage over the SAME weights with future removed.

No learning. Existing410-episode DEV only. Results are a transfer diagnostic,
not a trained Phase-B student, not closed-loop success.
"""

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from stage3_oxe_eval.prepare import write_json,sha
from stage3_oxe_eval.metrics import episode_metrics
from stage3_oxe_eval.summarize import paired
from whenami_timeline_joint.data import Inputs,tensorize,frozen_features
from whenami_timeline_transfer.loader import load_fixed,CHECKPOINT
from whenami_timeline_transfer.contracts import make_prefix,select_anchor


@torch.no_grad()
def sample(model,conditioning,noise):
    x=noise.clone()
    conditioning={k:v.expand(len(x),-1,-1) for k,v in conditioning.items()}
    ts=torch.linspace(0,1,10,device=x.device)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        for j in range(9):
            v=model(noisy_actions=x,tau=ts[j].expand(len(x)),conditioning=conditioning)["velocity"]
            x=x+v.float()*(ts[j+1]-ts[j])
    if not torch.isfinite(x).all():raise FloatingPointError("nonfinite sampled action")
    return x.cpu().numpy()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device("cuda:0")
    data=Inputs()
    recipe={"checkpoint_manifest_sha256":sha(CHECKPOINT/"COMMITTED.json"),
            "dev_manifest_sha256":sha(data.dev_root/"MANIFEST.json"),
            "source_sha256":sha(__file__),"seeds":[42,43,44],
            "scope":"FixedA5k teacher vs same-weight causal prefix; current middle action only"}
    if args.resume and json.loads((args.output/"recipe.json").read_text())!=recipe:
        raise ValueError("resume recipe changed")
    write_json(args.output/"recipe.json",recipe)
    per=args.output/"per_episode";per.mkdir(exist_ok=True)
    model,visual,transform,tokenizer,prompt,record=load_fixed()
    write_json(args.output/"model.json",record)
    domains,metrics={"teacher":{},"causal_prefix":{}},{"teacher":[],"causal_prefix":[]}
    names=[]
    for index in tqdm(range(len(data.clips)),desc="A5k teacher-causal gap"):
        row=data.clips[index];dest=per/(row["dataset"]+"--"+row["episode_key"]+".npz")
        if not dest.exists():
            case=data.dev(index)
            x,y=tensorize([case],transform,tokenizer,prompt,device)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                features=frozen_features(visual,x["pixels"])
                all_conditions=model(**x,**features,visible=torch.ones_like(x["times"],dtype=torch.bool),
                                     condition_only=True)
                teacher=select_anchor(all_conditions,1,3,1)
                prefix=make_prefix(x,features,1)
                causal=select_anchor(model(**prefix.kwargs(),condition_only=True),1,2,1)
            noise=[]
            for seed in recipe["seeds"]:
                identity=int.from_bytes(hashlib.sha256(
                    f"transfer-gap:{row['episode_key']}:{seed}".encode()).digest()[:8],"little")%(2**63-1)
                g=torch.Generator(device=device).manual_seed(identity)
                noise.append(torch.randn((16,7),generator=g,device=device))
            noise=torch.stack(noise)
            teacher_action=sample(model,teacher,noise)
            causal_action=sample(model,causal,noise)
            with np.load(data.dev_root/row["file"]) as raw:
                target=np.asarray(raw["raw_actions"][1:2]);mask=np.asarray(raw["action_mask"][1:2])
            tmp=dest.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f,teacher=teacher_action[None],causal_prefix=causal_action[None],
                                    target_raw=target,action_mask=mask,noise=noise.cpu().numpy(),
                                    data_sha256=np.asarray(row["sha256"]))
            os.replace(tmp,dest)
        with np.load(dest) as a:
            if str(a["data_sha256"])!=row["sha256"]:raise ValueError("case identity mismatch")
            stat=data.stats[row["dataset"]]["action"]
            for mode in metrics:
                m=episode_metrics(a[mode],a["target_raw"],a["action_mask"],stat,stat)["mean"]
                metrics[mode].append(m)
                domains[mode].setdefault(row["dataset"],[]).append(m)
        names.append(row["dataset"])
        if index%10==0:
            write_json(args.output/"progress.json",{"utc":datetime.now(timezone.utc).isoformat(),
                "completed":index+1,"total":len(data.clips)})
    summary={}
    for mode in metrics:
        by_domain={d:{k:float(np.mean([m[k] for m in rows])) for k in rows[0]}
                   for d,rows in domains[mode].items()}
        summary[mode]={"domains":by_domain,
            "macro":{k:float(np.mean([m[k] for m in by_domain.values()]))
                     for k in ("mse6","unclipped_mse6","mae6","first_action_mse6","gripper_accuracy")}}
    comparison=paired(np.asarray([m["mse6"] for m in metrics["causal_prefix"]]),
                      np.asarray([m["mse6"] for m in metrics["teacher"]]),np.asarray(names))
    write_json(args.output/"RESULT.json",{"complete":True,"episodes":len(names),"domains":len(set(names)),
        "models":summary,"teacher_vs_causal_positive_is_teacher_advantage":comparison,
        "scope":recipe["scope"],"notes":["Removing future is an inference intervention, not matched B training.",
              "Teacher need not beat prefix on every episode; latent target matching is not action success."]})
    print("TRANSFER_GAP_COMPLETE",comparison,flush=True)


if __name__=="__main__":main()
