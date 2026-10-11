"""Raw train/DEV nodes: only immutable visual features may be cached."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from stage3.common import DEFAULT_FUTURE, STORE
from stage3_scale.model import tokenize_instruction
from whenami_timeline_a.data import RawAnchors, INDEX
from stage3_oxe_eval.prepare import sha
from evaluation.past_protocol import normalize_actions

DEV=STORE/"outputs/whenami_timeline_a/dev27_16ep_01"


def tensorize(cases,transform,tokenizer,prompt,device):
    # Observations and action labels arrive as separate fields and leave separately.
    frames=[[transform(Image.fromarray(image)) for image in c["observations"]["images"]] for c in cases]
    pixels={key:torch.stack([torch.stack([x[key] for x in clip]) for clip in frames]).to(device)
            for key in frames[0][0]}
    ids=[tokenize_instruction(text,tokenizer,prompt) for c in cases for text in c["observations"]["instructions"]]
    longest=max(len(x) for x in ids)
    tokens=torch.full((len(ids),longest),tokenizer.pad_token_id,dtype=torch.long,device=device)
    attention=torch.zeros_like(tokens,dtype=torch.bool)
    for i,value in enumerate(ids):tokens[i,:len(value)]=value.to(device);attention[i,:len(value)]=True
    inputs={"pixels":pixels,"input_ids":tokens,"attention_mask":attention,
            "times":torch.stack([c["observations"]["times"] for c in cases]).to(device)}
    labels={key:torch.stack([c["labels"][key] for c in cases]).to(device)
            for key in ("actions","action_mask")}
    return inputs,labels


@torch.no_grad()
def frozen_features(visual,pixels):
    # Single frame per node; no full-video feature later sliced by a mask.
    b,k=next(iter(pixels.values())).shape[:2]
    flat={key:value.flatten(0,1) for key,value in pixels.items()}
    p,v=visual(flat,flat,video_pixel_values={key:value[:,None] for key,value in flat.items()})
    n=int(p.shape[1]**.5)
    if n*n!=p.shape[1]:raise ValueError("non-square visual patches")
    p=p.reshape(b*k,n,n,-1).permute(0,3,1,2)
    p=torch.nn.functional.adaptive_avg_pool2d(p,(8,8)).flatten(2).transpose(1,2)
    return {"patches":p.reshape(b,k,*p.shape[1:]),"vae":v.reshape(b,k,*v.shape[1:])}


class Inputs:
    def __init__(self,nodes=3,dev_root=DEV,dev_limit=0):
        self.raw=RawAnchors(INDEX,nodes)
        self.stats=json.loads((DEFAULT_FUTURE.parents[1]/"dataset_statistics.json").read_text())
        self.dev_root=Path(dev_root)
        self.manifest=json.loads((self.dev_root/"MANIFEST.json").read_text())
        if not self.manifest["complete"] or self.manifest["nodes"]!=nodes or self.manifest["train_overlap"]:
            raise ValueError("invalid dev contract")
        self.clips=[];counts={}
        for row in self.manifest["clips"]:
            name=row["dataset"];counts[name]=counts.get(name,0)+1
            if not dev_limit or counts[name]<=dev_limit:self.clips.append(row)

    def batch(self,count,seed,step,rank,micro=0):
        cases=[]
        for i in range(count):
            e,t=self.raw.select(seed,step,rank,micro*count+i)
            obs,labels,prov=self.raw.materialize(e,t,self.stats)
            cases.append({"observations":obs,"labels":labels,"provenance":prov})
        return cases

    def dev(self,index):
        row=self.clips[index];path=self.dev_root/row["file"]
        if sha(path)!=row["sha256"]:raise ValueError("dev clip changed")
        with np.load(path) as a:
            obs={"images":a["images"],"instructions":a["instructions"].tolist(),
                 "times":torch.from_numpy(a["times"].copy())}
            labels={"actions":torch.from_numpy(normalize_actions(a["raw_actions"],self.stats[row["dataset"]]["action"])),
                    "action_mask":torch.from_numpy(a["action_mask"].copy())}
        return {"observations":obs,"labels":labels,"provenance":row}
