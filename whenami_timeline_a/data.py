"""Distinct actual anchors from original OXE TRAIN only; no labels in model inputs."""

import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from stage3.common import STORE
from whenami_revision.raw_data import RevisionEpisodes
from stage3_scale.oxe_data import action_window
from evaluation.past_protocol import normalize_actions

INDEX=STORE/"outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json"


def choose_times(length,rng,nodes=3,max_gap=12):
    if nodes<2 or length<nodes+1:raise ValueError("not enough distinct actual frames")
    upper=min(max_gap,(length-2)//(nodes-1))
    gaps=rng.integers(1,upper+1,size=nodes-1)
    start=int(rng.integers(0,length-1-int(gaps.sum())))
    return np.r_[start,start+np.cumsum(gaps)].astype(np.int64)


def visibility(times,seed,hide_interior=False,causal=False):
    if times.ndim!=2 or times.shape[1]<2:raise ValueError("batched time anchors required")
    mask=torch.ones_like(times,dtype=torch.bool)
    rng=np.random.default_rng(seed)
    if hide_interior:
        for b in range(len(times)):
            # One random node with at least one visible supporting node remains.
            lo,hi=(1,times.shape[1]-1) if times.shape[1]>2 else (0,times.shape[1])
            mask[b,int(rng.integers(lo,hi))]=False
    cutoff=times.clone() if causal else times[:,-1:].expand_as(times).clone()
    return mask,cutoff


class RawAnchors(RevisionEpisodes):
    def __init__(self,index=INDEX,nodes=3):
        super().__init__(index)
        self.nodes=nodes
        self.episodes={d:[e for e in rows if e["length"]>=nodes+1]
                       for d,rows in self.episodes.items()}
        if any(not rows for rows in self.episodes.values()):raise ValueError("domain lacks requested anchor count")
        self.domains=sorted(self.episodes)
        p=np.asarray([self.index["sampling_probabilities"][d] for d in self.domains])
        self.probabilities=p/p.sum()

    def select(self,seed,step=0,rank=0,ordinal=0,domain=None):
        rng=np.random.default_rng(np.random.SeedSequence([seed,step,rank,ordinal]))
        name=domain or self.domains[int(rng.choice(len(self.domains),p=self.probabilities))]
        rows=self.episodes[name]
        for _ in range(512):
            entry=rows[int(rng.integers(len(rows)))]
            _,episode=self._episode(entry)
            times=choose_times(episode["length"],rng,self.nodes)
            texts=episode["instructions"][int(times[0]):int(times[-1])+1]
            if not texts[0].strip() or any(x!=texts[0] for x in texts):continue
            if any(int(episode["frame_spans"][int(t),1])<=0 for t in times):continue
            return entry,times
        raise RuntimeError(f"failed to sample a constant-instruction clip in {name}")

    def materialize(self,entry,times,statistics):
        payload,episode=self._episode(entry)
        if len(times)!=self.nodes or (np.diff(times)<=0).any() or min(times)<0 or max(times)>=episode["length"]-1:
            raise ValueError("invalid actual anchor times")
        images=[]
        for t in times:
            lo,n=episode["frame_spans"][int(t)].tolist()
            with Image.open(io.BytesIO(payload["frames"][lo:lo+n].numpy().tobytes())) as image:
                images.append(np.asarray(image.convert("RGB").resize((224,224))))
        actions,masks=[],[]
        for t in times:
            a,m=action_window(episode["raw_actions"].numpy(),int(t))
            actions.append(normalize_actions(a,statistics[episode["dataset"]]["action"]))
            masks.append(m)
        observations={
            "images":np.stack(images),
            "instructions":[episode["instructions"][int(t)] for t in times],
            "times":torch.from_numpy(times-times[0]),
        }
        labels={"actions":torch.from_numpy(np.stack(actions)),
                "action_mask":torch.from_numpy(np.stack(masks))}
        provenance={
            "dataset":episode["dataset"],"episode_key":episode["episode_key"],
            "source_shard":entry["shard"],"source_times":times.tolist(),
            "actual_gaps":np.diff(times).tolist(),"role":"original_STAGE3_TRAIN",
            "encoder_support":[[int(t)] for t in times],
            "action_padding":(np.stack(masks).sum(-1)<16).tolist(),
        }
        return observations,labels,provenance


def stack_cases(cases,device):
    inputs={k:torch.stack([c["inputs"][k] for c in cases]).to(device)
            for k in ("patches","vae","z","times")}
    labels={k:torch.stack([c["labels"][k] for c in cases]).to(device)
            for k in ("actions","action_mask")}
    return inputs,labels
