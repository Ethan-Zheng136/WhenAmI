"""Physical student prefix is made BEFORE per-node image/token encoding."""

from dataclasses import dataclass
import numpy as np
import torch

from whenami_timeline_joint.data import Inputs,tensorize,frozen_features,DEV
from whenami_timeline_transfer.contracts import Prefix


@dataclass(frozen=True)
class CurrentLabel:
    actions:torch.Tensor
    valid:torch.Tensor


def raw_prefix(case,current_index=1):
    o=case["observations"];n=current_index+1
    if len(o["images"])<=current_index:raise ValueError("current anchor absent")
    # .copy() prevents views retaining future data storage.
    return {
        "observations":{"images":o["images"][:n].copy(),
                        "instructions":list(o["instructions"][:n]),
                        "times":o["times"][:n].clone()},
        "labels":{k:v[:n].clone() for k,v in case["labels"].items()},
        "provenance":{k:case["provenance"][k] for k in ("dataset","episode_key")},
    }


def batch(cases,visual,transform,tokenizer,prompt,device):
    full,labels=tensorize(cases,transform,tokenizer,prompt,device)
    # Different token padding/metadata from teacher cannot pass into the student.
    causal,_=tensorize([raw_prefix(c) for c in cases],transform,tokenizer,prompt,device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        all_features=frozen_features(visual,full["pixels"])
    # This visual path is frame-local and immutable. Slice copied features, NOT
    # full-video states; pixel and language prefix was already independently made.
    prefix=Prefix(
        pixels=causal["pixels"],input_ids=causal["input_ids"],
        attention_mask=causal["attention_mask"],
        patches=all_features["patches"][:,:2].clone(),vae=all_features["vae"][:,:2].clone(),
        times=causal["times"],
    ).validate()
    return full,all_features,prefix,CurrentLabel(labels["actions"][:,1],labels["action_mask"][:,1])


def student_inputs(cases,visual,transform,tokenizer,prompt,device):
    """Pure causal evaluator: does not encode/load full future through any model."""
    causal,labels=tensorize([raw_prefix(c) for c in cases],transform,tokenizer,prompt,device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        features=frozen_features(visual,causal["pixels"])
    prefix=Prefix(
        pixels=causal["pixels"],input_ids=causal["input_ids"],attention_mask=causal["attention_mask"],
        patches=features["patches"],vae=features["vae"],times=causal["times"]).validate()
    return prefix,CurrentLabel(labels["actions"][:,1],labels["action_mask"][:,1])
