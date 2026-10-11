"""Same OXE27 split; three OBSERVED nodes, action target at the last node."""
from dataclasses import dataclass
import copy

import torch

from whenami_timeline_joint.data import Inputs as OriginalInputs, DEV, INDEX, tensorize, frozen_features
from whenami_timeline_transfer.contracts import Prefix
from whenami_revision_control.contracts import ObservedWindow, RevisionInput


@dataclass(frozen=True)
class CurrentLabel:
    actions: torch.Tensor
    valid: torch.Tensor


def observed_case(case, start=0, stop=None):
    o = case["observations"]
    stop = len(o["images"]) if stop is None else stop
    times = o["times"][start:stop].clone()
    times -= times[:1].clone()
    return {"observations": {
        "images": o["images"][start:stop].copy(),
        "instructions": list(o["instructions"][start:stop]), "times": times},
        "labels": {k: v[start:stop].clone() for k, v in case["labels"].items()},
        "provenance": case["provenance"]}


def executed_intervals(case, width=12):
    """Earlier anchor action windows contain the complete already-executed gap.

    The last anchor's target actions are NEVER consulted to create model inputs.
    """
    times = case["observations"]["times"]
    k = len(times)
    commands = case["labels"]["actions"].new_zeros(k, width, 7)
    mask = torch.zeros(k, width, dtype=torch.bool)
    for j in range(1, k):
        gap = int(times[j] - times[j-1])
        if not 1 <= gap <= width or not case["labels"]["action_mask"][j-1, :gap].all():
            raise ValueError("missing actual executed commands between observed anchors")
        commands[j, :gap] = case["labels"]["actions"][j-1, :gap].detach()
        mask[j, :gap] = True
    return commands, mask


class Inputs(OriginalInputs):
    pass


def teacher_prefix(cases, visual, transform, tokenizer, prompt, device, stop):
    subset = [observed_case(c, 0, stop) for c in cases]
    images, _ = tensorize(subset, transform, tokenizer, prompt, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = frozen_features(visual, images["pixels"])
    return images, features


def batch(cases, visual, transform, tokenizer, prompt, device):
    counts = {len(c["observations"]["images"]) for c in cases}
    if len(counts) != 1 or not 1 <= next(iter(counts)) <= 3:
        raise ValueError("batch must contain equally sized observed windows")
    k = next(iter(counts))
    observed, labels = tensorize([observed_case(c) for c in cases], transform, tokenizer, prompt, device)
    # Independently construct the original B input. Earlier unused instructions
    # cannot change its padding, images, or local time origin.
    base_cases = [observed_case(c, max(0, k-2)) for c in cases]
    base, _ = tensorize(base_cases, transform, tokenizer, prompt, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = frozen_features(visual, observed["pixels"])
        base_features = frozen_features(visual, base["pixels"])
    intervals = [executed_intervals(c) for c in cases]
    prefix = Prefix(base["pixels"], base["input_ids"], base["attention_mask"],
                    base_features["patches"], base_features["vae"], base["times"]).validate()
    window = ObservedWindow(features["patches"], features["vae"], observed["times"],
                            torch.stack([x[0] for x in intervals]).to(device),
                            torch.stack([x[1] for x in intervals]).to(device)).validate()
    frames=[]
    for j in range(k):
        f, _ = tensorize([observed_case(c,j,j+1) for c in cases],transform,tokenizer,prompt,device)
        frames.append(Prefix(f["pixels"],f["input_ids"],f["attention_mask"],
                             features["patches"][:,j:j+1],features["vae"][:,j:j+1],
                             f["times"]).validate())
    return RevisionInput(prefix, window, tuple(frames)).validate(), CurrentLabel(
        labels["actions"][:, -1], labels["action_mask"][:, -1]), observed, features
