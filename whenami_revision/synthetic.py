"""Controlled partial-observation branches. NOT robot data or a benchmark score."""

import numpy as np
import torch


BRANCHES = ("empty_attempt", "grasp_then_drop", "grasp_and_hold")
DECISIONS = ("confirm", "retry", "recover", "transport")


def paired_branches(groups, seed, first_group=0, split="train"):
    if groups < 1 or first_group < 0 or split not in ("train", "diagnostic"):
        raise ValueError("invalid synthetic group request")
    offset = 0 if split == "train" else 1
    obs, times, actions, masks, states, decisions, identities, branches = [], [], [], [], [], [], [], []
    for group in range(first_group, first_group + groups):
        rng = np.random.default_rng(np.random.SeedSequence([seed, group, offset]))
        # Identical nuisance signals, executed commands, timing and prompt within
        # every group. Hidden outcome cannot be read from a random-seed shortcut.
        gaps = rng.integers(1, 4, size=3)
        ts = np.r_[0, np.cumsum(gaps)].astype(np.int64)
        common = rng.normal(0, .1, size=(4, 12)).astype(np.float32)
        common[:, :4] = 0
        common[:, 3] = [0., 1., 1., .5]  # common observed hand position, not outcome
        common[1:3, 0] = 1              # outcome evidence visible at1,2 only
        acts = np.zeros((4, 3, 7), np.float32); valid = np.zeros((4, 3), bool)
        for i, gap in enumerate(gaps, 1):
            valid[i, :gap] = True
            acts[i, :gap] = rng.normal(0, .05, size=(gap, 7))
            acts[i, :gap, 6] = 1. if i == 1 else 0.
        for branch in range(3):
            x = common.copy()
            acquired = branch != 0
            held_after = branch == 2
            x[1, 1] = float(acquired)
            x[2, 1] = float(held_after)
            # Frame3 is occluded again; empty_attempt and later_drop require
            # DIFFERENT current decisions, despite the same latest image.
            labels = np.asarray([acquired, acquired, held_after, held_after], np.int64)
            action_label = np.asarray([
                0, 1 if not acquired else 3,
                (1, 2, 3)[branch], (1, 2, 3)[branch],
            ], np.int64)
            obs.append(x); times.append(ts.copy()); actions.append(acts.copy()); masks.append(valid.copy())
            states.append(labels); decisions.append(action_label); branches.append(branch)
            identities.append(f"synthetic/{split}/{seed}/{group}/{BRANCHES[branch]}")
    inputs = {
        "observations": torch.from_numpy(np.stack(obs)),
        "times": torch.from_numpy(np.stack(times)),
        "executed_actions": torch.from_numpy(np.stack(actions)),
        "executed_mask": torch.from_numpy(np.stack(masks)),
    }
    labels = {"states": torch.from_numpy(np.stack(states)),
              "decisions": torch.from_numpy(np.stack(decisions)),
              "branch": torch.tensor(branches), "identities": identities}
    return inputs, labels


def prefix(inputs, length):
    if not 1 <= length <= inputs["times"].shape[1]:
        raise ValueError("bad evidence cutoff")
    return {k: v[:, :length].clone() for k, v in inputs.items()}
