"""Fixed, probabilistically specified stress task; not robot-performance evidence."""

import numpy as np
import torch


def delayed_groups(groups, length, seed, first_group=0, split="train", noise=.1,
                   flip_unrelated=False):
    if groups < 1 or length < 6 or not 0 <= noise < .5 or split not in ("train", "diagnostic"):
        raise ValueError("bad delayed-evidence specification")
    all_x, all_t, all_a, all_m, all_y, all_d, all_post = [], [], [], [], [], [], []
    ids, branches, reveals, relevances = [], [], [], []
    for group in range(first_group, first_group + groups):
        rng = np.random.default_rng(np.random.SeedSequence([seed, group, int(split == "diagnostic")]))
        r1 = int(rng.integers(1, max(2, length // 3)))
        r2 = int(rng.integers(r1 + 1, max(r1 + 2, 2 * length // 3)))
        times = np.r_[0, np.cumsum(rng.integers(1, 4, length - 1))].astype(np.int64)
        common = rng.normal(0, .1, (length, 12)).astype(np.float32)
        common[:, :6] = 0.
        # Fields: visible, noisy bit, relevant-to-target, acquired-event/current-event type,
        # target event identity (fixed across branches), occlusion marker.
        # Mask/relevance/type are legal external annotations in THIS synthetic task.
        # They are not state labels and do not reveal the hidden branch.
        common[:, 4] = 1.
        common[:, 5] = 1.
        relevance = np.zeros(length, bool)
        relevance[[r1, r2]] = True
        common[[r1, r2], 0] = 1.
        common[[r1, r2], 2] = 1.
        common[r1, 3] = -1.  # Measures acquisition at original event.
        common[r2, 3] = 1.   # Measures current possession after possible disturbance.
        common[[r1, r2], 5] = 0.
        unrelated = np.zeros(length, bool)
        for t in range(r1 + 1, length - 1):
            if t != r2 and rng.random() < .5:
                unrelated[t] = True
                common[t, 0] = 1.
                common[t, 1] = float(rng.integers(0, 2))
                common[t, 2] = 0.
                common[t, 3] = float(rng.choice([-1, 1]))
                common[t, 5] = 0.
        if flip_unrelated:
            common[unrelated, 1] = 1. - common[unrelated, 1]
        actions = np.zeros((length, 3, 7), np.float32)
        mask = np.zeros((length, 3), bool)
        for t in range(1, length):
            gap = times[t] - times[t - 1]
            mask[t, :gap] = True
            actions[t, :gap] = rng.normal(0, .05, (gap, 7))
            actions[t, :gap, 6] = .5  # Does not leak hidden grasp outcome.
        flips = rng.random(2) < noise
        # Exactly balanced branches share the entire unobserved prefix, exogenous
        # commands/noise, interval lengths and evidence placement.
        truth_acquired = np.asarray([0, 1, 1])
        truth_current = np.asarray([0, 0, 1])
        for branch in range(3):
            x = common.copy()
            observed = np.asarray([truth_acquired[branch], truth_current[branch]]) ^ flips
            x[r1, 1], x[r2, 1] = observed.astype(np.float32)
            y = np.full(length, truth_acquired[branch], np.int64)
            y[r2:] = truth_current[branch]
            # Before both kinds of evidence, confirmation is the defined diagnostic
            # action. Afterwards labels are branch-conditioned; CE should learn
            # uncertainty when evidence is noisy, not pretend it knows the branch.
            decisions = np.zeros(length, np.int64)
            decisions[r2:] = branch + 1
            # Exact posterior under independent binary sensor flips and uniform
            # branch prior; unrelated evidence carries no information by design.
            post = np.full((length, 3), 1/3, np.float64)
            p = np.full(3, 1/3, np.float64)
            for t in range(length):
                if t in (r1, r2):
                    target = truth_acquired if t == r1 else truth_current
                    likelihood = np.where(target == int(x[t, 1]), 1. - noise, noise)
                    p = p * likelihood
                    p = p / p.sum()
                post[t] = p
            all_x.append(x); all_t.append(times); all_a.append(actions); all_m.append(mask)
            all_y.append(y); all_d.append(decisions); all_post.append(post)
            reveals.append((r1, r2)); branches.append(branch); relevances.append(relevance)
            ids.append(f"delayed/{split}/{seed}/{group}/{branch}")
    inputs = {
        "observations": torch.from_numpy(np.stack(all_x)),
        "times": torch.from_numpy(np.stack(all_t)),
        "executed_actions": torch.from_numpy(np.stack(all_a)),
        "executed_mask": torch.from_numpy(np.stack(all_m)),
    }
    labels = {
        "states": torch.from_numpy(np.stack(all_y)),
        "decisions": torch.from_numpy(np.stack(all_d)),
        "bayes_branch": torch.from_numpy(np.stack(all_post)).float(),
        "reveal_indices": torch.tensor(reveals),
        "relevant": torch.from_numpy(np.stack(relevances)),
        "branch": torch.tensor(branches), "identities": ids,
    }
    return inputs, labels
