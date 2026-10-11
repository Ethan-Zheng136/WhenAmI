"""Ensure the stronger synthetic task is well specified before training."""

import torch
import pytest

from whenami_revision.delayed_evidence import delayed_groups
from whenami_revision.model import EvidenceTimePolicy


def test_unobserved_prefix_actions_timing_and_nuisance_match_across_branches():
    x, y = delayed_groups(8, 12, 42)
    for i in range(0, 24, 3):
        cutoff = int(y["reveal_indices"][i, 0])
        assert cutoff >= 1
        for j in (i + 1, i + 2):
            for k in ("times", "executed_actions", "executed_mask"):
                assert torch.equal(x[k][i], x[k][j])
            assert torch.equal(x["observations"][i, :cutoff], x["observations"][j, :cutoff])
    assert y["bayes_branch"].shape == (24, 12, 3)
    torch.testing.assert_close(y["bayes_branch"].sum(-1), torch.ones(24, 12))


def test_unrelated_cue_intervention_preserves_relevant_evidence():
    x, y = delayed_groups(8, 24, 42)
    z, q = delayed_groups(8, 24, 42, flip_unrelated=True)
    assert torch.equal(y["bayes_branch"], q["bayes_branch"])
    assert torch.equal(y["states"], q["states"])
    for key in ("times", "executed_actions", "executed_mask"):
        assert torch.equal(x[key], z[key])
    assert torch.equal(x["observations"][y["relevant"]], z["observations"][y["relevant"]])
    different = (x["observations"] != z["observations"])
    assert different.any()
    assert not different[:, :, [0,2,3,4,5,6,7,8,9,10,11]].any()


def test_clean_two_cues_identify_branches_but_no_cue_does_not():
    _, y = delayed_groups(8, 12, 42, noise=0.)
    assert torch.equal(y["bayes_branch"][:, -1].argmax(-1), y["branch"])
    torch.testing.assert_close(y["bayes_branch"][:, 0], torch.full((24, 3), 1/3))
    # Drop does not rewrite acquisition truth.
    dropped = y["branch"] == 1
    assert (y["states"][dropped, 0] == 1).all() and (y["states"][dropped, -1] == 0).all()


def test_legal_interval_counts_and_invalid_specs():
    x, _ = delayed_groups(2, 8, 42)
    assert torch.equal(x["executed_mask"][:, 1:].sum(-1), torch.diff(x["times"], dim=1))
    assert not x["executed_mask"][:, 0].any()
    for kwargs in ({"length":3}, {"noise":.5}, {"split":"sealed"}):
        with pytest.raises(ValueError):
            delayed_groups(**({"groups":2,"length":8,"seed":42} | kwargs))


def test_new_long_prefix_remains_causal_for_both_models():
    x, _ = delayed_groups(2, 12, 42)
    y = {k:v.clone() for k,v in x.items()}
    y["observations"][:, 5:] += 100.
    for mode in ("incremental", "direct_history"):
        model = EvidenceTimePolicy(hidden=16, mode=mode).eval()
        a, b = model(**x), model(**y)
        torch.testing.assert_close(a["decision_history"][:, :5], b["decision_history"][:, :5], rtol=0, atol=0)
