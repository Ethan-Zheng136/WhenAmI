"""Information contracts/gradients/streaming. No production checkpoint or GPU."""

from dataclasses import replace
import io

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from whenami_revision.contracts import (
    EvidencePrefix, canonical_anchors, revision_pair, current_action_supervision,
)
from whenami_revision.model import EvidenceTimePolicy, RevisionConditionBridge
from whenami_revision.synthetic import paired_branches, prefix
from whenami_revision.raw_data import tensorize


def setup():
    torch.manual_seed(88)
    return EvidenceTimePolicy(hidden=32), paired_branches(2, 42)


def test_future_changes_cannot_change_any_prefix_output():
    model, (x, _) = setup()
    model.eval()
    changed = {k: v.clone() for k, v in x.items()}
    changed["observations"][:, 2:] += 500.
    changed["executed_actions"][:, 2:] -= 500.
    a, b = model(**x), model(**changed)
    torch.testing.assert_close(a["decision_history"][:, :2], b["decision_history"][:, :2], rtol=0, atol=0)
    for i in (0, 1):
        torch.testing.assert_close(a["belief_versions"][i], b["belief_versions"][i], rtol=0, atol=0)


def test_full_replay_equals_streaming_and_separate_prefixes():
    model, (x, _) = setup()
    state = None
    for t in range(4):
        state, out = model.step(x["observations"][:, t], x["times"][:, t],
                                x["executed_actions"][:, t], x["executed_mask"][:, t], state)
        replay = model(**prefix(x, t + 1))
        torch.testing.assert_close(state.revised, replay["bank"], rtol=0, atol=0)
        torch.testing.assert_close(out["decision_logits"], replay["decision_logits"], rtol=0, atol=0)


def test_duplicate_evidence_rejected_not_counted_twice():
    model, (x, _) = setup()
    state, _ = model.step(x["observations"][:, 0], x["times"][:, 0],
                          x["executed_actions"][:, 0], x["executed_mask"][:, 0])
    with pytest.raises(ValueError, match="advance time"):
        model.step(x["observations"][:, 0], x["times"][:, 0],
                   x["executed_actions"][:, 0], x["executed_mask"][:, 0], state)


def test_global_clock_shift_has_no_effect():
    model, (x, _) = setup()
    a = model(**x)
    changed = x | {"times": x["times"] + 100000}
    b = model(**changed)
    torch.testing.assert_close(a["decision_logits"], b["decision_logits"], rtol=0, atol=0)


def test_unavailable_action_padding_cannot_affect_prediction():
    model, (x, _) = setup()
    actions = x["executed_actions"].clone()
    actions[~x["executed_mask"]] = float("nan")
    a = model(**x); b = model(**(x | {"executed_actions": actions}))
    torch.testing.assert_close(a["decision_logits"], b["decision_logits"], rtol=0, atol=0)


def test_wrong_action_interval_and_future_label_kwargs_rejected():
    model, (x, _) = setup()
    mask = x["executed_mask"].clone(); mask[:, 1] = False
    with pytest.raises(ValueError, match="complete intervening"):
        model(**(x | {"executed_mask": mask}))
    with pytest.raises(TypeError):
        model(**x, target_actions=torch.randn(6, 16, 7))
    with pytest.raises(TypeError):
        model(**x, future_observations=x["observations"])
    with pytest.raises(TypeError):
        model(**x, state_labels=torch.ones(6, 4))


def test_action_decision_gradient_reaches_transition_revision_and_encoder():
    model, (x, labels) = setup()
    out = model(**x)
    F.cross_entropy(out["decision_logits"], labels["decisions"][:, -1]).backward()
    for module in (model.observe, model.action_encoder, model.transition, model.filter_update,
                   model.revise, model.decision_head):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads) > 0


def test_unknown_paired_prefix_has_identical_inputs_and_numerically_equal_predictions():
    model, (x, _) = setup()
    out = model(**prefix(x, 1))
    for start in (0, 3):
        for other in (start + 1, start + 2):
            assert torch.equal(x["observations"][start, 0], x["observations"][other, 0])
            # Identical row values can differ in the final FP32 bit in batched
            # matrix kernels; a 1e-6 tolerance is numerical, not a leakage budget.
            torch.testing.assert_close(out["event_logits"][start], out["event_logits"][other],
                                       rtol=0, atol=1e-6)


def test_zero_bridge_preserves_base_and_can_open_gradient_path():
    model, (x, _) = setup()
    bridge = RevisionConditionBridge(hidden=32)
    base = torch.randn(6, 16, 32)
    bank = model(**x)["bank"]
    torch.testing.assert_close(bridge(base, bank), base, rtol=0, atol=0)
    opt = torch.optim.AdamW(list(model.parameters()) + list(bridge.parameters()), lr=.001)
    for _ in range(2):
        opt.zero_grad()
        corrected = bridge(base, model(**x)["bank"])
        corrected.square().mean().backward()
        if _ == 1:
            assert sum(float(p.grad.square().sum()) for p in model.revise.parameters()
                       if p.grad is not None) > 0
        opt.step()


def make_prefix(anchors=(0, 3), cutoff=3):
    images = []
    for i in anchors:
        out = io.BytesIO(); Image.new("RGB", (4, 4), (i, 5, 9)).save(out, format="PNG")
        images.append(out.getvalue())
    return EvidencePrefix("toy", "episode", anchors, cutoff, tuple(images),
                          tuple("pick" for _ in anchors),
                          tuple(np.ones((0 if i == 0 else k - anchors[i-1], 7), np.float32)
                                for i, k in enumerate(anchors)),
                          tuple(() if i == 0 else tuple(range(anchors[i-1], k))
                                for i, k in enumerate(anchors))).validate()


def test_canonical_anchors_remove_duplicates_and_unavailable_frames():
    assert canonical_anchors((0, 0, 3, 7, 11), 7, 14) == (0, 3, 7)
    with pytest.raises(ValueError): canonical_anchors((0, 4, 3), 4, 10)
    with pytest.raises(ValueError): canonical_anchors((0, 3), 4, 10)
    with pytest.raises(ValueError): canonical_anchors((0, 3.5), 3, 10)
    with pytest.raises(ValueError): canonical_anchors((0, 3), 3.0, 10)


def test_pair_same_event_identity_and_old_evidence_immutable():
    a, b = make_prefix(), make_prefix((0, 3, 7), 7)
    pair = revision_pair(a, b, 0)
    assert pair["event_id"] == "toy/episode/0"
    with pytest.raises(ValueError, match="episode"):
        revision_pair(a, replace(b, episode_key="wrong"), 0)
    with pytest.raises(ValueError, match="evidence changed"):
        revision_pair(a, replace(b, instructions=("bad", "pick", "pick")), 0)


def test_old_action_prediction_from_later_evidence_rejected():
    actions = np.arange(10 * 7, dtype=np.float32).reshape(10, 7)
    with pytest.raises(ValueError, match="old actions"):
        current_action_supervision(actions, 3, 7)
    labels = current_action_supervision(actions, 7, 7)
    assert labels.valid.sum() == 3
    np.testing.assert_array_equal(labels.raw_actions[:3], actions[7:])


def test_policy_tensor_view_does_not_include_labels_or_absolute_clock():
    a = make_prefix((10, 13), 13)
    out = tensorize(a, lambda im: torch.from_numpy(np.array(im, copy=True)).float().mean((0, 1)))
    assert set(out) == {"observations", "times", "executed_actions", "executed_mask"}
    assert out["times"].tolist() == [[0, 3]]
    assert out["executed_mask"].sum().item() == 3
    bad = replace(a, interval_action_times=((), (11, 12, 13)))
    with pytest.raises(ValueError, match="PRECEDING"):
        bad.validate()


def test_direct_history_control_is_same_capacity_and_initial_weights():
    torch.manual_seed(88); a = EvidenceTimePolicy(mode="incremental")
    torch.manual_seed(88); b = EvidenceTimePolicy(mode="direct_history")
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    for name in a.state_dict():
        torch.testing.assert_close(a.state_dict()[name], b.state_dict()[name], rtol=0, atol=0)


def test_direct_history_control_has_same_prefix_safety():
    model = EvidenceTimePolicy(hidden=32, mode="direct_history")
    x, _ = paired_branches(2, 42)
    changed = {k: v.clone() for k, v in x.items()}
    changed["observations"][:, 2:] += 100
    a, b = model(**x), model(**changed)
    torch.testing.assert_close(a["decision_history"][:, :2], b["decision_history"][:, :2], rtol=0, atol=0)
    for i in (0, 1):
        torch.testing.assert_close(a["belief_versions"][i], b["belief_versions"][i], rtol=0, atol=0)
