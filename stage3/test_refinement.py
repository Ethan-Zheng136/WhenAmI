"""R+K invariants on CPU; no pretrained weights or dataset writes."""

import torch
from torch import nn
import pytest

from stage3.behavior import FrozenBehavior
from stage3.refinement import (
    RefinementConfig, PastRefinement, online_inputs, refinement_loss,
    differentiable_rollout, repeated_fm_loss,
)

torch.set_num_threads(1)


def config():
    return RefinementConfig(
        feature_dim=12, cognition_dim=20, tokens=3, condition_dim=4,
        r_hidden=16, r_layers=1, r_heads=4, k_hidden=16, k_layers=1, k_heads=4,
    )


def batch():
    generator = torch.Generator().manual_seed(11)
    return {
        **{f"{side}_h": torch.randn(2, 3, 12, generator=generator) for side in ("past", "future")},
        **{f"{side}_z": torch.randn(2, 1, 20, generator=generator) for side in ("past", "future")},
        **{f"{side}_c": torch.randn(2, 3, 4, generator=generator) for side in ("past", "future")},
        **{f"{side}_actions": torch.randn(2, 16, 7, generator=generator) for side in ("past", "future")},
    }


class TinyVelocity(nn.Module):
    future_action_window_size = 15
    in_channels = 7

    def __init__(self):
        super().__init__()
        self.condition = nn.Linear(4, 7)
        self.cognition = nn.Linear(20, 7)

    def forward(self, x, tau, z, per_token):
        return .2 * x + self.condition(per_token.mean(1))[:, None] + self.cognition(z)


def reader():
    return FrozenBehavior(TinyVelocity())


@pytest.mark.parametrize("mode", ["adapter", "predict", "teacher"])
def test_initial_behavior_exactly_past(mode):
    m, b, r = PastRefinement(config(), mode=mode), batch(), reader()
    out = m(*online_inputs(b))
    torch.testing.assert_close(out["online"], b["past_c"], rtol=0, atol=0)
    oracle = m.oracle(*online_inputs(b), b["future_c"])
    torch.testing.assert_close(oracle, b["past_c"], rtol=0, atol=0)
    torch.testing.assert_close(
        r.sample(b["past_z"], out["online"], 42), r.sample(b["past_z"], b["past_c"], 42),
        rtol=0, atol=0,
    )


def test_modes_have_identical_initial_weights():
    states = []
    for mode in ("adapter", "predict", "teacher"):
        torch.manual_seed(42)
        states.append(PastRefinement(config(), mode=mode).state_dict())
    for key in states[0]:
        for other in states[1:]:
            torch.testing.assert_close(states[0][key], other[key], rtol=0, atol=0)


def test_online_never_reads_privilege_or_actions_after_nonzero_k():
    m, b = PastRefinement(config()), batch()
    with torch.no_grad():
        m.k.output.weight.normal_(std=.1)
    expected = m(*online_inputs(b))["online"]
    for key in ("future_h", "future_z", "future_c", "past_actions", "future_actions"):
        b[key] *= 100
    torch.testing.assert_close(expected, m(*online_inputs(b))["online"], rtol=0, atol=0)


def test_oracle_target_is_detached_and_uses_shared_k():
    m, b = PastRefinement(config()), batch()
    with torch.no_grad():
        m.k.output.weight.normal_(std=.1)
    b["future_c"].requires_grad_(True)
    m.oracle(*online_inputs(b), b["future_c"]).square().mean().backward()
    assert b["future_c"].grad is None
    assert m.k.blocks[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.r.parameters())


def test_action_alone_reaches_r_after_first_k_update_not_via_alignment():
    m, b, r = PastRefinement(config(), mode="predict"), batch(), reader()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.002, weight_decay=0.)
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        out = m(*online_inputs(b))
        loss = r.loss(b["past_actions"], b["past_z"], out["online"], 42)
        loss.backward()
        r_grad = sum(float(p.grad.abs().sum()) for p in m.r.parameters() if p.grad is not None)
        assert r_grad == 0 if step == 0 else r_grad > 0
        assert m.k.output.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in r.parameters())
        opt.step()


def test_teacher_objective_averages_actions_does_not_double_scale():
    m, b, r = PastRefinement(config()), batch(), reader()
    loss, values = refinement_loss(m, r, b, 42, alignment_weight=.1)
    expected = .5 * (values["online_action_fm"] + values["oracle_action_fm"]) + .1 * values["future_alignment"]
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert values["delta_rms_standardized"] == 0


def test_adapter_cannot_train_r_or_see_future():
    m, b, r = PastRefinement(config(), mode="adapter"), batch(), reader()
    assert all(not p.requires_grad for p in m.r.parameters())
    loss, values = refinement_loss(m, r, b, 42)
    assert "future_alignment" not in values and "oracle_action_fm" not in values
    loss.backward()
    assert m.k.output.weight.grad.abs().sum() > 0


def test_multi_tau_means_independent_draws_not_sum():
    r, b = reader(), batch()
    actual = repeated_fm_loss(r, b["past_actions"], b["past_z"], b["past_c"], 42, draws=3)
    expected = sum(r.loss(b["past_actions"], b["past_z"], b["past_c"], 42 + j * 104729) for j in range(3)) / 3
    torch.testing.assert_close(actual, expected)


def test_rollout_keeps_full_chain_gradient_and_does_not_update_reader():
    class LinearODE(nn.Module):
        future_action_window_size = 0
        in_channels = 1

        def __init__(self):
            super().__init__()
            self.coefficient = nn.Parameter(torch.tensor(.2))

        def forward(self, x, tau, z, per_token):
            return self.coefficient * x + per_token

    r = FrozenBehavior(LinearODE())
    c = torch.ones(1, 1, 1, requires_grad=True)
    out = differentiable_rollout(r, torch.zeros(1, 1, 1), c, 42, steps=4)
    out.sum().backward()
    # 4 intervals of dx/dt=.2*x+c: total derivative includes all earlier x.
    expected = sum((1 + .2 / 4) ** j / 4 for j in range(4))
    torch.testing.assert_close(c.grad, torch.full_like(c, expected))
    assert r.net.coefficient.grad is None
    # If intermediate states were detached, this gradient would be only .25.
    assert c.grad.item() > 1.


def test_rollout_matches_frozen_sampler_on_same_grid_and_seed():
    r, b = reader(), batch()
    actual = differentiable_rollout(r, b["past_z"], b["past_c"], 42, steps=9)
    expected = r.sample(b["past_z"], b["past_c"], 42, num_steps=10)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_rollout_objective_reaches_r_k_and_freezes_labels():
    m, b, r = PastRefinement(config()), batch(), reader()
    with torch.no_grad():
        m.k.output.weight.normal_(std=.02)
    b["past_actions"].requires_grad_(True)
    loss, metrics = refinement_loss(m, r, b, 42, rollout_steps=4, rollout_weight=.25)
    loss.backward()
    assert metrics["online_rollout_mse"] > 0
    assert m.r.input_adapters["past"][1].weight.grad.abs().sum() > 0
    assert m.k.output.weight.grad.abs().sum() > 0
    assert b["past_actions"].grad is None
    assert all(p.grad is None for p in r.parameters())


def test_checkpoint_roundtrip(tmp_path):
    m, b = PastRefinement(config()), batch()
    with torch.no_grad():
        m.k.output.weight.normal_(std=.02)
    path = tmp_path / "rk.pt"
    torch.save({"config": m.config.as_dict(), "state": m.state_dict()}, path)
    cp = torch.load(path, weights_only=True)
    restored = PastRefinement(RefinementConfig(**cp["config"]))
    restored.load_state_dict(cp["state"], strict=True)
    torch.testing.assert_close(m(*online_inputs(b))["online"], restored(*online_inputs(b))["online"], rtol=0, atol=0)


def test_invalid_sampling_objectives_rejected():
    m, b, r = PastRefinement(config()), batch(), reader()
    with pytest.raises(ValueError):
        refinement_loss(m, r, b, 42, rollout_steps=4, rollout_weight=0)
    with pytest.raises(ValueError):
        repeated_fm_loss(r, b["past_actions"], b["past_z"], b["past_c"], 42, draws=0)
