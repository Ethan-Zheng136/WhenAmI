"""Causal-input barriers, teacher isolation, paired objectives, and resume state."""

import copy

import pytest
import torch

from stage3.causal_student import CausalStudent, VARIANTS, past_inputs, student_loss, cosine_learning_rate
from stage3.test_reader_study import make_model, batch
from stage3.train_causal_student import trim_logs
from stage3.causal_student_sweep import assignment

torch.set_num_threads(1)


def models():
    torch.manual_seed(42)
    policy = make_model(evidence="mean")
    # Simulate a trained5k mean-policy warm start; nonzero K/readout outputs.
    with torch.no_grad():
        policy.probe.k.output.weight.normal_(std=.03)
        for block in policy.net.blocks:
            block.adapter.up.weight.normal_(std=.02)
    student = CausalStudent(policy)
    teacher = copy.deepcopy(policy)
    teacher.evidence = "future"
    teacher.requires_grad_(False).eval()
    return student, teacher


def test_online_is_invariant_to_future_and_action_label_changes():
    student, _ = models()
    b = batch()
    expected = student(*past_inputs(b))
    sample = student.sample(*past_inputs(b), 42)
    b["future_c"] *= 100
    b["past_actions"] *= 100
    b["future_h"] = torch.randn(2, 3, 12)
    b["future_z"] = torch.randn(2, 1, 20)
    actual = student(*past_inputs(b))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(sample, student.sample(*past_inputs(b), 42), rtol=0, atol=0)


def test_action_only_gradient_reaches_r_k_and_reader_adapter():
    student, teacher = models()
    b = batch()
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
    base_before = {n: p.clone() for n, p in student.policy.base_parameters()}
    loss, _ = student_loss(student, None, b, "latent", 42, draws=4)
    loss.backward()
    assert student.r.input_adapters["past"][1].weight.grad.abs().sum() > 0
    assert student.policy.probe.k.future_input.weight.grad.abs().sum() > 0
    assert student.policy.net.blocks[0].adapter.up.weight.grad.abs().sum() > 0
    for name, p in student.policy.base_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, base_before[name], rtol=0, atol=0)
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)


def test_latent_has_no_future_target_loss_or_gradient():
    student, _ = models()
    b = batch()
    expected, metrics = student_loss(student, None, b, "latent", 42, draws=4)
    b["future_c"] *= 100
    actual, _ = student_loss(student, None, b, "latent", 42, draws=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(metrics) == {"action_fm", "total"}


def test_teacher_is_frozen_detached_but_oracle_can_change_distillation_target():
    student, teacher = models()
    b = batch()
    b["future_c"].requires_grad_(True)
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    loss, metrics = student_loss(student, teacher, b, "distill_future", 42, draws=4)
    loss.backward()
    assert b["future_c"].grad is None
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    for k, v in teacher.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    old = metrics["behavior_distillation"].detach()
    b["future_c"] = b["future_c"].detach() * 100
    _, other = student_loss(student, teacher, b, "distill_future", 42, draws=4)
    assert not torch.equal(old, other["behavior_distillation"].detach())


def test_teacher_uses_same_noisy_actions_times_and_current_z():
    student, teacher = models()
    seen = {}

    def hook(name):
        def callback(module, inputs, kwargs):
            seen[name] = tuple(x.detach().clone() for x in inputs[:3])
        return callback

    a = student.policy.net.register_forward_pre_hook(hook("student"), with_kwargs=True)
    b = teacher.net.register_forward_pre_hook(hook("teacher"), with_kwargs=True)
    try:
        student_loss(student, teacher, batch(), "distill_future", 42, draws=4)
    finally:
        a.remove()
        b.remove()
    assert seen["student"][0].shape == (8, 16, 7)
    for x, y in zip(seen["student"], seen["teacher"]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_all_variants_match_primary_action_draws_and_capacity():
    student, teacher = models()
    b = batch()
    losses = []
    for variant in VARIANTS:
        t = None
        if variant.startswith("distill_"):
            t = copy.deepcopy(teacher)
            t.evidence = "future" if variant == "distill_future" else "mean"
        _, metrics = student_loss(student, t, b, variant, 42, draws=4)
        losses.append(metrics["action_fm"].detach())
    for loss in losses[1:]:
        torch.testing.assert_close(loss, losses[0], rtol=0, atol=0)


def test_wrong_teacher_evidence_and_unfrozen_teacher_rejected():
    student, teacher = models()
    with pytest.raises(ValueError):
        student_loss(student, teacher, batch(), "distill_past", 42)
    teacher.requires_grad_(True)
    with pytest.raises(ValueError):
        student_loss(student, teacher, batch(), "distill_future", 42)


def test_schedule_warmup_peak_floor():
    assert cosine_learning_rate(1, 20000) == 1e-4 / 500
    assert cosine_learning_rate(500, 20000) == 1e-4
    assert abs(cosine_learning_rate(20000, 20000) - 1e-5) < 1e-12
    assert cosine_learning_rate(1000, 20000) > cosine_learning_rate(10000, 20000)


def test_compact_optimizer_resume_reproduces_uninterrupted_updates(tmp_path):
    model, teacher = models()
    initial = copy.deepcopy(model)
    b = batch()

    def optimizer(m):
        return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=.001, weight_decay=0.)

    def advance(m, opt, first, last):
        for step in range(first, last + 1):
            opt.zero_grad(set_to_none=True)
            loss, _ = student_loss(m, teacher, b, "distill_future", 42 + step, draws=2)
            loss.backward()
            opt.step()

    opt = optimizer(model)
    advance(model, opt, 1, 3)
    path = tmp_path / "checkpoint.pt"
    torch.save({"state": model.compact_state(), "optimizer": opt.state_dict()}, path)
    advance(model, opt, 4, 6)
    loaded = torch.load(path, weights_only=True)
    resumed = initial
    resumed.load_compact_state(loaded["state"])
    opt2 = optimizer(resumed)
    opt2.load_state_dict(loaded["optimizer"])
    advance(resumed, opt2, 4, 6)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


def test_resume_trims_only_post_checkpoint_rows_and_partial_tail(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"step": 1, "loss": 1}\n{"step": 2, "loss": 2}\n{"step": 3, "loss": 3}\n{"step":')
    trim_logs(tmp_path, 2)
    assert '"step": 3' not in p.read_text()
    assert len(p.read_text().splitlines()) == 2


def test_sweep_is_four_objectives_by_four_seeds():
    expected = {(v, s) for v in VARIANTS for s in (42, 43, 44, 45)}
    assert {assignment(i) for i in range(16)} == expected
    with pytest.raises(ValueError):
        assignment(16)
