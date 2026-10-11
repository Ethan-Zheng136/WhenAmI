"""Raw-joint semantics and actual distributed-checkpoint CPU roundtrip tests."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from stage3_scale.model import JointPolicy, cognition, objective, tokenize_instruction
from stage3.refinement import RefinementConfig
from stage3.behavior import flow_draw

torch.set_num_threads(1)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Embedding(64, 20)
        self.image = nn.Linear(6, 20)

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        x = self.tokens(input_ids) + self.image(pixel_values["rgb"])[:, None]
        patches = torch.zeros(len(x), 2, 20)
        return SimpleNamespace(hidden_states=[torch.cat([x[:, :1], patches, x[:, 1:]], 1)])


class TinyPast(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(12, 12)
        self.vae = nn.Linear(5, 12)
        self.v_proj = nn.Linear(12, 4)

    def encode_queries(self, features, vae):
        return self.encoder(features) + self.vae(vae)


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.z = nn.Linear(20, 7)
        self.c = nn.Linear(4, 7)

    def forward(self, x, tau, z, per_token):
        return .1 * x + self.z(z) + self.c(per_token.mean(1))[:, None]


def fixture():
    torch.manual_seed(42)
    cfg = RefinementConfig(feature_dim=12, cognition_dim=20, tokens=3, condition_dim=4,
                           r_hidden=16, r_layers=1, r_heads=4, k_hidden=16, k_layers=1, k_heads=4)
    stats = {s: {"mean": torch.zeros(3, 4), "std": torch.ones(3, 4)} for s in ("past", "future")}
    model = JointPolicy(TinyVLM(), TinyPast(), TinyDiT(), stats, cfg)
    with torch.no_grad():
        model.k.output.weight.normal_(std=.02)  # simulate trainedK warm start
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool),
        "current": {"rgb": torch.randn(2, 6)},
        "history_features": torch.randn(2, 3, 12),
        "history_vae": torch.randn(2, 3, 5),
    }
    return model, batch


def test_cognition_selects_last_nonpadding_text_after_patch_insertion():
    model, batch = fixture()
    actual = cognition(model.vlm, batch["input_ids"], batch["attention_mask"], batch["current"])
    expected = model.vlm.tokens(torch.tensor([3, 7])) + model.vlm.image(batch["current"]["rgb"])
    torch.testing.assert_close(actual[:, 0], expected)


def test_joint_action_gradient_reaches_vlm_past_r_k_dit_without_auxiliary():
    model, batch = fixture()
    actions = torch.randn(2, 16, 7)
    x, tau, _ = flow_draw(actions.repeat_interleave(2, 0), 42)
    result = model(**batch, noisy_actions=x, tau=tau)
    loss, _ = objective(result, actions, 42, draws=2, alignment_weight=0)
    loss.backward()
    for part in (model.vlm, model.past_qformer, model.r, model.k, model.net):
        assert sum(float(p.grad.abs().sum()) for p in part.parameters() if p.grad is not None) > 0


def test_auxiliary_future_and_teacher_detach_no_student_input_leak():
    model, batch = fixture()
    actions = torch.randn(2, 16, 7)
    x, tau, _ = flow_draw(actions, 42)
    before = model(**batch, noisy_actions=x, tau=tau)
    target = torch.randn(2, 3, 4, requires_grad=True)
    teacher = torch.randn(2, 16, 7, requires_grad=True)
    loss, _ = objective(before, actions, 42, draws=1, future_target=target,
                        future_scale=torch.ones(3, 4), teacher_velocity=teacher)
    loss.backward()
    assert target.grad is None and teacher.grad is None
    after = model(**batch, noisy_actions=x, tau=tau)
    torch.testing.assert_close(before["velocity"], after["velocity"], rtol=0, atol=0)
    with pytest.raises(TypeError):
        model(**batch, noisy_actions=x, tau=tau, future_images=torch.randn(2, 3))


def test_counter_sampler_replays_without_process_rng_state():
    def draws(seed, step, micro, rank):
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, micro, rank]))
        return rng.integers(10**6, size=32)
    assert np.array_equal(draws(42, 5, 0, 0), draws(42, 5, 0, 0))
    assert not np.array_equal(draws(42, 5, 0, 0), draws(42, 5, 0, 1))


def test_encode_then_decode_api_matches_training_forward_in_eval():
    model, batch = fixture()
    model.eval()
    x, tau, _ = flow_draw(torch.randn(2, 16, 7), 9)
    original = model(**batch, noisy_actions=x, tau=tau)["velocity"]
    condition = model(**batch, condition_only=True)
    readout = model(noisy_actions=x, tau=tau, conditioning=condition)["velocity"]
    torch.testing.assert_close(readout, original, rtol=0, atol=0)
    with pytest.raises(ValueError):
        model(**batch, noisy_actions=x, tau=tau, conditioning=condition)


def test_dcp_single_process_model_optimizer_restore(tmp_path):
    from stage3_scale.checkpoint import save, load
    torch.manual_seed(42)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    x = torch.randn(4, 3)
    model(x).square().mean().backward()
    optimizer.step()
    path = tmp_path / "step-1"
    save(path, model, optimizer, 1, {"data": "test", "world": 1})
    expected = copy.deepcopy(model.state_dict())
    restored = nn.Linear(3, 2)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.01)
    assert load(path, restored, opt2, {"data": "test", "world": 1}) == 1
    for key in expected:
        torch.testing.assert_close(restored.state_dict()[key], expected[key], rtol=0, atol=0)
    for m, opt in [(model, optimizer), (restored, opt2)]:
        opt.zero_grad()
        m(x).square().mean().backward()
        opt.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        load(path, restored, opt2, {"data": "changed", "world": 1})


def _distributed_checkpoint_worker(rank, rendezvous, root):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from stage3_scale.checkpoint import save, load
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    torch.manual_seed(42)
    model = DDP(nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 2)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    x = torch.randn(2, 3) + rank
    model(x).square().mean().backward()
    optimizer.step()
    save(Path(root) / "step-1", model, optimizer, 1, {"data": "paired", "world": 2})
    restored = DDP(nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 2)))
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.01)
    assert load(Path(root) / "step-1", restored, opt2, {"data": "paired", "world": 2}) == 1
    for m, opt in [(model, optimizer), (restored, opt2)]:
        opt.zero_grad()
        m(x).square().mean().backward()
        opt.step()
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    dist.destroy_process_group()


def test_actual_two_process_dcp_restore(tmp_path):
    import multiprocessing as mp
    context = mp.get_context("spawn")
    workers = [
        context.Process(target=_distributed_checkpoint_worker,
                        args=(rank, str(tmp_path / "rendezvous"), str(tmp_path)))
        for rank in range(2)
    ]
    for p in workers:
        p.start()
    for p in workers:
        p.join(90)
        if p.is_alive():
            p.terminate()
            p.join()
        assert p.exitcode == 0
