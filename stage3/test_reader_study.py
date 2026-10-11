"""Safety contracts for the new reader-adaptation factorial."""

import numpy as np
import torch
from torch import nn

from stage3.reader_study_data import assign_partition, episode_key
from stage3.reader_adaptation import ReaderStudyModel
from stage3.refinement import RefinementConfig
from stage3.reader_study_sweep import assignment, VARIANTS, SEEDS

torch.set_num_threads(1)


def test_episode_partition_deterministic_and_all_roles_represented():
    keys = [f"episode-{i}" for i in range(200)]
    roles = [assign_partition(k, 20261002) for k in keys]
    assert set(roles) == {"train", "dev", "test"}
    assert roles == [assign_partition(k, 20261002) for k in keys]
    assert all(0.04 < roles.count(x) / len(roles) < .2 for x in ("dev", "test"))


def test_content_episode_key_does_not_depend_on_enumeration():
    actions = np.ones((32, 7), np.float32)
    language, images = [b"pick"] * 32, [b"encoded_frame"] * 32
    key = episode_key("bridge", actions, language, images)
    assert key == episode_key("bridge", actions.copy(), list(language), list(images))
    changed = actions.copy()
    changed[7, 2] = 5
    assert key != episode_key("bridge", changed, language, images)
    assert key != episode_key("fractal", actions, language, images)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.linear = nn.Linear(8, 8)

    def forward(self, x, per_token):
        return x + self.linear(self.norm1(x)) + per_token.mean(1, keepdim=True)


class TinyNet(nn.Module):
    future_action_window_size = 15
    in_channels = 7

    def __init__(self):
        super().__init__()
        self.action_input = nn.Linear(7, 8)
        self.condition_input = nn.Linear(4, 8)
        self.cognition_input = nn.Linear(20, 8)
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.output = nn.Linear(8, 7)

    def forward(self, x, tau, z, per_token):
        x = self.action_input(x) + self.cognition_input(z)
        c = self.condition_input(per_token)
        for block in self.blocks:
            x = block(x, c)
        return self.output(x)


def make_model(adaptive=True, evidence="future"):
    cfg = RefinementConfig(feature_dim=12, cognition_dim=20, tokens=3, condition_dim=4,
                           r_hidden=16, r_layers=1, r_heads=4, k_hidden=16, k_layers=1, k_heads=4)
    stats = {s: {"mean": torch.zeros(3, 4), "std": torch.ones(3, 4)} for s in ("past", "future")}
    return ReaderStudyModel(TinyNet(), stats, evidence, adaptive, adapter_rank=2, config=cfg)


def batch():
    return {"past_h": torch.randn(2, 3, 12), "past_z": torch.randn(2, 1, 20),
            "past_c": torch.randn(2, 3, 4), "future_c": torch.randn(2, 3, 4),
            "past_actions": torch.randn(2, 16, 7)}


def test_zero_reader_and_condition_adapters_preserve_original_function():
    torch.manual_seed(42)
    base = TinyNet().eval()
    original = {k: v.clone() for k, v in base.state_dict().items()}
    torch.manual_seed(42)
    model = make_model()
    b = batch()
    x, tau = torch.randn(2, 16, 7), torch.rand(2)
    expected = base(x, tau, b["past_z"], b["past_c"])
    actual = model.net(x, tau, b["past_z"], model.condition(b))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for k, v in base.state_dict().items():
        torch.testing.assert_close(v, original[k], rtol=0, atol=0)


def test_adapter_and_k_update_but_original_reader_never_changes():
    model = make_model().train()
    before = {name: p.clone() for name, p in model.base_parameters()}
    adapters = [p.clone() for p in model.adapter_parameters()]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    b = batch()
    for step in range(2):
        opt.zero_grad()
        model(b, 42 + step).backward()
        assert model.probe.k.output.weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapter_parameters())
        opt.step()
    assert not model.net.training
    for name, p in model.base_parameters():
        assert not p.requires_grad and p.grad is None
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    assert any(not torch.equal(p, old) for p, old in zip(model.adapter_parameters(), adapters))


def test_frozen_cell_does_not_train_reader_adapters():
    model = make_model(adaptive=False)
    assert all(not p.requires_grad for p in model.adapter_parameters())
    model(batch(), 42).backward()
    assert all(p.grad is None for p in model.adapter_parameters())


def test_mean_control_cannot_read_future_and_both_cells_match_parameter_counts():
    torch.manual_seed(42)
    mean = make_model(evidence="mean")
    torch.manual_seed(42)
    future = make_model(evidence="future")
    assert sum(p.numel() for p in mean.parameters() if p.requires_grad) == sum(
        p.numel() for p in future.parameters() if p.requires_grad)
    b = batch()
    with torch.no_grad():
        mean.probe.k.output.weight.normal_(std=.1)
    a = mean(b, 42)
    b["future_c"] *= 100
    torch.testing.assert_close(a, mean(b, 42), rtol=0, atol=0)


def test_compact_checkpoint_excludes_base_and_roundtrips():
    model = make_model()
    with torch.no_grad():
        model.probe.k.output.weight.normal_(std=.1)
        for block in model.net.blocks:
            block.adapter.up.weight.normal_(std=.1)
    b = batch()
    saved = model.compact_state()
    assert set(saved) == {"probe", "adapters"}
    expected = model.sample(b, 42)
    model.load_compact_state(saved)
    torch.testing.assert_close(expected, model.sample(b, 42), rtol=0, atol=0)


def test_sweep_assignments_cover_factorial_once_without_ddp():
    jobs = [assignment(rank) for rank in range(16)]
    assert len(set(jobs)) == 16
    assert set(jobs) == {(variant, seed) for variant in VARIANTS for seed in SEEDS}
