"""OXE mixture, boundary masks, and checkpoint-v2 regression tests."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from stage3_scale.oxe_inventory import original_mixture, sampling_probabilities, metadata_files
from stage3_scale.oxe_data import frame_indices, action_window, valid_centres
from stage3_scale.model import objective
from stage3.behavior import flow_draw
from stage3_scale import checkpoint_v2

torch.set_num_threads(1)


def test_selected_mixture_has_all_original27():
    pairs = original_mixture()
    assert len(pairs) == 27 and "kuka" in dict(pairs) and "droid" in dict(pairs)


def test_sampling_reproduces_weight_times_transitions_not_uniform():
    pairs = [("large", .1), ("small", 2.)]
    probs = sampling_probabilities(pairs, {"large": {"num_transitions": 1000},
                                         "small": {"num_transitions": 100}})
    assert probs == {"large": 1 / 3, "small": 2 / 3}


def test_metadata_selects_only_one_physical_shard_layout(tmp_path):
    info = {"name": "dobbe", "splits": [{"name": "train", "shardLengths": ["1", "2"]}]}
    selected = metadata_files(tmp_path, info)
    assert [p.name for p in selected] == ["dobbe-train.tfrecord-00000-of-00002", "dobbe-train.tfrecord-00001-of-00002"]
    assert all("01016" not in str(p) for p in selected)


def test_short_episode_is_kept_without_cross_episode_or_fake_action_targets():
    actions = np.ones((9, 7), np.float32)
    images = [b"image"] * 9
    language = [b"pick"] * 9
    assert valid_centres(actions, images, language) == list(range(8))
    assert frame_indices(0, 9) == [0, 0, 0, 0, 0, 3, 7, 8, 8]
    a, mask = action_window(actions, 7)
    assert mask.sum() == 2 and a.shape == (16, 7)
    assert np.all(a[2:, :6] == 0) and np.all(a[2:, 6] == 1)
    with pytest.raises(ValueError):
        frame_indices(8, 9)


def test_padded_action_errors_do_not_contribute_to_loss():
    actions = torch.randn(2, 16, 7)
    _, _, target = flow_draw(actions, 42)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, :2], mask[1, :5] = True, True
    pred = target.clone()
    pred[~mask] += 100
    out = {"velocity": pred}
    loss, _ = objective(out, actions, 42, draws=1, action_mask=mask)
    assert loss == 0
    pred[mask] += 1
    loss, _ = objective(out, actions, 42, draws=1, action_mask=mask)
    torch.testing.assert_close(loss, torch.tensor(1.))


def test_v2_checkpoint_preserves_empty_parameter_and_optimizer_groups(tmp_path):
    class EmptyParameters(nn.Module):
        def __init__(self):
            super().__init__()
            self.empty = nn.Parameter(torch.empty(0))
            self.linear = nn.Linear(3, 2)

        def forward(self, x):
            return self.linear(x)
    torch.manual_seed(42)
    model = EmptyParameters()
    optimizer = torch.optim.AdamW([{"params": [model.empty]}, {"params": list(model.linear.parameters())}], lr=.001)
    x = torch.randn(4, 3)
    model(x).square().mean().backward()
    optimizer.step()
    checkpoint_v2.save(tmp_path / "checkpoint", model, optimizer, 1, {"test": True})
    restored = EmptyParameters()
    opt2 = torch.optim.AdamW([{"params": [restored.empty]}, {"params": list(restored.linear.parameters())}], lr=.001)
    assert checkpoint_v2.load(tmp_path / "checkpoint", restored, opt2, {"test": True}) == 1
    for m, opt in ((model, optimizer), (restored, opt2)):
        opt.zero_grad()
        m(x).square().mean().backward()
        opt.step()
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def _fsdp_cpu_worker(rank, rendezvous, output):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    torch.manual_seed(42)
    def build():
        return FSDP(nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 1)),
                    device_id=torch.device("cpu"), use_orig_params=True,
                    sharding_strategy=ShardingStrategy.FULL_SHARD)
    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    x = torch.randn(4, 8) + rank
    def update(m, o):
        o.zero_grad()
        m(x).square().mean().backward()
        m.clip_grad_norm_(1.)
        o.step()
    update(model, opt)
    checkpoint_v2.save(Path(output) / "fsdp", model, opt, 1, {"true_fsdp_cpu": True})
    restored = build()
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.001)
    assert checkpoint_v2.load(Path(output) / "fsdp", restored, opt2, {"true_fsdp_cpu": True}) == 1
    update(model, opt)
    update(restored, opt2)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    empties = sum(p.numel() == 0 for p in model.parameters())
    totals = [None] * 4
    dist.all_gather_object(totals, empties)
    assert any(n > 0 for n in totals)
    if rank == 0:
        Path(output, "fsdp_cpu_audit.json").write_text(json.dumps(
            {"complete": True, "empty_local_parameters_per_rank": totals, "next_update_bitwise_equal": True}))
    dist.destroy_process_group()


def test_four_rank_cpu_fsdp_real_empty_local_shards_restore(tmp_path):
    import multiprocessing as mp
    context = mp.get_context("spawn")
    workers = [context.Process(target=_fsdp_cpu_worker,
                               args=(r, str(tmp_path / "rendezvous"), str(tmp_path))) for r in range(4)]
    for p in workers:
        p.start()
    for p in workers:
        p.join(120)
        if p.is_alive():
            p.terminate()
            p.join()
        assert p.exitcode == 0
    assert json.loads((tmp_path / "fsdp_cpu_audit.json").read_text())["complete"]
