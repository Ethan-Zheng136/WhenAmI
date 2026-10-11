"""Extra read-only audit tests, outside recipe-hashed training packages."""

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from stage3_scale.model import FrozenVisualTargets, objective
from stage3_scale.test_joint import fixture
from stage3.behavior import flow_draw
from stage3_perf.runtime import topology


class VisualProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.calls = []

    def forward(self, flat, current, video_pixel_values=None):
        b = current["rgb"].shape[0]
        frames = flat["rgb"].reshape(b, -1, 1)
        video = (video_pixel_values["rgb"] if video_pixel_values is not None
                 else torch.cat([current["rgb"][:, None], frames], dim=1))
        self.calls.append(video.detach().clone())
        return flat["rgb"][:, None].expand(-1, 64, 2) * self.weight, video.mean(1)[:, None]


class FutureProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, patches, vae):
        return (patches.mean(1)[:, None] + vae) * self.weight


def test_history_chronology_and_no_future_contamination():
    encoder = VisualProbe()
    visual = FrozenVisualTargets(encoder, FutureProbe())
    visual.train()
    assert not visual.training and all(not p.requires_grad for p in visual.parameters())
    past = {"rgb": torch.tensor([[[-15.], [-11.], [-7.], [-3.]]])}
    current = {"rgb": torch.tensor([[0.]])}
    future = {"rgb": torch.tensor([[[3.], [7.], [11.], [15.]]])}
    first = visual(current, past, future)
    assert encoder.calls[0].flatten().tolist() == [-15., -11., -7., -3., 0.]
    assert encoder.calls[1].flatten().tolist() == [0., 3., 7., 11., 15.]
    changed = visual(current, past, {"rgb": future["rgb"] + 100.})
    for key in ("history_features", "history_vae"):
        torch.testing.assert_close(first[key], changed[key], rtol=0, atol=0)
    assert not torch.equal(first["future_condition"], changed["future_condition"])
    assert all(not v.requires_grad for v in changed.values())


def test_action_only_gradient_reaches_r_k_and_past():
    model, batch = fixture()
    actions = torch.randn(2, 16, 7)
    x, tau, _ = flow_draw(actions.repeat_interleave(4, 0), 77)
    out = model(**batch, noisy_actions=x, tau=tau)
    loss, _ = objective(out, actions, 77, draws=4, alignment_weight=0)
    loss.backward()
    for part in (model.vlm, model.past_qformer, model.r, model.k, model.net):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.square().sum()) for g in grads) > 0


def test_alignment_gradient_reaches_past_vlm_but_not_k_dit_or_target():
    model, batch = fixture()
    out = model(**batch, condition_only=True)
    target = torch.randn_like(out["future_condition"], requires_grad=True)
    loss = ((out["future_condition"] - target.detach()) / model.r.future_scale).square().mean()
    loss.backward()
    for part in (model.vlm, model.past_qformer, model.r):
        assert sum(float(p.grad.square().sum()) for p in part.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for part in (model.k, model.net) for p in part.parameters())
    assert target.grad is None


def test_active_graph_does_not_train_reverse_adapters():
    model, _ = fixture()
    assert all(not p.requires_grad for part in (
        model.r.input_adapters["future"], model.r.task_adapters["future"], model.r.readouts["past"],
    ) for p in part.parameters())
    # This is forecast + correction, NOT a secretly active backward reconstruction loss.
    assert model.r.readouts["future"].weight.requires_grad


def _reference_worker(rank, rendezvous, output, sharding):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    os.environ["LOCAL_WORLD_SIZE"] = "2"
    strategy, groups = topology(sharding, torch.device("cpu"))

    def network():
        torch.manual_seed(440)
        return nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 2))

    reference = network()
    distributed = FSDP(network(), device_id=torch.device("cpu"), use_orig_params=True,
                       sharding_strategy=strategy, process_group=groups)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=0.)
    opt = torch.optim.AdamW(distributed.parameters(), lr=.001, weight_decay=0.)
    # Same union of rank-local samples and same optimizer update, with clipping.
    torch.manual_seed(123)
    all_x = torch.randn(4, 3, 8) + torch.arange(4)[:, None, None]
    all_y = torch.randn(4, 3, 2)
    norms = []
    for _ in range(2):
        ref_opt.zero_grad(); opt.zero_grad()
        (reference(all_x.flatten(0, 1)) - all_y.flatten(0, 1)).square().mean().backward()
        ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .1)
        (distributed(all_x[rank]) - all_y[rank]).square().mean().backward()
        norm = distributed.clip_grad_norm_(.1)
        torch.testing.assert_close(norm, ref_norm, rtol=2e-6, atol=2e-7)
        ref_opt.step(); opt.step()
        with FSDP.summon_full_params(distributed, recurse=True):
            for a, b in zip(distributed.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
        norms.append(float(norm))
    if rank == 0:
        Path(output).write_text(json.dumps({
            "passed": True, "world": 4, "shard_width": 2, "updates": 2,
            "hybrid_gradient_average_and_clip_match_unsharded_global_batch": True,
            "grad_norms": norms, "device": "cpu", "precision": "fp32", "sharding": sharding,
        }, indent=2))
    dist.destroy_process_group()


@pytest.mark.parametrize("sharding", ["hybrid", "hybrid_zero2"])
def test_hybrid_gradient_averaging_matches_global_reference(tmp_path, sharding):
    import multiprocessing as mp
    context = mp.get_context("spawn")
    workers = [context.Process(target=_reference_worker, args=(
        rank, str(tmp_path / "rendezvous"), str(tmp_path / "report.json"), sharding,
    )) for rank in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(90)
        if worker.is_alive():
            worker.terminate()
            worker.join()
        assert worker.exitcode == 0
    assert json.loads((tmp_path / "report.json").read_text())["passed"]
