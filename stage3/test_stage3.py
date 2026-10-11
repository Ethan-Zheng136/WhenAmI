"""CPU tests for semantics/gradient paths; no pretrained or dataset writes."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from stage3.connector import SharedConnector, ConnectorConfig
from stage3.behavior import FrozenBehavior, flow_draw
from stage3.cache import episode_split, assert_episode_disjoint, EndpointCache, FIELDS
from stage3.objectives import connector_inputs, directional_losses
from stage3.common import reference_imports

torch.set_num_threads(1)


def config():
    return ConnectorConfig(feature_dim=12, cognition_dim=20, tokens=3, condition_dim=4, hidden=16, heads=4, layers=2)


def batch():
    torch.manual_seed(10)
    return {
        **{f"{side}_h": torch.randn(2, 3, 12) for side in ("past", "future")},
        **{f"{side}_z": torch.randn(2, 1, 20) for side in ("past", "future")},
        **{f"{side}_c": torch.randn(2, 3, 4) for side in ("past", "future")},
        **{f"{side}_actions": torch.randn(2, 16, 7) for side in ("past", "future")},
    }


class TinyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Linear(4, 7)
        self.cognition = nn.Linear(20, 7)
        self.future_action_window_size = 15
        self.in_channels = 7

    def forward(self, x, tau, z, per_token=None):
        return self.condition(per_token.mean(1))[:, None] + self.cognition(z) + x * 0.1


def readers():
    return {side: FrozenBehavior(TinyVelocity()) for side in ("past", "future")}


def test_direction_outputs_have_target_shape_and_share_trunk():
    r = SharedConnector(config())
    output = r(*connector_inputs(batch(), "bidirectional"))
    assert set(output) == {"forward", "backward"}
    assert all(t.shape == (2, 3, 4) for t in output.values())
    assert len(r.blocks) == 2  # No independent forward/backward transformers.


def test_source_only_api_and_no_opposite_side_leakage():
    r = SharedConnector(config()).eval()
    b = batch()
    a = r(*connector_inputs(b, "bidirectional"))
    b["future_h"] = torch.randn_like(b["future_h"]) * 100
    b["future_z"] = torch.randn_like(b["future_z"]) * 100
    b["future_c"] *= 200
    b["past_actions"] *= 300
    c = r(*connector_inputs(b, "bidirectional"))
    torch.testing.assert_close(a["forward"], c["forward"], rtol=0, atol=0)
    assert not torch.equal(a["backward"], c["backward"])
    assert len(connector_inputs(b, "forward")) == 2


def test_backwards_prediction_does_not_read_past_inputs_or_action_targets():
    r = SharedConnector(config()).eval()
    b = batch()
    p = r(*connector_inputs(b, "bidirectional"))["backward"]
    b["past_h"] *= 50
    b["past_z"] *= 50
    b["future_actions"] *= 200
    torch.testing.assert_close(p, r(*connector_inputs(b, "bidirectional"))["backward"], rtol=0, atol=0)


def test_action_loss_alone_reaches_connector_but_not_frozen_readers():
    r = SharedConnector(config())
    frozen = readers()
    b = batch()
    b["future_c"].requires_grad_(True)
    before = {side: {k: v.clone() for k, v in reader.state_dict().items()} for side, reader in frozen.items()}
    predictions = r(*connector_inputs(b, "bidirectional"))
    loss, _ = directional_losses(r, predictions, b, frozen, 42, alignment_weight=0., action_weight=1.)
    loss.backward()
    assert r.blocks[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert r.input_adapters["past"][1].weight.grad.abs().sum() > 0
    assert r.input_adapters["future"][1].weight.grad.abs().sum() > 0
    for side, reader in frozen.items():
        assert not reader.training and not reader.net.training
        assert all(p.grad is None for p in reader.parameters())
        for name, value in reader.state_dict().items():
            torch.testing.assert_close(value, before[side][name], rtol=0, atol=0)
    assert b["future_c"].grad is None or not b["future_c"].grad.any()


def test_alignment_targets_are_detached():
    r = SharedConnector(config())
    target = torch.randn(2, 3, 4, requires_grad=True)
    pred = torch.randn(2, 3, 4, requires_grad=True)
    r.alignment_loss(pred, target, "future").backward()
    assert pred.grad.abs().sum() > 0 and target.grad is None


def test_reader_stays_eval_and_checkpointing_preserves_input_gradients():
    frozen = FrozenBehavior(TinyVelocity(), checkpoint_blocks=True)
    frozen.train()
    assert not frozen.net.training
    c = torch.randn(2, 3, 4, requires_grad=True)
    frozen.loss(torch.randn(2, 16, 7), torch.randn(2, 1, 20), c, 7).backward()
    assert c.grad.abs().sum() > 0 and all(p.grad is None for p in frozen.parameters())


def test_forward_controls_have_same_draws_independent_of_reverse():
    b = batch()
    r = SharedConnector(config())
    frozen = readers()
    outputs = r(*connector_inputs(b, "bidirectional"))
    _, both = directional_losses(r, outputs, b, frozen, 42)
    _, fwd = directional_losses(r, {"forward": outputs["forward"]}, b, frozen, 42)
    torch.testing.assert_close(both["forward_action_fm"], fwd["forward_action_fm"], rtol=0, atol=0)
    for x, y in zip(flow_draw(b["future_actions"], 42), flow_draw(b["future_actions"], 42)):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_exact_wog_flow_loss_for_explicit_draw():
    reference_imports()
    from action_model.models import DiT
    net = DiT(
        in_channels=7, inj_channels=4, token_size=20, hidden_size=32,
        depth=1, num_heads=4, future_action_window_size=15,
        use_per_attn=True, obs_dropout_prob=0.,
    )
    # Simulate a trained reader, not its initially-zero final output.
    with torch.no_grad():
        net.final_layer.linear.weight.normal_(std=.1)
        net.blocks[0].per_attn.in_proj_weight.normal_(std=.1)
    frozen = FrozenBehavior(net)
    b = batch()
    c = b["future_c"].clone().requires_grad_()
    x, tau, u = flow_draw(b["future_actions"], 41)
    expected = (net(x, tau, b["future_z"], per_token=c) - u).square().mean()
    actual = frozen.loss(b["future_actions"], b["future_z"], c, 41)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert c.grad.abs().sum() > 0


def test_endpoint_ownership_of_z_and_labels():
    r = SharedConnector(config())
    b = batch()
    seen = {}
    class Reader:
        def __init__(self, side):
            self.side = side
        def loss(self, a, z, c, seed):
            seen[self.side] = (a, z)
            return c.square().mean()
    directional_losses(
        r, r(*connector_inputs(b, "bidirectional")), b,
        {s: Reader(s) for s in ("past", "future")}, 17,
    )
    for side in ("past", "future"):
        assert torch.equal(seen[side][0], b[f"{side}_actions"])
        assert torch.equal(seen[side][1], b[f"{side}_z"])


def test_episode_split_and_train_only_statistics(tmp_path):
    ids = np.array(["a"] * 8 + ["b"] * 8)
    episodes = np.tile(np.repeat(np.arange(4), 2), 2)
    train, valid = episode_split(ids, episodes)
    assert_episode_disjoint(ids, episodes, train, valid)
    with pytest.raises(ValueError):
        assert_episode_disjoint(ids, episodes, np.arange(0, 16, 2), np.arange(1, 16, 2))
    manifest = {"protocol": "whenami_stage3_frozen_endpoints_v1", "complete": True, "n": 16}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for name, shape in FIELDS.items():
        arr = np.ones((16, *shape), np.float32)
        arr[valid] = 1000
        np.save(tmp_path / f"{name}.npy", arr)
    for name, array in [
        ("dataset", ids), ("episode", episodes), ("timestep", np.arange(16)),
        ("train_indices", train), ("valid_indices", valid),
    ]:
        np.save(tmp_path / f"{name}.npy", array)
    cache = EndpointCache(tmp_path, check_provenance=False)
    stats = cache.target_statistics()
    assert torch.all(stats["past"]["mean"] == 1)
    assert torch.all(stats["future"]["std"] == 0)


def test_checkpoint_roundtrip_and_nonzero_initial_projection(tmp_path):
    torch.manual_seed(0)
    r = SharedConnector(config()).eval()
    b = batch()
    pred = r(*connector_inputs(b, "bidirectional"))
    path = tmp_path / "connector.pt"
    torch.save({"connector": r.state_dict(), "config": r.config.as_dict()}, path)
    cp = torch.load(path, weights_only=True)
    loaded = SharedConnector(ConnectorConfig(**cp["config"])).eval()
    loaded.load_state_dict(cp["connector"], strict=True)
    for name, tensor in loaded(*connector_inputs(b, "bidirectional")).items():
        torch.testing.assert_close(tensor, pred[name], rtol=0, atol=0)
    pred["forward"].square().mean().backward()
    assert r.input_adapters["past"][1].weight.grad.abs().sum() > 0


def test_forward_only_unused_adapters_frozen_and_shared_trunk_updates():
    r = SharedConnector(config())
    r.freeze_backward_only_adapters()
    assert not r.input_adapters["future"][1].weight.requires_grad
    assert not r.readouts["past"].weight.requires_grad
    before = r.blocks[0].linear1.weight.clone()
    opt = torch.optim.AdamW([p for p in r.parameters() if p.requires_grad], lr=.01)
    b = batch()
    loss = r(*connector_inputs(b, "forward"))["forward"].square().mean()
    loss.backward()
    opt.step()
    assert not torch.equal(before, r.blocks[0].linear1.weight)


def test_small_supervised_task_is_learnable_not_only_nonzero_grad():
    torch.manual_seed(42)
    r = SharedConnector(config())
    b = batch()
    opt = torch.optim.AdamW(r.parameters(), lr=.005)
    def objective():
        p = r(*connector_inputs(b, "bidirectional"))
        return r.alignment_loss(p["forward"], b["future_c"], "future") + r.alignment_loss(
            p["backward"], b["past_c"], "past"
        )
    initial = float(objective().detach())
    for _ in range(45):
        opt.zero_grad()
        loss = objective()
        loss.backward()
        opt.step()
    assert float(objective().detach()) < initial * .4


def _ddp_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        for mode in ("forward", "bidirectional"):
            torch.manual_seed(42)
            model = SharedConnector(config())
            if mode == "forward":
                model.freeze_backward_only_adapters()
            wrapped = DistributedDataParallel(model, find_unused_parameters=False)
            frozen = readers()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            b = batch()
            b["past_h"] += rank
            for step in range(3):
                optimizer.zero_grad()
                predicted = wrapped(*connector_inputs(b, mode))
                loss, _ = directional_losses(model, predicted, b, frozen, seed=step + rank)
                loss.backward()
                optimizer.step()
            value = model.blocks[0].linear1.weight.detach()
            peers = [torch.empty_like(value) for _ in range(2)]
            dist.all_gather(peers, value)
            torch.testing.assert_close(peers[0], peers[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("WAI_TEST_DDP") != "1", reason="opt-in actual 2-process DDP gradient check")
def test_two_process_ddp_forward_and_bidirectional(tmp_path):
    import multiprocessing as mp
    context = mp.get_context("spawn")
    processes = [context.Process(target=_ddp_worker, args=(rank, str(tmp_path / "rdzv"))) for rank in range(2)]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=90)
            assert not process.is_alive(), "DDP test timed out"
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
