"""Small deterministic tests of diagnostic semantics; no large-model loads."""

from types import SimpleNamespace

import numpy as np
import torch

from stage3.diagnose import donor_indices, gradient_relation, shared_parameters
from stage3.connector import ConnectorConfig, SharedConnector
from stage3.evaluate_wog_stage2 import current_image_array
from stage3.behavior import FrozenBehavior


def test_donors_are_training_only_different_episode_same_dataset():
    cache = SimpleNamespace(
        datasets=np.array(["a", "a", "a", "b", "b", "b"]),
        episodes=np.array([0, 1, 2, 0, 1, 2]),
        train_indices=np.array([0, 1, 3, 4]),
    )
    donors, matched = donor_indices(cache, ["x", "y", "y", "z", "q", "none"], [2, 5], 42)
    assert donors[0] == 1 and matched.tolist() == [True, False]
    assert donors[1] in (3, 4)
    np.testing.assert_array_equal(donors, donor_indices(cache, ["x", "y", "y", "z", "q", "none"], [2, 5], 42)[0])


def test_gradient_relation_sign_and_scale():
    a = (torch.tensor([1., 2.]),)
    b = (torch.tensor([-2., -4.]),)
    result = gradient_relation(a, b)
    assert abs(result["cosine"] + 1) < 1e-6
    assert abs(result["second_over_first_norm"] - 2) < 1e-6
    assert gradient_relation(a, (torch.zeros(2),))["cosine"] == 0


def test_shared_gradient_scope_excludes_direction_specific_coordinates():
    cfg = ConnectorConfig(feature_dim=12, cognition_dim=20, tokens=3, condition_dim=4,
                          hidden=16, heads=4, layers=1)
    model = SharedConnector(cfg)
    ids = {id(p) for p in shared_parameters(model)}
    assert id(model.direction) not in ids
    assert id(model.input_adapters["past"][1].weight) not in ids
    assert id(model.blocks[0].linear1.weight) in ids
    assert id(model.query_position) in ids


def test_wog_stage2_selects_only_current_frame():
    frames = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    current = current_image_array(frames, [-3, 0, 3], 1).copy()
    frames[:, [0, 2]] = -999
    np.testing.assert_array_equal(current_image_array(frames, [-3, 0, 3], 1), current)


def test_current_only_paired_sampler_matches_original_euler_algorithm():
    class CurrentOnly(torch.nn.Module):
        future_action_window_size = 15
        in_channels = 7

        def forward(self, x, tau, z, per_token=None):
            assert per_token is None
            return .2 * x + z + tau[:, None, None]

    net = CurrentOnly()
    reader = FrozenBehavior(net)
    z = torch.ones(1, 1, 7)
    actual = reader.sample(z, None, 42)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        expected = torch.randn(1, 16, 7)
        ts = torch.linspace(0, 1, 10)
        for i in range(1, len(ts)):
            expected = expected + net(expected, ts[i - 1].expand(1), z) * (ts[i] - ts[i - 1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
