import numpy as np
import pytest
import torch
from torch import nn

from whenami_timeline.test_timeline import fixture
from whenami_timeline_analysis.evaluate import (
    noise_seed, make_noise, disable_old_revision, sample_actions, per_anchor_metrics,
)


def test_noise_is_identity_based_and_reproducible():
    a = {"dataset": "a", "episode_key": "one", "source_times": [0, 2, 5]}
    b = {"dataset": "b", "episode_key": "two", "source_times": [1, 3, 6]}
    x = make_noise([a, b], [42, 43], "cpu")
    y = make_noise([b, a], [42, 43], "cpu")
    torch.testing.assert_close(x[:6], y[6:], rtol=0, atol=0)
    assert not torch.equal(x[0], x[3])


def test_sampler_has_no_labels_and_integrates_velocity():
    class Reader(nn.Module):
        def forward(self, x, tau, z, per_token):
            return torch.ones_like(x) * per_token[:, :1, :1]
    x = torch.zeros(2, 16, 7)
    y = sample_actions(Reader(), torch.zeros(2, 1, 4), torch.ones(2, 3, 1), x)
    torch.testing.assert_close(y, torch.ones_like(x))
    assert torch.equal(x, torch.zeros_like(x))
    with pytest.raises(TypeError):
        sample_actions(Reader(), torch.zeros(2, 1, 4), torch.ones(2, 3, 1), x,
                       targets=torch.ones_like(x))


def test_mask_metrics_ignore_invalid_actions_and_preserve_first():
    y = np.zeros((2, 3, 16, 7))
    p = np.zeros((2, 3, 3, 16, 7))
    valid = np.zeros((2, 3, 16), dtype=bool); valid[:, :, :2] = True
    p[:, :, :, 2:] = 100
    m = per_anchor_metrics(p, y, valid)
    assert np.max(m["mse6"]) == 0
    p[:, :, :, :2, :6] = 2.
    m = per_anchor_metrics(p, y, valid)
    assert np.all(m["mse6"] == 1.) and np.all(m["unclipped_mse6"] == 4.)
    assert np.all(m["first_action_mse6"] == 1.)


def test_gate_intervention_restores_model_and_first_anchor():
    model, x, _ = fixture()
    a = model(**x)["conditions"]
    with disable_old_revision(model):
        b = model(**x)["conditions"]
    c = model(**x)["conditions"]
    torch.testing.assert_close(a[:, 0], b[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(a, c, rtol=0, atol=0)
    assert (a[:, 1:] - b[:, 1:]).abs().max() > 0
