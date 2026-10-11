import numpy as np
import pytest

from stage3.compare import paired_episode_interval, paired_errors


def test_cluster_interval_constant_improvement_and_reproducibility():
    x = np.full(9, -.125)
    episode = np.repeat([0, 1, 2], [2, 3, 4])
    a = paired_episode_interval(x, episode, draws=100)
    b = paired_episode_interval(x, episode, draws=100)
    assert a == b and a["episodes"] == 3 and a["windows"] == 9
    assert a["ci95_percentile"] == pytest.approx([-.125, -.125])


def test_pairing_rejects_different_episodes_or_targets():
    stats = {"q01": [-1.] * 7, "q99": [1.] * 7, "min": [-1.] * 7,
             "max": [1.] * 7, "mask": [True] * 6 + [False]}
    a = {"indices": np.arange(2), "episode": np.array([1, 2]),
         "timestep": np.array([15, 15]), "target": np.zeros((2, 16, 7)),
         "predictions": np.zeros((2, 16, 7))}
    b = {k: v.copy() for k, v in a.items()}
    errors = paired_errors(a, b, stats)
    assert not errors[0].any() and not errors[1].any()
    b["episode"][0] = 3
    with pytest.raises(ValueError, match="episode"):
        paired_errors(a, b, stats)
