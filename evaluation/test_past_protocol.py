import numpy as np
import pytest

from past_protocol import (
    action_metrics, condition_frames, interior_centres, normalize_actions,
    ordered_predictions, sample_seed,
)

PAST = [-15, -11, -7, -3, 0]
OFFSETS = PAST + [3, 7, 11, 15]
STATS = {"q01": [-2.] * 6 + [0.], "q99": [2.] * 6 + [1.],
         "min": [-3.] * 6 + [0.], "max": [3.] * 6 + [1.], "mask": [True] * 6 + [False]}


def test_windows_are_raw_timesteps_and_never_cross_episode_boundaries():
    assert interior_centres(30, PAST) == []
    assert interior_centres(31, PAST) == [15]
    for t in interior_centres(80, PAST):
        assert t - 15 >= 0 and t + 15 < 80
    assert len(interior_centres(200, PAST, per_episode=4)) == 4


@pytest.mark.parametrize("mode,expected,key", [
    ("past", [-15, -11, -7, -3], "past_frames"),
    ("past_static", [0, 0, 0, 0], "past_frames"),
    ("future_oracle", [3, 7, 11, 15], "cond_frames"),
])
def test_exact_model_inputs(mode, expected, key):
    now, kwargs = condition_frames(OFFSETS, OFFSETS, mode, PAST)
    assert now == 0 and kwargs == {key: expected}
    assert "cond_actions" not in kwargs


def test_seed_is_independent_of_rank_partition():
    samples = [(i // 4, i * 3) for i in range(12)]
    sequential = [sample_seed("fractal", e, t, 42) for e, t in samples]
    by_rank = {}
    for rank in range(5):
        for i in range(rank, len(samples), 5):
            by_rank[i] = sample_seed("fractal", *samples[i], 42)
    assert sequential == [by_rank[i] for i in range(12)]
    assert sample_seed("bridge", 0, 0, 42) != sample_seed("fractal", 0, 0, 42)


def test_normalization_preserves_standardized_absolute_gripper():
    raw = np.array([[0., 2., -2., 4., -4., 1., 1.]])
    actual = normalize_actions(raw, STATS)
    np.testing.assert_allclose(actual, [[0, 1, -1, 1, -1, 0.5, 1]], atol=1e-7)


def test_perfect_prediction_metrics():
    target = np.random.default_rng(0).normal(size=(8, 16, 7))
    target[:, :, 6] = target[:, :, 6] > 0
    m = action_metrics(target, target, STATS)
    assert m["r_6"] == pytest.approx(1.)
    assert m["r2_6"] == 1 and m["normalized_mse_6"] == 0 and m["gripper_accuracy"] == 1


def test_degenerate_targets_do_not_produce_nan_json():
    m = action_metrics(np.zeros((2, 16, 7)), np.zeros((2, 16, 7)), STATS)
    assert m["r_6"] is None and m["r2_6"] is None


def test_metric_rejects_invalid_shape_or_nonfinite():
    target = np.zeros((2, 16, 7))
    with pytest.raises(ValueError):
        action_metrics(target[:, :15], target, STATS)
    pred = target.copy()
    pred[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        action_metrics(pred, target, STATS)


def test_distributed_reassembly_with_empty_ranks_and_duplicate_detection():
    shards = [
        {"indices": np.array([1]), "predictions": np.ones((1, 1, 16, 7))},
        {"indices": np.array([], dtype=int), "predictions": np.empty((0, 1, 16, 7))},
        {"indices": np.array([0]), "predictions": np.zeros((1, 1, 16, 7))},
    ]
    p = ordered_predictions(shards, 2)
    assert not p[0].any() and p[1].all()
    with pytest.raises(ValueError):
        ordered_predictions(shards + [shards[0]], 2)
