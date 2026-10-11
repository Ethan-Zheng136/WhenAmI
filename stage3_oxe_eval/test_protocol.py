import unittest
import numpy as np
import torch

from stage3_oxe_eval.prepare import selected_centres, HISTORY
from stage3_oxe_eval.metrics import episode_metrics, summarize_raw
from stage3_oxe_eval.evaluate import initial_noise
from stage3_oxe_eval.summarize import paired


STATS = {"q01": [-1.] * 6 + [0.], "q99": [1.] * 7,
         "min": [-1.] * 6 + [0.], "max": [1.] * 7, "mask": [True] * 6 + [False]}


class ProtocolTests(unittest.TestCase):
    def test_noise_identity(self):
        a = initial_noise("d", "e", 5)
        self.assertTrue(torch.equal(a, initial_noise("d", "e", 5)))
        self.assertFalse(torch.equal(a, initial_noise("d", "e", 6)))
        self.assertFalse(torch.equal(a[0], a[1]))
        self.assertEqual(a.dtype, torch.float32)

    def test_centres_causal_and_deterministic(self):
        actions = np.zeros((20, 7), np.float32)
        images = np.asarray([b"x"] * 20)
        texts = np.asarray([b"instruction"] * 20)
        a = selected_centres(actions, images, texts, "episode")
        self.assertEqual(a, selected_centres(actions, images, texts, "episode"))
        self.assertEqual(len(a), 2)
        self.assertEqual(len(a), len(set(a)))
        self.assertTrue(all(max(0, t + off) <= t for t in a for off in HISTORY))

    def test_padding_does_not_score(self):
        y = np.zeros((1, 16, 7))
        p = np.zeros((1, 3, 16, 7))
        m = np.zeros((1, 16), bool); m[:, 0] = True
        p[:, :, 1:, :6] = 999
        a = episode_metrics(p, y, m, STATS, STATS)
        self.assertLess(a["mean"]["mse6"], 1e-14)
        self.assertAlmostEqual(a["mean"]["gripper_accuracy"], 1.)
        self.assertTrue(np.allclose(a["moments"]["se"], 0))

    def test_episode_moments_sum_correctly(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(-.9, .9, (2, 16, 7)); y[..., 6] = 0.
        p = np.repeat(y[:, None], 3, axis=1)
        m = np.ones((2, 16), bool); m[0, 4:] = False
        a = episode_metrics(p, y, m, STATS, STATS)
        raw = summarize_raw([a["moments"]])
        self.assertAlmostEqual(raw["r6"], 1.)
        self.assertAlmostEqual(raw["r2_6"], 1.)
        self.assertLess(a["mean"]["mse6"], 1e-14)

    def test_own_statistics_decoded_before_common_score(self):
        own = dict(STATS, q01=[-2.] * 6 + [0.], q99=[2.] * 6 + [1.])
        y = np.full((1, 16, 7), .4); y[..., 6] = 0.
        p = np.full((1, 3, 16, 7), .2); p[..., 6] = 0.
        a = episode_metrics(p, y, np.ones((1, 16), bool), own, STATS)
        self.assertLess(a["mean"]["mse6"], 1e-14)

    def test_paired_bootstrap_identity_and_known_improvement(self):
        a = np.asarray([1., 2., 3., 4.])
        names = np.asarray(["a", "a", "b", "b"])
        same = paired(a, a, names, repeats=100)
        self.assertEqual(same["relative_reduction_percent"], 0.)
        better = paired(a, a / 2, names, repeats=100)
        self.assertAlmostEqual(better["relative_reduction_percent"], 50.)
        self.assertTrue(np.allclose(better["paired_episode_stratified_95pct_reduction_interval"], 50.))


if __name__ == "__main__":
    unittest.main()
