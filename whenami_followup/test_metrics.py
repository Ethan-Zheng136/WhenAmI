import unittest
import numpy as np

from whenami_followup.metrics import score, aggregate, paired


class Tests(unittest.TestCase):
    def setUp(self):
        self.stats = {"min": [0.] * 7, "max": [1.] * 7}

    def test_perfect(self):
        target = np.zeros((16, 7), np.float32)
        result = score(np.stack([target] * 3), target, np.ones(16, bool), self.stats)
        np.testing.assert_array_equal(result["mse6"], np.zeros(3))
        np.testing.assert_array_equal(result["gripper_accuracy"], np.ones(3))

    def test_masked_tail(self):
        pred = np.ones((1, 16, 7), np.float32)
        pred[:, 0] = 0
        mask = np.zeros(16, bool)
        mask[0] = True
        result = score(pred, np.zeros((16, 7)), mask, self.stats)
        self.assertEqual(float(result["mse6"][0]), 0.)
        self.assertEqual(float(result["gripper_accuracy"][0]), 1.)

    def test_first_action_separate(self):
        pred = np.zeros((1, 16, 7), np.float32)
        pred[0, 0, 0] = 1.
        result = score(pred, np.zeros((16, 7)), np.ones(16, bool), self.stats)
        self.assertAlmostEqual(float(result["first_action_mse6"][0]), 1 / 6)
        self.assertAlmostEqual(float(result["mse6"][0]), 1 / 96)

    def test_domain_macro(self):
        names = ("mse6", "unclipped_mse6", "first_action_mse6", "gripper_accuracy")
        rows = [{"dataset": "a", **dict.fromkeys(names, 4.)}]
        rows.extend({"dataset": "b", **dict.fromkeys(names, 1.)} for _ in range(3))
        self.assertEqual(aggregate(rows)["macro"]["mse6"], 2.5)

    def test_paired_direction(self):
        result = paired([2., 4.], [1., 2.], ["a", "b"])
        self.assertEqual(result["improvement"], 50.)
        self.assertEqual(result["domains_favorable"], 2)

    def test_pp_not_relative_gripper(self):
        result = paired([.8], [.9], ["a"], larger=True)
        self.assertAlmostEqual(result["improvement"], 10.)
        self.assertEqual(result["unit"], "percentage_points")


if __name__ == "__main__":
    unittest.main()
