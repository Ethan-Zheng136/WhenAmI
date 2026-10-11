"""CPU-only tests for the external monitor; no training package imports."""

import json
from pathlib import Path
import tempfile
import unittest

from monitor_stage3 import read_recent, snapshot, render


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        (self.run / "recipe.json").write_text(json.dumps({"settings": {
            "steps": 100, "save_every": 25, "global_batch": 1280,
            "lr": 2e-5, "new_lr": 1e-4,
        }}))

    def log(self, steps):
        (self.run / "train.jsonl").write_text("".join(
            json.dumps({"step": n, "seconds": float(n), "lr": n * 2e-8,
                        "total": .4, "action_fm": .18}) + "\n" for n in steps))

    def test_empty(self):
        self.assertEqual(snapshot(self.run)["step"], 0)

    def test_eta_window(self):
        self.log([1, 2, 3])
        s = snapshot(self.run, window=2)
        self.assertEqual(s["seconds_per_step"], 2.5)
        self.assertEqual(s["remaining_training_seconds"], 97 * 2.5)
        self.assertEqual(s["next_checkpoint_step"], 25)
        self.assertEqual(s["training_seconds_to_next_checkpoint"], 22 * 2.5)
        self.assertAlmostEqual(s["rk_lr_from_schedule"], 3e-7)
        self.assertIn("ETA(train only)", render(s))

    def test_partial_append(self):
        self.log([1, 2])
        with (self.run / "train.jsonl").open("a") as f:
            f.write('{"step": 3')
        self.assertEqual(snapshot(self.run)["step"], 2)

    def test_resumed_log(self):
        self.log([1, 2, 3, 2])
        self.assertEqual([r["step"] for r in read_recent(self.run / "train.jsonl", 100)], [1, 2])

    def test_final_step_not_completion(self):
        self.log([100])
        s = snapshot(self.run)
        self.assertFalse(s["complete"])
        self.assertEqual(s["remaining_training_seconds"], 0)
        self.assertIn("waiting_for_final", s["status"])
        (self.run / "complete.json").write_text('{"complete": true}')
        self.assertTrue(snapshot(self.run)["complete"])

    def test_invalid_duration(self):
        self.log([0])
        with self.assertRaises(ValueError):
            snapshot(self.run)


if __name__ == "__main__":
    unittest.main()
