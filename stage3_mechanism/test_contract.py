import unittest
import numpy as np
import torch
from torch import nn
from types import SimpleNamespace

from stage3_mechanism.interventions import capture_condition_path, condition_from_path
from stage3_mechanism.evaluate import balanced_order
from stage3_oxe_eval50.evaluate import initial_noise


class ThreeInput(nn.Module):
    def forward(self, h, z, f):
        return h + z + f


class Tests(unittest.TestCase):
    def test_condition_identity_and_no_k(self):
        cp, delta, scale = torch.ones(1, 16, 32), torch.ones(1, 16, 32)*2, torch.ones(16, 32)*3
        self.assertTrue(torch.equal(condition_from_path(cp, delta, scale), cp+delta*scale))
        self.assertIs(condition_from_path(cp, delta, scale, False), cp)

    def test_capture_only_observes_and_handles_removed(self):
        projection, k = nn.Identity(), ThreeInput()
        p = SimpleNamespace(policy=SimpleNamespace(past_qformer=SimpleNamespace(v_proj=projection), k=k))
        x = torch.ones(1)
        with capture_condition_path(p) as values:
            out = projection(x)
            answer = k(x, x*2, x*3)
        self.assertIs(values["cp"], out)
        self.assertTrue(torch.equal(answer, x*6))
        self.assertTrue(torch.equal(values["standardized_future"], x*3))
        self.assertEqual(len(k._forward_hooks), 0)
        self.assertEqual(len(projection._forward_hooks), 0)

    def test_domain_round_robin_no_sample_loss(self):
        rows = [{"dataset": d, "id": i} for d in ("a", "b", "c") for i in range(3)]
        ordered = balanced_order(rows)
        self.assertEqual([r["dataset"] for r in ordered[:3]], ["a", "b", "c"])
        self.assertEqual(sorted((r["dataset"], r["id"]) for r in rows),
                         sorted((r["dataset"], r["id"]) for r in ordered))

    def test_same_seed_array(self):
        self.assertTrue(torch.equal(initial_noise("d", "e", 7), initial_noise("d", "e", 7)))


if __name__ == "__main__":
    unittest.main()
