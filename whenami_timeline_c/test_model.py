from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from whenami_timeline_a.model import Config
from whenami_timeline_joint.model import AddressedConditioner
from whenami_timeline_transfer.contracts import Prefix
from whenami_timeline_c.model import TimeQueryConditioner, TimelineStudent, objective
from whenami_timeline_c.data import sample_times


CFG = Config(feature=8, cognition=12, queries=3, hidden=8, heads=2, layers=2, condition=4)


class VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, CFG.cognition)
        self.image = nn.Linear(3, CFG.cognition)

    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        image = self.image(pixel_values["rgb"].mean((-1, -2)))[:, None]
        hidden = torch.cat((image, self.embedding(input_ids) + image), 1)
        return SimpleNamespace(hidden_states=[hidden])


class Local(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Linear(CFG.feature, CFG.feature)
        self.v = nn.Linear(4, CFG.feature)
        self.query = nn.Parameter(torch.randn(CFG.queries, CFG.feature))
        self.v_proj = nn.Linear(CFG.feature, CFG.condition)

    def encode_queries(self, patches, vae):
        return self.query[None] + (self.p(patches.mean(1)) + self.v(vae.mean(1)))[:, None]


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Linear(7, CFG.condition)
        self.z = nn.Linear(CFG.cognition, CFG.condition)
        self.attn = nn.MultiheadAttention(CFG.condition, 2, dropout=0., batch_first=True)
        self.out = nn.Linear(CFG.condition, 7)

    def forward(self, x, tau, z, per_token):
        q = self.x(x) + self.z(z) + tau[:, None, None]
        value = self.attn(q, per_token, per_token, need_weights=False)[0]
        return self.out(q + value)


def prefix(k=2):
    return Prefix(
        pixels={"rgb": torch.randn(2, k, 3, 4, 4)},
        input_ids=torch.randint(0, 32, (2*k, 5)),
        attention_mask=torch.ones(2*k, 5, dtype=torch.bool),
        patches=torch.randn(2, k, 4, CFG.feature),
        vae=torch.randn(2, k, 2, 4),
        times=torch.arange(k)[None].repeat(2, 1) * 3,
    ).validate()


def model(variant):
    m = TimelineStudent(VLM(), Local(), Reader(), variant, future_gap=4, config=CFG)
    nn.init.normal_(m.timeline.output[-1].weight, std=.08)
    return m


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_existing_time_queries_identical(self):
        old = AddressedConditioner(CFG, "timeline")
        nn.init.normal_(old.output[-1].weight, std=.08)
        new = TimeQueryConditioner(CFG, "timeline")
        new.load_state_dict(old.state_dict(), strict=True)
        h, z = torch.randn(2, 3, 3, 8), torch.randn(2, 3, 1, 12)
        times = torch.tensor([[0, 3, 7], [4, 8, 12]])
        visible = torch.tensor([[True, False, True], [True, True, True]])
        cutoffs = torch.tensor([[7, 7, 7], [4, 8, 12]])
        a, b = old(h, z, times, visible, cutoffs), new(h, z, times, visible, cutoffs)
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for key in ("state", "reader_z", "allowed_sources", "residual"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_future_query_has_no_observation(self):
        m, p = model("future_read"), prefix()
        output = m(prefix=p, condition_only=True)
        self.assertTrue(torch.equal(output["target_observed"],
                                    torch.tensor([[True, False], [True, False]])))
        self.assertTrue(torch.equal(output["query_times"], torch.tensor([[3, 7], [3, 7]])))

    def test_sources_after_cutoff_cannot_leak(self):
        m = TimeQueryConditioner(CFG)
        nn.init.normal_(m.output[-1].weight, std=.08)
        h, z = torch.randn(1, 3, 3, 8), torch.randn(1, 3, 1, 12)
        times, query, cut = torch.tensor([[0, 3, 7]]), torch.tensor([[3, 7]]), torch.tensor([[3, 3]])
        visible = torch.ones_like(times, dtype=torch.bool)
        a = m.query(h, z, times, query, visible, cut)
        h[:, 2] += 100
        z[:, 2] -= 70
        b = m.query(h, z, times, query, visible, cut)
        torch.testing.assert_close(a["state"], b["state"], rtol=0, atol=0)

    def test_two_variants_start_identically(self):
        a, b, p = model("future_aux"), model("future_read"), prefix()
        b.load_state_dict(a.state_dict(), strict=True)
        x, tau = torch.randn(4, 16, 7), torch.rand(4)
        aa, bb = a(prefix=p, noisy_actions=x, tau=tau), b(prefix=p, noisy_actions=x, tau=tau)
        torch.testing.assert_close(aa["velocity"], bb["velocity"], rtol=0, atol=0)

    def test_action_gradient_to_predicted_future(self):
        p, x, tau = prefix(), torch.randn(4, 16, 7), torch.rand(4)
        for variant in ("future_aux", "future_read"):
            m = model(variant)
            m.set_step(1000)
            output = m(prefix=p, noisy_actions=x, tau=tau)
            output["future_condition"].retain_grad()
            output["velocity"].square().mean().backward()
            norm = float(output["future_condition"].grad.abs().sum())
            self.assertEqual(norm == 0, variant == "future_aux")

    def test_shared_gradients_and_teacher_detach(self):
        m, p = model("future_read"), prefix()
        m.set_step(1000)
        output = m(prefix=p, noisy_actions=torch.randn(4, 16, 7), tau=torch.rand(4))
        target = torch.randn_like(output["node_conditions"], requires_grad=True)
        loss, metrics = objective(output, torch.randn(4, 16, 7), torch.ones(2, 16, dtype=torch.bool), target)
        loss.backward()
        self.assertIsNone(target.grad)
        for module in (m.vlm, m.local_encoder, m.timeline, m.reader):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))

    def test_student_rejects_three_observed_frames(self):
        with self.assertRaisesRegex(ValueError, "history"):
            model("future_aux")(prefix=prefix(3), condition_only=True)

    def test_duplicate_condition_attention(self):
        reader = Reader()
        x, tau, z, c = torch.randn(2, 16, 7), torch.rand(2), torch.randn(2, 1, 12), torch.randn(2, 3, 4)
        a, b = reader(x, tau, z, c), reader(x, tau, z, torch.cat((c, c), 1))
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_read_schedule_and_state_restore(self):
        a, b = model("future_read"), model("future_read")
        a.set_step(500)
        b.load_state_dict(a.state_dict())
        self.assertEqual(float(b.read_scale), .5)
        aux = model("future_aux")
        aux.set_step(20000)
        self.assertEqual(float(aux.read_scale), 0)

    def test_fixed_future_sampling(self):
        rng = np.random.default_rng(42)
        for length in range(7, 70):
            for _ in range(10):
                ts = sample_times(length, rng, future_gap=4)
                self.assertEqual(ts[2] - ts[1], 4)
                self.assertGreater(ts[1], ts[0])
                self.assertGreaterEqual(ts[0], 0)
                self.assertLessEqual(ts[-1], length - 2)


if __name__ == "__main__":
    unittest.main()
