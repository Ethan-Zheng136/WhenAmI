"""Fast real-module tests, including restart and execution-version semantics."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from whenami_timeline_a.model import Config
from whenami_unified.contracts import Frame, Evidence, Executed
from whenami_timeline_force.model import TimelinePolicy
from whenami_timeline_force.losses import (
    TEACHER_REQUESTS, STUDENT_REQUESTS, requested_labels, draw, fm, current_loss, revision_loss)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = nn.Linear(3, 16)
        self.projector = nn.Linear(16, 16)
        self.llm_backbone = nn.Linear(16, 16)
    def forward(self, input_ids, attention_mask, pixel_values, **kwargs):
        x = pixel_values["rgb"].mean((-1, -2))
        h = self.llm_backbone(self.projector(self.vision_backbone(x)))
        h = h[:, None].expand(-1, input_ids.shape[1], -1)
        return SimpleNamespace(hidden_states=(h,))


class TinyQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 4)
    def encode_queries(self, p, v):
        return self.linear(p+v)


class TinyReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Linear(7, 7)
        self.z = nn.Linear(16, 7)
        self.c = nn.Linear(4, 7)
        self.recorded_z = None
    def forward(self, x, tau, z, per_token):
        self.recorded_z = z
        return self.x(x)+self.z(z)+self.c(per_token.mean(1))[:, None]


def tiny(student=False, variant="revise"):
    c = Config(feature=8, cognition=16, queries=4, hidden=16, heads=4, layers=2, condition=4)
    model = TimelinePolicy(TinyVLM(), TinyQ(), TinyReader(), student, variant, c)
    nn.init.normal_(model.timeline.output[-1].weight, std=.03)
    return model


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.m = tiny()
        frames = tuple(Frame({"rgb": torch.randn(2, 3, 2, 2)}, torch.ones(2, 3, dtype=torch.long),
            torch.ones(2, 3, dtype=torch.bool), torch.randn(2, 4, 8), torch.randn(2, 4, 8)) for _ in range(4))
        self.ev = Evidence(frames, torch.tensor([[0, 2, 5, 7], [1, 3, 6, 9]]),
                           torch.ones(2, 4, dtype=torch.bool))
        self.labels = SimpleNamespace(actions=torch.randn(2, 4, 16, 7),
                                     valid=torch.ones(2, 4, 16, dtype=torch.bool))
        command = torch.randn(2, 3, 12, 7)
        mask = torch.zeros(2, 3, 12, dtype=torch.bool)
        mask[:, 1, :2] = True
        mask[:, 2, :3] = True
        self.ex = Executed(command, mask)

    def test_all_reader_conditions_from_same_node(self):
        actions, valid = requested_labels(self.labels, TEACHER_REQUESTS)
        x, tau, target = draw(actions, 42, 1)
        out = self.m(evidence=self.ev, requests=TEACHER_REQUESTS, noisy_actions=x, tau=tau)
        z = torch.stack([out["node_z_versions"][v][:, i] for i, v in TEACHER_REQUESTS], 1).flatten(0, 1)
        self.assertTrue(torch.equal(self.m.reader.recorded_z, z))
        raw = out["raw_z"][:, 2]
        self.assertFalse(torch.equal(raw, out["node_z_versions"][2][:, 2]))
        state = out["versions"][-1]
        state.retain_grad()
        fm(out["velocity"], target, valid).backward()
        self.assertGreater(float(state.grad.abs().sum()), 0)
        self.assertGreater(float(self.m.node_cognition.readout.weight.grad.abs().sum()), 0)

    def test_future_hidden_and_stable_identity(self):
        out = self.m(evidence=self.ev, condition_only=True)
        frames = list(self.ev.frames)
        f = frames[-1]
        frames[-1] = Frame({k: x+10 for k, x in f.pixels.items()},
                            f.input_ids, f.attention_mask, f.patches-3, f.vae+2)
        changed = self.m(evidence=Evidence(tuple(frames), self.ev.times, self.ev.visible), condition_only=True)
        for v in range(3):
            self.assertTrue(torch.equal(out["node_versions"][v], changed["node_versions"][v]))
            self.assertTrue(torch.equal(out["node_z_versions"][v], changed["node_z_versions"][v]))
        shifted = self.m(evidence=Evidence(self.ev.frames, self.ev.times+700, self.ev.visible), condition_only=True)
        self.assertTrue(torch.equal(out["conditions"], shifted["conditions"]))
        hidden = self.ev.hide(3)
        a = self.m(evidence=hidden, condition_only=True)
        b = self.m(evidence=Evidence(tuple(frames), self.ev.times, hidden.visible), condition_only=True)
        self.assertTrue(torch.equal(a["conditions"], b["conditions"]))
        self.assertTrue(torch.equal(a["current_z"], b["current_z"]))

    def test_student_current_only_and_revision_input_grad(self):
        student = tiny(True)
        extra = student.load_state_dict(self.m.state_dict(), strict=False)
        self.assertTrue(all(k.startswith(("executed_adapter.", "forecast_head.")) for k in extra.missing_keys))
        teacher = self.m.requires_grad_(False)
        with torch.no_grad():
            targets = teacher(evidence=self.ev, condition_only=True)
        actions, valid = requested_labels(self.labels, STUDENT_REQUESTS)
        x, tau, target = draw(actions, 43, 1)
        output = student(evidence=self.ev.prefix(3), executed=self.ex,
                         requests=STUDENT_REQUESTS, noisy_actions=x, tau=tau)
        old = output["revised_states"][-1]
        old.retain_grad()
        loss = fm(output["velocity"], target, valid)
        lc, lz = current_loss(output, targets)
        xn, tn, _ = draw(self.labels.actions[:, 1], 44, 1)
        lr, _ = revision_loss(output, targets, teacher, xn, tn, self.labels.valid[:, 1])
        (loss+.1*lc+.1*lz+.1*lr).backward()
        self.assertGreater(float(old.grad.abs().sum()), 0)
        self.assertGreater(float(student.forecast_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(student.node_cognition.readout.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        with self.assertRaises(ValueError):
            student(evidence=self.ev.prefix(3), executed=self.ex, requests=((1, 2),),
                    noisy_actions=x[:2], tau=tau[:2])
        with self.assertRaises(ValueError):
            teacher(evidence=self.ev, executed=self.ex, condition_only=True)
        with self.assertRaises(ValueError):
            student(evidence=self.ev, executed=self.ex, condition_only=True)

    def test_direct_and_stale_versions(self):
        student = tiny(True)
        prefix = self.ev.prefix(3)
        normal = student(evidence=prefix, executed=self.ex, condition_only=True)
        stale = student(evidence=prefix, executed=self.ex, condition_only=True, stale_current=True)
        self.assertGreater(float((normal["current_z"]-stale["current_z"]).abs().max()), 0)
        direct = tiny(True, "direct")
        direct.load_state_dict(student.state_dict())
        self.assertEqual(set(direct.state_dict()), set(student.state_dict()))
        reconstructed = direct(evidence=prefix, executed=self.ex, condition_only=True)
        self.assertFalse(torch.equal(normal["current_z"], reconstructed["current_z"]))

    def test_full_fp32_adam_next_update_restart(self):
        model = tiny(True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        actions, valid = requested_labels(self.labels, STUDENT_REQUESTS)
        x, tau, target = draw(actions, 42, 1)
        def update(m, optimizer):
            optimizer.zero_grad(set_to_none=True)
            out = m(evidence=self.ev.prefix(3), executed=self.ex, requests=STUDENT_REQUESTS,
                    noisy_actions=x, tau=tau)
            loss = fm(out["velocity"], target, valid)
            loss.backward()
            optimizer.step()
        update(model, opt)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"checkpoint.pt"
            torch.save({"model": model.state_dict(), "adam": opt.state_dict(),
                        "rng": torch.get_rng_state()}, path)
            update(model, opt)
            expected = copy.deepcopy(model.state_dict())
            restored = tiny(True)
            restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3, foreach=False)
            record = torch.load(path, weights_only=True)
            restored.load_state_dict(record["model"])
            restored_opt.load_state_dict(record["adam"])
            torch.set_rng_state(record["rng"])
            update(restored, restored_opt)
            self.assertTrue(all(torch.equal(v, expected[k]) for k, v in restored.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
