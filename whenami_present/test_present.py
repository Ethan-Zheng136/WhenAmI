"""CPU unit tests of present-token modules with real Config dims (no 8B backbone)."""
import torch
from torch import nn
from whenami_timeline_a.model import Config
from whenami_present.model import PresentPolicy, PresentConfig
from whenami_present import losses as L


class FakeLocal(nn.Module):
    def __init__(self, c):
        super().__init__(); self.v_proj = nn.Linear(c.feature, c.condition)


def build(student, **kw):
    c = Config()
    m = PresentPolicy(nn.Linear(1, 1), FakeLocal(c), nn.Linear(1, 1), student, PresentConfig(**kw), c)
    nn.init.normal_(m.timeline.output[-1].weight, std=.02)   # emulate trained A50k (zero-init otherwise)
    m.present.bwd.load_state_dict(m.timeline.state_dict())   # same copy rule as loading.py
    if student:
        m.present.imaginer.updater.load_state_dict(m.timeline.state_dict())
    return m, c


def fake_enc(m, c, b=2, k=4):
    nodes = torch.randn(b, k, c.queries, c.feature)
    z = torch.randn(b, k, 1, c.cognition)
    times = torch.tensor([[0, 3, 7, 12]]*b)[:, :k]
    vis = torch.ones(b, k, dtype=torch.bool)
    versions = m.timeline(nodes, z, times, vis)["versions"]
    return {"nodes": nodes, "z": z, "raw_z": z, "versions": versions, "times": times, "visible": vis}


def test_shapes_and_causality():
    m, c = build(False)
    enc = fake_enc(m, c)
    for p in range(3):
        o = m.locate(enc, p, enc["nodes"][:, p+1:], enc["times"][:, p+1:], enc["visible"][:, p+1:])
        assert o["present"].shape == (2, 1, 16, 1024) and o["condition"].shape == (2, 1, 16, 32)
        assert o["reader_z"].shape == (2, 1, 4096) and o["tau"].shape == (2,) and o["logits"].shape == (2, 8)
        assert 0 < float(o["w_fwd"].min()) < 1
        if p:
            assert o["revised"].shape == (2, p, 16, 1024)
    # fwd branch never sees future: change future nodes -> p_fwd identical
    o1 = m.locate(enc, 1, enc["nodes"][:, 2:], enc["times"][:, 2:], enc["visible"][:, 2:])
    o2 = m.locate(enc, 1, enc["nodes"][:, 2:]+5, enc["times"][:, 2:], enc["visible"][:, 2:])
    assert torch.equal(o1["p_fwd"], o2["p_fwd"]) and not torch.equal(o1["present"], o2["present"])
    # dropped future -> fused == fwd exactly
    o3 = m.locate(enc, 1, enc["nodes"][:, 2:], enc["times"][:, 2:], torch.zeros(2, 2, dtype=torch.bool))
    assert torch.allclose(o3["present"].float(), o3["p_fwd"].float()) and float(o3["w_fwd"].min()) == 1


def test_raw_z_flag_and_fusion_modes():
    m, c = build(False, raw_z_bypass=True)
    enc = fake_enc(m, c)
    o = m.locate(enc, 2)
    assert torch.equal(o["reader_z"], enc["raw_z"][:, 2])
    m2, _ = build(False, fusion="mean")
    o = m2.locate(enc, 1, enc["nodes"][:, 2:], enc["times"][:, 2:], enc["visible"][:, 2:])
    assert torch.allclose(o["w_fwd"], torch.full_like(o["w_fwd"], .5))
    m3, _ = build(False, backward=False)
    o = m3.locate(enc, 1, enc["nodes"][:, 2:], enc["times"][:, 2:], enc["visible"][:, 2:])
    assert "p_bwd" not in o


def test_imaginer_student_and_losses_backward():
    m, c = build(True)
    enc = fake_enc(m, c, k=3)
    outs = []
    for p in range(3):
        ft = enc["times"][:, p:p+1] + torch.tensor([[4, 8, 12]])
        fut = m.imagine(enc, p, ft)
        assert fut.shape == (2, 3, 16, 1024)
        outs.append(m.locate(enc, p, fut, ft, torch.ones(2, 3, dtype=torch.bool)))
    loss = sum(L.precision_nll(o, o["p_fwd"].detach()) + L.agreement(o)[0] for o in outs)
    loss = loss + L.monotonic(torch.stack([o["tau"] for o in outs], 1)) + sum(L.phase_alignment(outs))
    loss.backward()
    assert m.present.imaginer.query.grad is not None and m.present.embed.grad is not None
    assert m.present.precision_bwd.net[-1].bias.grad.abs().sum() > 0


def test_precision_not_uniform_under_calibration():
    """NLL drives log-variance heads to track per-branch error (weights leave 0.5)."""
    m, c = build(False, bwd_logvar_init=0.)
    with torch.no_grad():
        enc = fake_enc(m, c)
    opt = torch.optim.Adam(list(m.present.precision_fwd.parameters())+list(m.present.precision_bwd.parameters()), 1e-2)
    for _ in range(60):
        o = m.locate(enc, 1, enc["nodes"][:, 2:], enc["times"][:, 2:], enc["visible"][:, 2:])
        target = o["p_fwd"].detach() + .01*torch.randn_like(o["p_fwd"])   # fwd is the accurate branch
        opt.zero_grad(); L.precision_nll(o, target).backward(); opt.step()
    assert float(o["w_fwd"].mean()) > .8


def test_param_budget():
    m, _ = build(True)
    new = sum(p.numel() for n, p in m.named_parameters() if n.startswith(("present.", "node_cognition.", "executed_adapter.")))
    assert new < 100e6, new
    print("NEW_PARAMS_STUDENT", new)


def test_sinkhorn_no_underflow_nan():
    """Regression for job 30368: collapsed phase usage made linear-domain Sinkhorn return NaN."""
    def old(logits, iters=3, eps=.05):
        q = torch.exp((logits - logits.max()) / eps).T
        q /= q.sum(); k, n = q.shape
        for _ in range(iters):
            q /= q.sum(1, keepdim=True); q /= k; q /= q.sum(0, keepdim=True); q /= n
        return (q*n).T
    l = torch.full((192, 8), -10.); l[:, 0] = 10; l[:, 1] = 5          # 64 ranks x 3 presents, collapsed
    assert torch.isnan(old(l)).any()                                     # reproduces the failure
    q = L.sinkhorn(l)
    assert torch.isfinite(q).all() and torch.allclose(q.sum(1), torch.ones(192), atol=1e-4)
    assert float(q.sum(0).min()) > 192/8*.5                              # still equipartitions
    lf = torch.randn(64, 3, 8, requires_grad=True)
    outs = [{"logits_fwd": lf[:, i]*10, "logits_bwd": lf[:, i]*10+l[:64], "logits": lf[:, i],
             "usable": torch.ones(64, dtype=torch.bool)} for i in range(3)]
    swap, order = L.phase_alignment(outs)
    assert torch.isfinite(swap) and torch.isfinite(order)
    swap.backward(); assert torch.isfinite(lf.grad).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
