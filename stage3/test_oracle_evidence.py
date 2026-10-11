import numpy as np
import torch

from stage3.oracle_evidence import EvidenceProbe, lift_condition
from stage3.refinement import PastRefinement, RefinementConfig
from stage3.prepare_static_future import static_frames


def tiny():
    return RefinementConfig(feature_dim=12, cognition_dim=20, tokens=3, condition_dim=4,
                            r_hidden=16, r_layers=1, r_heads=4,
                            k_hidden=16, k_layers=1, k_heads=4)


def test_lift_preserves_condition_and_nullspace_stays_at_train_mean():
    rng = np.random.default_rng(42)
    w, b = rng.normal(size=(4, 12)), rng.normal(size=4)
    mean, condition = rng.normal(size=(3, 12)), rng.normal(size=(5, 3, 4))
    lifted, error = lift_condition(condition, mean, w, b)
    assert error < 1e-10
    np.testing.assert_allclose(lifted @ w.T + b, condition, rtol=1e-5, atol=1e-6)
    nullspace = np.eye(12) - np.linalg.pinv(w) @ w
    np.testing.assert_allclose((lifted - mean) @ nullspace, 0, atol=1e-6)


def test_condition_probe_matches_original_k_initialization_and_function():
    stats = {side: {"mean": torch.randn(3, 4), "std": torch.ones(3, 4)} for side in ("past", "future")}
    torch.manual_seed(42)
    original = PastRefinement(tiny(), stats, mode="adapter")
    torch.manual_seed(42)
    probe = EvidenceProbe(tiny(), stats, stats["future"]["mean"], stats["future"]["std"])
    for name, value in original.k.state_dict().items():
        torch.testing.assert_close(value, probe.k.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        original.k.output.weight.fill_(.01)
        probe.k.output.weight.fill_(.01)
    h, z, c, f = torch.randn(2, 3, 12), torch.randn(2, 1, 20), torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    torch.testing.assert_close(original.oracle(h, z, c, f), probe(h, z, c, f), rtol=0, atol=0)


def test_evidence_is_detached_and_initial_residual_zero():
    probe = EvidenceProbe(tiny(), None, torch.zeros(3, 12), torch.ones(3, 12))
    h, z, c = torch.randn(2, 3, 12), torch.randn(2, 1, 20), torch.randn(2, 3, 4)
    evidence = torch.randn(2, 3, 12, requires_grad=True)
    output = probe(h, z, c, evidence)
    torch.testing.assert_close(output, c, rtol=0, atol=0)
    output.square().mean().backward()
    assert evidence.grad is None
    assert probe.k.output.weight.grad.abs().sum() > 0


def test_static_evidence_uses_only_current():
    current = object()
    images = static_frames(current)
    assert len(images) == 4 and all(image is current for image in images)
