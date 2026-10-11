"""B50 changes the initialization/teacher, not the original-B learning algorithm."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from stage3.common import BASE
from whenami_pooled_b50 import loading


def test_original_training_loop_preserved():
    old = ast.parse((BASE / "whenami_timeline_b/train.py").read_text())
    new = ast.parse((BASE / "whenami_pooled_b50/train.py").read_text())
    def loop(tree):
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        return next(n for n in main.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                    and n.target.id == "step")
    original, upgraded = loop(old), loop(new)
    # Only final provenance changes inside the update/evaluation/save loop.
    for n in ast.walk(original):
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and k.value == "teacher_fixed_step":
                    v.value = 50000
    assert ast.dump(original) == ast.dump(upgraded)


def test_original_model_data_objective_and_evaluator_reused():
    from whenami_pooled_b50 import train
    from whenami_timeline_b import model, data, evaluate
    from whenami_pooled_b50.model import build_student
    assert train.build_student is build_student
    assert train.build_student is not model.build_student
    assert train.draw is model.draw
    assert train.objective is model.objective
    assert train.batch is data.batch
    assert train.evaluate is evaluate.evaluate


def test_strict_loader_reused_with_A50k_default():
    from whenami_timeline_b.loading import load_model_only
    assert loading._load_model_only is load_model_only
    assert loading.load_model_only.__defaults__[0] == loading.TEACHER
    assert loading.TEACHER.name == "step-050000"


def test_wrong_teacher_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(loading, "TEACHER", tmp_path)
    (tmp_path / "COMMITTED.json").write_text(json.dumps({"complete": True, "step": 10000}))
    with pytest.raises(ValueError, match="pinned A50k"):
        loading.teacher_record()


def test_wrong_step_even_with_matching_digest_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(loading, "TEACHER", tmp_path)
    manifest = tmp_path / "COMMITTED.json"
    manifest.write_text(json.dumps({"complete": True, "step": 10000,
                                   "recipe": {"settings": {"mode": "timeline"}}}))
    monkeypatch.setattr(loading, "TEACHER_MANIFEST_SHA256",
                        hashlib.sha256(manifest.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="completed pooled A50k"):
        loading.teacher_record()


def test_pooled_factory_reuses_body_and_copies_time_weights():
    import torch
    from whenami_timeline_joint.test_joint import fixture
    from whenami_pooled_b50.model import from_base
    base, _ = fixture("pooled_query")
    student = from_base(base)
    assert student.timeline.mode == "pooled_query"
    assert student.vlm is base.vlm and student.reader is base.reader
    for k, v in base.timeline.state_dict().items():
        torch.testing.assert_close(v, student.timeline.state_dict()[k], rtol=0, atol=0)
    assert set(student.state_dict()) - set(base.state_dict()) == {
        "forecast_head.weight", "forecast_head.bias"}


def test_pooled_factory_and_loader_reject_timeline_architecture(tmp_path):
    from whenami_timeline_joint.test_joint import fixture
    from whenami_pooled_b50.model import from_base
    base, _ = fixture("timeline")
    with pytest.raises(ValueError, match="pooled_query"):
        from_base(base)
    with pytest.raises(ValueError, match="timeline architecture"):
        loading.load_model_only(base, tmp_path)


def test_factory_requests_pooled_A(monkeypatch):
    from whenami_timeline_joint.test_joint import fixture
    from whenami_pooled_b50 import model
    calls = []
    base, _ = fixture("pooled_query")
    def fake(mode, seed):
        calls.append((mode, seed))
        return base, "visual", "transform", "tokenizer", "prompt"
    monkeypatch.setattr(model, "load_joint", fake)
    student, *rest = model.build_student(123)
    assert calls == [("pooled_query", 123)]
    assert student.timeline.mode == "pooled_query"
    assert rest == ["visual", "transform", "tokenizer", "prompt"]


def test_student_matches_pooled_A_current_not_timeline():
    import copy
    import torch
    from whenami_timeline_joint.test_joint import fixture
    from whenami_pooled_b50.model import from_base
    from whenami_timeline_transfer.contracts import make_prefix, select_anchor
    base, x = fixture("pooled_query")
    student = from_base(base).eval()
    features = {k: x.pop(k) for k in ("patches", "vae")}
    x.pop("visible")
    prefix = make_prefix(x, features, 1)
    with torch.no_grad():
        expected = select_anchor(base.eval()(**prefix.kwargs(), condition_only=True), 2, 2, 1)
        actual = student(prefix=prefix, condition_only=True)
        timeline = copy.deepcopy(student)
        timeline.timeline.mode = "timeline"
        wrong = timeline(prefix=prefix, condition_only=True)
    for k in ("conditions", "reader_z"):
        torch.testing.assert_close(actual[k], expected[k], rtol=0, atol=0)
    assert not torch.allclose(actual["conditions"], wrong["conditions"])


def test_teacher_in_training_is_constructed_pooled():
    tree = ast.parse((BASE / "whenami_pooled_b50/train.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "load_joint"]
    assert len(calls) == 1 and isinstance(calls[0].args[0], ast.Name)
    assert calls[0].args[0].id == "MODE"
