"""B50 changes the initialization/teacher, not the original-B learning algorithm."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from stage3.common import BASE
from whenami_timeline_b50 import loading


def test_original_training_loop_preserved():
    old = ast.parse((BASE / "whenami_timeline_b/train.py").read_text())
    new = ast.parse((BASE / "whenami_timeline_b50/train.py").read_text())
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
    from whenami_timeline_b50 import train
    from whenami_timeline_b import model, data, evaluate
    assert train.build_student is model.build_student
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
    with pytest.raises(ValueError, match="completed timeline A50k"):
        loading.teacher_record()
