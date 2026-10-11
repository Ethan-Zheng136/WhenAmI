"""Bind completed local tests/smoke to exact launch sources; no cluster claims."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from stage3.common import BASE
from stage3_oxe_eval.prepare import sha, write_json
from whenami_pooled_b50.loading import teacher_record
from whenami_pooled_b50.train import fingerprints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    suites = list(ET.parse(root / "pytest.xml").getroot().iter("testsuite"))
    if not suites or any(int(s.get(k, 0)) for s in suites for k in ("errors", "failures", "skipped")):
        raise ValueError("local tests incomplete")
    tests = sum(int(s.get("tests", 0)) for s in suites)
    if tests < 26:
        raise ValueError("missing local tests")
    cases = {c.get("name") for s in suites for c in s.findall("testcase")}
    if "test_cpu4rank_A_to_B_warmstart_and_B_optimizer_restart" not in cases:
        raise ValueError("nested FSDP warmstart/restart test absent")
    for name in ("test_pooled_factory_reuses_body_and_copies_time_weights",
                 "test_pooled_factory_and_loader_reject_timeline_architecture",
                 "test_factory_requests_pooled_A",
                 "test_student_matches_pooled_A_current_not_timeline",
                 "test_teacher_in_training_is_constructed_pooled"):
        if name not in cases:
            raise ValueError("pooled semantics test absent: " + name)
    smoke_path = root / "real_A50k_smoke/RESULT.json"
    smoke = json.loads(smoke_path.read_text())
    for key in ("complete", "actual_A50k_model_loaded", "student_teacher_nonaliased",
                "future_raw_image_and_language_invariance", "teacher_frozen"):
        if smoke.get(key) is not True:
            raise ValueError("real smoke proof missing: " + key)
    teacher = teacher_record()
    if smoke["teacher"] != teacher or smoke["causal_dev_domains"] != 27:
        raise ValueError("real smoke source/domains mismatch")
    if smoke["strict_load"]["loaded_A_state_entries"] != 1548:
        raise ValueError("real A state coverage changed")
    if (smoke.get("student_architecture_mode") != "pooled_query"
            or smoke.get("teacher_architecture_mode") != "pooled_query"):
        raise ValueError("real smoke did not execute pooled student and teacher")
    for name, value in smoke["source"].items():
        if sha(BASE / "whenami_pooled_b50" / name) != value:
            raise ValueError("new source changed after real smoke: " + name)
    protected = json.loads((root / "PROTECTED_SOURCE_BEFORE.json").read_text())
    for name, value in protected.items():
        if sha(BASE / name) != value:
            raise ValueError("protected existing source changed: " + name)
    if smoke["arms"]["action_only"]["aux_gradient"] != 0:
        raise ValueError("B0 accidentally optimizes teacher alignment")
    if smoke["arms"]["condition_align"]["aux_gradient"] <= 0:
        raise ValueError("B1 auxiliary gradient absent")
    proof = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "local_passed": True, "source": fingerprints(),
        "launcher_sha256": sha(BASE / "run_pooled_b50_320.sh"),
        "teacher_manifest_sha256": teacher["committed_sha256"],
        "teacher_fixed_step": 50000, "tests_passed": tests,
        "student_architecture_mode": "pooled_query",
        "teacher_architecture_mode": "pooled_query",
        "nested_cpu4rank_A_to_B_warmstart_and_B_optimizer_restart": True,
        "real_A50k_student_teacher_loaded_and_nonaliased": True,
        "real_causal_future_image_language_invariance": True,
        "real_causal_dev_domains": 27,
        "peak_real_smoke_allocated_gib": smoke["peak_allocated_gib"],
        "real_smoke_update": smoke["actual_update"],
        "B0_teacher_loss_unused_verified": True,
        "B1_auxiliary_gradient_verified": True,
        "original_training_loop_preserved_except_teacher_step_provenance": True,
        "protected_A_B_C_shared_sources_unchanged": True,
        "actual320PPU_B50_admission_complete": False,
        "B50_scientific_effectiveness_established": False,
        "evidence_root": str(root),
        "real_smoke_sha256": sha(smoke_path),
        "pytest_xml_sha256": sha(root / "pytest.xml"),
        "notes": [
            "Matched pooled A50k initialization and pooled_query student/teacher architecture.",
            "Original B data/objective/evaluator and update loop preserved; source-bank aggregation is pooled_query.",
            "Single-PPU BF16-body SGD is not full-FP32-Adam cluster admission.",
            "Each submitted arm must pass actual320 topology and full-model100->120 admission.",
        ],
    }
    write_json(BASE / "whenami_pooled_b50/LOCAL_VERIFICATION.json", proof)
    print("B50_LOCAL_PROOF_WRITTEN", tests, teacher["committed_sha256"], flush=True)


if __name__ == "__main__":
    main()
