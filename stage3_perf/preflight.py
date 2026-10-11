"""Validate original scientific admission plus this explicitly new execution recipe."""

import argparse
import json
from pathlib import Path

from stage3.common import BASE, STORE, REFERENCE, sha256
from stage3_scale.oxe_data import oxe_source_hashes
from stage3_perf.checkpoint import SCHEMA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    store = STORE / "outputs/stage3_joint"
    admission = json.loads((store / "oxe27_admission_32_v2_01/ADMISSION_COMPLETE.json").read_text())
    if not admission["complete"] or not admission["full_policy_train_save_resume_passed"]:
        raise ValueError("Original model correctness admission missing")
    if {f.name: sha256(f) for f in (BASE / "stage3_scale").glob("*.py")} != admission["joint_source"]:
        raise ValueError("Original scientific model/data code changed")
    for key, expected in admission["reference_source"].items():
        if sha256(REFERENCE / key) != expected:
            raise ValueError(f"Reference changed: {key}")
    for key, expected in admission["reused_source"].items():
        if sha256(BASE / key) != expected:
            raise ValueError(f"Dependency changed: {key}")
    data = json.loads(a.data_index.read_text())
    if (not data.get("complete") or data.get("domain_count") != 27 or
            data.get("data_mix") != "oxe_magic_soup_plus" or data.get("missing_shards") or
            data.get("window_policy") != "episode_boundary_repeat_with_valid_action_mask"):
        raise ValueError("Complete full OXE27 data required, not admission fixture")
    if data["oxe_source_sha256"] != oxe_source_hashes():
        raise ValueError("Data standardization changed")
    audit = json.loads((store / "oxe27_domain_audit_02/results.json").read_text())
    if not audit["complete"] or audit["passed"] != 27:
        raise ValueError("Domain audit missing")
    if audit["decoder_sha256"] != sha256(BASE / "stage3_scale/oxe_data.py"):
        raise ValueError("Decoder audit stale")
    verified = json.loads((BASE / "stage3_perf/LOCAL_VERIFICATION.json").read_text())
    actual = {f.name: sha256(f) for f in (BASE / "stage3_perf").glob("*.py")}
    if not verified["local_checks_passed"] or verified["source"] != actual:
        raise ValueError("Performance code has changed since local verification")
    if a.resume:
        r = json.loads((a.output / "recipe.json").read_text())
        if r.get("schema") != "whenami_joint_perf_v1" or r["performance_source"] != actual:
            raise ValueError("Not a compatible performance run; original320 cannot resume here")
        if (a.output / "complete.json").exists():
            raise ValueError("Run already complete")
        cps = list((a.output / "checkpoints").glob("step-*/COMMITTED.json"))
        if not cps or any(json.loads(c.read_text())["schema"] != SCHEMA for c in cps):
            raise ValueError("No compatible committed checkpoint")
    elif a.output.exists() or a.output.with_name(a.output.name + ".tiny").exists():
        raise ValueError("Output/probe exists: choose a fresh RUN_ID; do not overwrite")
    print("PERF_PREFLIGHT_OK full OXE27 + unchanged scientific code + local execution tests", flush=True)
    print("NOTE: this is a new performance recipe; multihost speed is NOT established by old B.", flush=True)


if __name__ == "__main__":
    main()
