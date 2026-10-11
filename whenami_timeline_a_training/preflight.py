"""Read-only checks before user-submitted32-PPU admission/full jobs."""

import argparse
import json
from pathlib import Path

from stage3.common import BASE
from stage3_oxe_eval.prepare import sha
from whenami_timeline_a_training.train import DEFAULT_DEV


def check_sources():
    proof = json.loads((BASE / "whenami_timeline_a/LOCAL_VERIFICATION.json").read_text())
    actual = {p.name: sha(p) for p in (BASE / "whenami_timeline_a").glob("*.py")}
    if not proof["complete"] or proof["source"] != actual:
        raise ValueError("admitted model source changed")
    local = BASE / "whenami_timeline_a_training/LOCAL_VERIFICATION.json"
    verified = json.loads(local.read_text())
    actual = {p.name: sha(p) for p in local.parent.glob("*.py")}
    if not verified["local_passed"] or actual != verified["source"]:
        raise ValueError("training integration has not passed current-source local admission")
    dev = json.loads((DEFAULT_DEV / "MANIFEST.json").read_text())
    if not dev["complete"] or dev["domains"] != 27 or dev["train_overlap"]:
        raise ValueError("development data incomplete/overlapping")
    if sha(DEFAULT_DEV / "MANIFEST.json") != verified["dev_manifest_sha256"]:
        raise ValueError("development manifest changed")


def check_admission(root, mode):
    root = Path(root)
    first, resumed = root / "admission1", root / "admission2"
    m1 = json.loads((first / "checkpoints/step-000100/COMMITTED.json").read_text())
    m2 = json.loads((resumed / "checkpoints/step-000120/COMMITTED.json").read_text())
    if not (m1["complete"] and m2["complete"]) or m1["recipe"] != m2["recipe"]:
        raise ValueError("admission checkpoint/recipe mismatch")
    if m2["world"] != 32 or m2["mode"] != mode:
        raise ValueError("requires same32-PPU topology/routing mode")
    resume = json.loads((resumed / "RESUME_VERIFIED.json").read_text())
    if not resume["complete"] or (resume["restored"], resume["first_update"]) != (100, 101):
        raise ValueError("new-process resume not verified")
    for run, step in ((first, 100), (resumed, 120)):
        result = json.loads((run / "development" / f"step-{step:06d}" / "RESULT.json").read_text())
        if not result["complete"] or len(result["domains"]) != 27:
            raise ValueError("admission must evaluate all27 domains")
        rows = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
        expected = list(range(1,101)) if run == first else list(range(101,121))
        if [r["step"] for r in rows] != expected:
            raise ValueError("admission updates missing")
        if not all(max(r["gradients"][k] for r in rows) > 0 for k in rows[0]["gradients"]):
            raise ValueError("one training path never received gradients")
    return resumed / "checkpoints/step-000120"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--phase", choices=("admit", "full"), required=True)
    args = p.parse_args()
    check_sources()
    if args.phase == "admit":
        if (args.root / "admission1").exists() or (args.root / "admission2").exists():
            raise ValueError("use a fresh RUN_ID; no overwriting partial or completed admission")
    else:
        check_admission(args.root, args.mode)
        if (args.root / "main").exists():
            raise ValueError("main output exists; explicit recovery protocol required")
    print("TIMELINE_A_PREFLIGHT_OK", args.phase, args.mode, flush=True)


if __name__ == "__main__":
    main()
