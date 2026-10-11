"""Require real early checkpoint, gradient and action-evaluation results before continuation."""

import argparse
import json
import math
from pathlib import Path
import statistics

from stage3.common import BASE, sha256
from stage3_perf.checkpoint import SCHEMA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--step", type=int, default=100)
    a = p.parse_args()
    run = a.output
    recipe = json.loads((run / "recipe.json").read_text())
    source = {f.name: sha256(f) for f in (BASE / "stage3_perf").glob("*.py")}
    assert source == recipe["performance_source"]
    cp = run / "checkpoints" / f"step-{a.step:06d}"
    meta = json.loads((cp / "COMMITTED.json").read_text())
    assert meta["schema"] == SCHEMA and meta["complete"] and meta["step"] == a.step
    assert meta["recipe"] == recipe and meta["world_size"] == recipe["world_size"]
    assert len(list(cp.glob("optimizer-*.pt"))) == meta["shard_width"]
    assert len(list(cp.glob("rng-*.pt"))) == recipe["world_size"]
    audit = json.loads((run / "gradient-audit-step000001.json").read_text())
    assert audit["complete"] and all(math.isfinite(v) and v > 0 for v in audit["squared_norm_sum"].values())
    evaluation = json.loads((run / "development" / f"step-{a.step:06d}" / "results.json").read_text())
    assert evaluation["complete"] and evaluation["causal"] and not evaluation["future_images_used"]
    for row in evaluation["results"]:
        assert row["n"] > 0 and math.isfinite(row["normalized_mse_6"])
    rows = [json.loads(l) for l in (run / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, a.step + 1))
    assert all(math.isfinite(r[k]) for r in rows for k in ("total", "grad_norm", "seconds"))
    steady = [r for r in rows if r["step"] > min(10, a.step // 2) and not r["profiled"]]
    assert steady
    summary = {
        "complete": True, "step": a.step, "world_size": recipe["world_size"],
        "save_and_action_eval_passed": True,
        "new_process_resume_is_next_not_yet_proven": True,
        "steady_seconds_per_step": statistics.mean(r["seconds"] for r in steady),
        "peak_allocated_gib_max": max(r["peak_allocated_gib_max"] for r in rows),
        "performance_source": source,
    }
    print("FIRST_SEGMENT_OK", json.dumps(summary), flush=True)
    # All node launchers validate; only node0 should write the shared report.
    import os
    if int(os.environ.get("RANK", 0)) == 0:
        (run / "FIRST_SEGMENT_COMPLETE.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
