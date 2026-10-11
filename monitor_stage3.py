#!/usr/bin/env python3
"""Read-only progress/ETA for a running joint trainer; no torch or GPU needed.

Kept OUTSIDE stage3_scale so adding monitoring does not invalidate its source
fingerprint. ETA uses recent logged training-step durations and explicitly
excludes checkpoint/evaluation overhead, which the trainer does not time.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time


DEFAULT_RUN = Path(
    "/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/oxe27_forecast_320_50k_01"
)


def read_recent(path, window):
    # Re-read so same-topology resume/log trimming is handled without stale state.
    rows = deque(maxlen=window)
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break  # Writer may be in the middle of appending a JSON record.
            row = json.loads(line)
            if rows and row["step"] <= rows[-1]["step"]:
                rows = deque((r for r in rows if r["step"] < row["step"]), maxlen=window)
            rows.append(row)
    return list(rows)


def snapshot(run, window=100):
    recipe = json.loads((run / "recipe.json").read_text())
    settings = recipe["settings"]
    path = run / "train.jsonl"
    rows = read_recent(path, window) if path.exists() else []
    total = settings["steps"]
    result = {
        "run": str(run), "observed_utc": datetime.now(timezone.utc).isoformat(),
        "step": rows[-1]["step"] if rows else 0, "total_steps": total,
        "complete": False, "eta_excludes_checkpoint_and_evaluation": True,
        "status": "waiting_for_first_step" if not rows else "training_log_observed",
    }
    if (run / "complete.json").exists():
        result["complete"] = bool(json.loads((run / "complete.json").read_text()).get("complete"))
        if result["complete"]:
            result["status"] = "complete"
    elif result["step"] >= total:
        result["status"] = "updates_done_waiting_for_final_save_or_evaluation"
    if not rows:
        return result
    step = result["step"]
    seconds = [r["seconds"] for r in rows]
    if not all(math.isfinite(v) and v > 0 for v in seconds):
        raise ValueError("Invalid step timing: refusing to produce a misleading ETA")
    mean = statistics.mean(seconds)
    save_step = min(((step // settings["save_every"]) + 1) * settings["save_every"], total)
    age = max(0., time.time() - path.stat().st_mtime)
    result.update(
        fraction=min(step / total, 1.), recent_steps=len(rows),
        seconds_per_step=mean, min_seconds=min(seconds), max_seconds=max(seconds),
        observations_per_second=settings["global_batch"] / mean,
        remaining_training_seconds=max(total - step, 0) * mean,
        next_checkpoint_step=save_step,
        training_seconds_to_next_checkpoint=max(save_step - step, 0) * mean,
        log_age_seconds=age,
        log_update_delayed=age > max(120., mean * 5),
        latest=rows[-1],
        # Inferred from the recorded schedule, not independently measured.
        rk_lr_from_schedule=rows[-1]["lr"] * settings["new_lr"] / settings["lr"],
    )
    return result


def duration(seconds):
    if seconds is None:
        return "?"
    minutes = int(max(0, seconds) // 60)
    days, hours = divmod(minutes // 60, 24)
    return f"{days}d {hours:02d}h {minutes % 60:02d}m" if days else f"{hours:02d}h {minutes % 60:02d}m"


def render(state):
    if state["step"] == 0:
        return f"Stage3 0/{state['total_steps']} | waiting for first complete train.jsonl record"
    width = 24
    filled = int(state["fraction"] * width)
    bar = "#" * filled + "-" * (width - filled)
    row = state["latest"]
    message = (
        f"Stage3 [{bar}] {state['fraction']:.2%} "
        f"{state['step']:,}/{state['total_steps']:,} | "
        f"{state['seconds_per_step']:.2f}s/step (last {state['recent_steps']}) | "
        f"ETA(train only) {duration(state['remaining_training_seconds'])} | "
        f"ckpt@{state['next_checkpoint_step']:,} ~{duration(state['training_seconds_to_next_checkpoint'])} | "
        f"loss={row.get('total', float('nan')):.4f} "
        f"action={row.get('action_fm', float('nan')):.4f} "
        f"lr={row['lr']:.2e} | log age={state['log_age_seconds']:.0f}s"
    )
    if state["log_update_delayed"]:
        message += " | no recent update: may be saving/evaluating, slow, or stopped; inspect job"
    if state["status"] != "training_log_observed":
        message += f" | {state['status']}"
    return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--interval", type=float, default=15.)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print one machine-readable snapshot")
    parser.add_argument("--plain", action="store_true", help="Print lines instead of refreshing a terminal row")
    args = parser.parse_args()
    if args.window < 1 or args.interval <= 0:
        parser.error("window and interval must be positive")
    refresh = sys.stdout.isatty() and not (args.plain or args.once or args.json)
    if not args.json:
        print("Read-only monitor. ETA excludes save/evaluation overhead; Ctrl-C stops only this monitor.", flush=True)
    try:
        while True:
            state = snapshot(args.run, args.window)
            if args.json:
                print(json.dumps(state, indent=2, allow_nan=False))
                return
            text = render(state)
            print(("\r\033[K" if refresh else "") + text, end="" if refresh else "\n", flush=True)
            if args.once or state["complete"]:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if refresh:
            print()


if __name__ == "__main__":
    main()
