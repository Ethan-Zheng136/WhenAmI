"""Run a command while recording node cgroup memory and child-process RSS.

No torch/TF imports. Remains outside torchrun, so it can record a worker SIGKILL.
Optional safety limits are only for local data-pipeline profiling, not training.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

path = Path(__file__).resolve().parents[1] / "prismatic/util/memory_diagnostics.py"
spec = importlib.util.spec_from_file_location("memory_diagnostics_standalone", path)
mem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=2)
    ap.add_argument("--max-child-rss-gib", type=float, default=0)
    ap.add_argument("--timeout", type=float, default=0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("missing command after --")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as log:
        proc = subprocess.Popen(command, start_new_session=True)
        start = time.monotonic()
        stop_reason = None
        def on_signal(signum, frame):
            nonlocal stop_reason
            stop_reason = f"parent_received_signal_{signum}"
            if proc.poll() is None:
                os.killpg(proc.pid, signum)
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        try:
            while True:
                try:
                    parent = psutil.Process(proc.pid)
                    children = [parent] + parent.children(recursive=True)
                    processes = [mem.process_memory(p.pid) for p in children]
                except psutil.Error:
                    processes = []
                rss_bytes = sum(p.get("VmRSS_kib", 0) for p in processes) * 1024
                elapsed = time.monotonic() - start
                record = {
                    "time_unix": time.time(), "elapsed_seconds": elapsed,
                    "node_rank": os.environ.get("RANK", "0"),
                    "cgroup": mem.cgroup_memory(),
                    "child_rss_sum_bytes": rss_bytes,
                    "processes": processes,
                    "returncode": proc.poll(), "stop_reason": stop_reason,
                }
                if not log.tell():
                    record["runtime_env"] = {
                        name: os.environ.get(name) for name in (
                            "WAI_BOUNDED_INPUT", "WAI_TF_PRIVATE_THREADS", "WAI_FRAME_PARALLEL_CALLS",
                            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                            "OMP_NUM_THREADS", "MALLOC_ARENA_MAX", "WAI_DATASET_STATISTICS",
                        )
                    }
                log.write(json.dumps(record) + "\n")
                log.flush()
                if proc.poll() is not None:
                    break
                if args.timeout and elapsed > args.timeout:
                    stop_reason = "profiling_wall_time_limit"
                if args.max_child_rss_gib and rss_bytes > args.max_child_rss_gib * 2**30:
                    stop_reason = "profiling_rss_safety_limit_not_a_kernel_oom"
                if stop_reason:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                else:
                    time.sleep(args.interval)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        return 124 if stop_reason else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
