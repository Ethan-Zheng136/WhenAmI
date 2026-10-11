"""Small, CPU-only memory snapshots; no torch/TF import or process mutation."""

import json
import os
import time
from pathlib import Path


def read_value(path):
    try:
        text = Path(path).read_text().strip()
        try:
            return int(text)
        except ValueError:
            return text
    except (OSError, PermissionError):
        return None


def process_memory(pid=None, include_pss=False):
    pid = os.getpid() if pid is None else pid
    result = {"pid": pid}
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        for line in status:
            name, _, value = line.partition(":")
            if name in ("VmRSS", "VmHWM", "VmSize"):
                result[name + "_kib"] = int(value.split()[0])
            elif name == "Threads":
                result["threads"] = int(value.strip())
        if include_pss:
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                name, _, value = line.partition(":")
                if name in ("Pss", "Private_Dirty", "Private_Clean"):
                    result[name + "_kib"] = int(value.split()[0])
    except (OSError, PermissionError, ValueError):
        pass
    return result


def cgroup_memory():
    # Prefer v2; support the v1 layout on this PPU test machine.
    roots = [
        (Path("/sys/fs/cgroup"), {
            "current_bytes": "memory.current", "limit_bytes": "memory.max",
            "peak_bytes": "memory.peak", "events": "memory.events",
            "pids_current": "pids.current", "pids_max": "pids.max",
        }),
        (Path("/sys/fs/cgroup/memory"), {
            "current_bytes": "memory.usage_in_bytes", "limit_bytes": "memory.limit_in_bytes",
            "peak_bytes": "memory.max_usage_in_bytes", "events": "memory.oom_control",
            "failcnt": "memory.failcnt",
        }),
    ]
    for root, paths in roots:
        if (root / paths["current_bytes"]).exists():
            return {k: read_value(root / p) for k, p in paths.items()}
    return {}


def emit_memory(phase, extra=None):
    """Per-rank lifecycle snapshots, enabled only by WAI_MEMORY_LOG_DIR."""
    out = os.environ.get("WAI_MEMORY_LOG_DIR")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "0")
    record = {
        "time_unix": time.time(), "phase": phase, "rank": rank,
        "local_rank": os.environ.get("LOCAL_RANK", "0"),
        "process": process_memory(include_pss=True), "cgroup": cgroup_memory(),
        **(extra or {}),
    }
    with (root / f"rank-{rank}.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
