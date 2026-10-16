"""160 resume preflight: original local proof + 320/16 or 160/16 resume source."""
import argparse, json
from pathlib import Path
from whenami_timeline_force.preflight import local_preflight
from whenami_timeline_force.loading import source_record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("teacher", "student"), required=True)
    p.add_argument("--variant", choices=("direct", "revise"), required=True)
    p.add_argument("--phase", choices=("resume",), required=True)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--resume-from", type=Path, required=True)
    a = p.parse_args()
    local_preflight()
    if a.stage == "student":
        if a.teacher is None:
            raise ValueError("explicit A+ teacher")
        source_record(a.teacher, "teacher")
    if (a.root/"main").exists():
        raise ValueError("resume output needs fresh RUN_ID")
    meta = json.loads((a.resume_from/"COMMITTED.json").read_text())
    s = meta["recipe"]["settings"]
    if (not meta["complete"] or meta["world_size"] not in (320, 160) or meta["shard_width"] != 16 or
            meta["optimizer_files"] != 16 or s["stage"] != a.stage or s["variant"] != a.variant or
            s["engineering"] or s["global_batch"] != 320):
        raise ValueError("resume needs complete formal 320/16 or 160/16 checkpoint of same stage/variant")
    missing = [i for i in range(160) if not (a.resume_from/f"rng-{i:05d}.pt").exists()]
    missing += [i for i in range(16) if not (a.resume_from/f"optimizer-{i:05d}.pt").exists()]
    if missing:
        raise ValueError("missing shard files "+str(missing[:5]))
    print("TIMELINE_FORCE_160_PREFLIGHT_OK", a.stage, a.variant, "from", meta["world_size"], "step", meta["step"], flush=True)


if __name__ == "__main__":
    main()
