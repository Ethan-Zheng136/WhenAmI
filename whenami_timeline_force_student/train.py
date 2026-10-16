"""Stage II student entry: UNCHANGED whenami_timeline_force(_160).train with ONLY the
teacher gate swapped for gate.teacher_record (intermediate A+ step-040000).

--base 160 -> whenami_timeline_force_160.train (10x16, accum2, virtual world 320; the code
            path the A+ teacher has used since 5k)
--base 320 -> whenami_timeline_force.train (20x16, original formal entry)
Model, losses, weights, LR, schedule, data, eval, checkpointing are byte-identical code.
"""
import importlib
import sys
from whenami_timeline_force_student import gate

MODULES = {"160": "whenami_timeline_force_160.train", "320": "whenami_timeline_force.train"}


def main():
    argv = sys.argv[1:]
    if "--base" not in argv:
        raise SystemExit("--base 160|320 required")
    i = argv.index("--base")
    base = argv[i+1]
    if base not in MODULES:
        raise SystemExit("--base must be 160 or 320")
    del argv[i:i+2]
    if "--stage" not in argv or argv[argv.index("--stage")+1] != "student":
        raise SystemExit("student entry only")
    module = importlib.import_module(MODULES[base])
    module.source_record = gate.teacher_record
    sys.argv = [MODULES[base]]+argv
    module.main()


if __name__ == "__main__":
    main()
