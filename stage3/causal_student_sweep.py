"""16 independent single-card causal students: four losses x four train seeds."""

import argparse
import os
from pathlib import Path
import sys

from stage3.causal_student import VARIANTS, SEEDS


def assignment(rank):
    if rank < 0 or rank >= len(VARIANTS) * len(SEEDS):
        raise ValueError("rank outside student factorial")
    return VARIANTS[rank % len(VARIANTS)], SEEDS[rank // len(VARIANTS)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--readers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fm-draws", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if args.print_plan:
        for rank in range(16):
            variant, seed = assignment(rank)
            print(f"GPU{rank:02d}: {variant} seed{seed}, {args.max_steps} NEW steps, "
                  f"batch={args.batch_size} independent observations x {args.fm_draws} FM draws")
        return
    rank = int(os.environ.get("RANK", 0))
    if int(os.environ.get("WORLD_SIZE", 1)) != 16 or int(os.environ.get("LOCAL_WORLD_SIZE", 16)) != 16:
        raise ValueError("requires one node x16 workers; not data-parallel SGD")
    variant, seed = assignment(rank)
    out = args.output / f"{variant}_seed{seed}"
    if args.resume and (out / "complete.json").exists():
        print("ALREADY_COMPLETE", out, flush=True)
        return
    worker_resume = args.resume and out.exists()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(RANK="0", WORLD_SIZE="1")
    os.environ.pop("GROUP_RANK", None)
    sys.argv = [
        "train_causal_student", "--cache", str(args.cache), "--readers", str(args.readers),
        "--output", str(out), "--variant", variant, "--seed", str(seed),
        "--max-steps", str(args.max_steps), "--batch-size", str(args.batch_size),
        "--fm-draws", str(args.fm_draws), "--lr", str(args.lr), "--warmup", str(args.warmup),
    ]
    if worker_resume:
        sys.argv.append("--resume")
    log = args.output / f"{variant}_seed{seed}.log"
    with log.open("a" if args.resume else "x", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        print("STUDENT_WORKER", rank, "GPU", os.environ.get("LOCAL_RANK"), "RESUME", worker_resume, flush=True)
        from stage3.train_causal_student import main as train
        train()


if __name__ == "__main__":
    main()
