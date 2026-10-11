"""One independent matched experiment per torchrun worker; NOT data-parallel SGD."""

import os
from pathlib import Path
import sys

VARIANTS = ("frozen_mean", "frozen_future", "adaptive_mean", "adaptive_future")
SEEDS = (42, 43, 44, 45)


def assignment(rank, seeds=SEEDS):
    if rank < 0 or rank >= len(VARIANTS) * len(seeds):
        raise ValueError("rank outside the factorial sweep")
    return VARIANTS[rank % len(VARIANTS)], seeds[rank // len(VARIANTS)]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter-rank", type=int, default=32)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    if args.print_plan:
        for rank in range(16):
            variant, seed = assignment(rank)
            print(f"GPU {rank:02d}: {variant} seed={seed} output={args.output / f'{variant}_seed{seed}'}")
        return
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world != 16 or int(os.environ.get("LOCAL_WORLD_SIZE", 16)) != 16:
        raise ValueError("one node x16 workers required for this four-seed factorial sweep")
    variant, seed = assignment(rank)
    adaptive, evidence = variant.split("_")
    output = args.output / f"{variant}_seed{seed}"
    args.output.mkdir(parents=True, exist_ok=True)
    # Torchrun's ranks here identify independent experiments, not training shards.
    # Keep LOCAL_RANK for device selection, suppress any implicit distributed rank
    # detection in dependencies. No process group is created.
    os.environ.update(RANK="0", WORLD_SIZE="1")
    os.environ.pop("GROUP_RANK", None)
    sys.argv = [
        "train_reader_study", "--cache", str(args.cache), "--output", str(output),
        "--evidence", evidence, "--seed", str(seed), "--max-steps", str(args.max_steps),
        "--batch-size", str(args.batch_size), "--lr", str(args.lr),
        "--adapter-rank", str(args.adapter_rank),
    ]
    if adaptive == "adaptive":
        sys.argv.append("--adaptive")
    logfile = args.output / f"{variant}_seed{seed}.log"
    # Per-experiment stderr/stdout; torchrun preserves exit status on failure.
    with logfile.open("x", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        print(f"Independent experiment worker={rank}, local_gpu={os.environ.get('LOCAL_RANK')}", flush=True)
        from stage3.train_reader_study import main as train
        train()


if __name__ == "__main__":
    main()
