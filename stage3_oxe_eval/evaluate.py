"""Resumable pure-noise OXE evaluation, with optional independent rank partitioning."""

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from stage3.common import BASE
from stage3_oxe_eval.policy import Policy, KINDS, CHECKPOINT
from stage3_oxe_eval.prepare import sha, write_json

SEEDS = (42, 43, 44)


def initial_noise(dataset, key, timestep, seeds=SEEDS):
    """Exact CPU arrays shared by all models/rank layouts/hardware."""
    values = []
    for seed in seeds:
        text = f"oxe35k-eval01:{dataset}:{key}:{int(timestep)}:{seed}"
        n = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)
        values.append(torch.randn((16, 7), generator=torch.Generator().manual_seed(n), dtype=torch.float32))
    return torch.stack(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--limit-per-domain", type=int, default=0, help="admission only; zero means all prepared episodes")
    p.add_argument("--stop-after", type=int, default=0, help="orderly local stop; completed cases remain resumable")
    args = p.parse_args()
    if args.world < 1 or not 0 <= args.rank < args.world or min(args.limit_per_domain, args.stop_after) < 0:
        p.error("invalid rank/limits")
    torch.set_num_threads(1)
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    manifest_path = args.data / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["complete"] or manifest["train_episode_overlap"] or manifest["domain_count"] != 27:
        raise ValueError("requires complete audited27-domain dev manifest")
    committed = json.loads((CHECKPOINT / "COMMITTED.json").read_text())
    if manifest["train_index_sha256"] != committed["recipe"]["data_index_sha256"]:
        raise ValueError("sample overlap audit used a different training index")
    all_rows = manifest["episodes"]
    if any(r["role"] != "dev" for r in all_rows):
        raise ValueError("non-dev sample in evaluation")
    seen, selected = {}, []
    for row in all_rows:
        d = row["dataset"]
        seen[d] = seen.get(d, 0) + 1
        if not args.limit_per_domain or seen[d] <= args.limit_per_domain:
            selected.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema": "oxe27_existing_dev_pure_noise_01", "kind": args.kind,
        "data_manifest_sha256": sha(manifest_path), "limit_per_domain": args.limit_per_domain,
        "episode_count": len(selected), "seeds": list(SEEDS), "intervals": 9,
        "initial_noise": "identity-based shared FP32 CPU torch.Generator arrays",
        "policy_precision": "BF16 parameters, FP32 scale buffers, FP32 integration",
        "source": {name: sha(BASE / "stage3_oxe_eval" / name)
                   for name in ("evaluate.py", "policy.py", "prepare.py")},
        "stage3_checkpoint_manifest_sha256": sha(CHECKPOINT / "COMMITTED.json"),
    }
    recipe_path = args.output / "protocol.json"
    # Independent ranks can race only when writing IDENTICAL immutable protocol.
    if recipe_path.exists():
        if json.loads(recipe_path.read_text()) != protocol:
            raise ValueError("evaluation protocol changed")
        if not args.resume:
            raise ValueError("existing evaluation; explicitly use --resume")
    else:
        write_json(recipe_path, protocol)
    directory = args.output / "per_episode"
    directory.mkdir(exist_ok=True)
    rows = [r for i, r in enumerate(selected) if i % args.world == args.rank]
    remaining = []
    for row in rows:
        path = directory / f"{row['dataset']}--{row['episode_key']}.npz"
        if path.exists():
            with np.load(path) as old:
                if str(old["episode_key"]) != row["episode_key"] or str(old["data_sha256"]) != row["sha256"]:
                    raise ValueError("resume case identity mismatch")
                if not np.isfinite(old["prediction"]).all():
                    raise ValueError("resume nonfinite prediction")
        else:
            remaining.append(row)
    if remaining:
        policy = Policy(args.kind, device=f"cuda:{local}")
        prior_records = list(args.output.glob("policy-rank*.json"))
        for prior_path in prior_records:
            prior = json.loads(prior_path.read_text())
            for key in ("kind", "checkpoint", "manifest_sha256", "checkpoint_bytes",
                        "checkpoint_mtime_ns", "config_sha256", "statistics_sha256"):
                if prior.get(key) != policy.record.get(key):
                    raise ValueError("model identity changed between evaluation processes")
        write_json(args.output / f"policy-rank{args.rank:04d}.json", policy.record)
        recent = deque(maxlen=50)
        progress = tqdm(total=len(rows), initial=len(rows) - len(remaining), desc=f"OXE27 {args.kind}")
        for count, row in enumerate(remaining, start=1):
            began = time.monotonic()
            source = args.data / row["dataset"] / row["file"]
            if sha(source) != row["sha256"]:
                raise ValueError("prepared episode changed")
            with np.load(source) as case:
                # Policy API takes observations/language only. Labels saved after prediction.
                predictions, noise_values = [], []
                for j, t in enumerate(case["timestep"]):
                    encoded = policy.encode(case["frames"][j, -1],
                                            None if args.kind == "wog_a" else case["frames"][j, :4],
                                            str(case["instructions"][j]))
                    noise = initial_noise(row["dataset"], row["episode_key"], t)
                    predictions.append(policy.sample(encoded, noise))
                    noise_values.append(noise.numpy())
                payload = {
                    "prediction": np.stack(predictions), "initial_noise": np.stack(noise_values),
                    "target_raw": np.asarray(case["raw_actions"]), "action_mask": np.asarray(case["action_mask"]),
                    "timestep": np.asarray(case["timestep"]), "episode_key": np.asarray(row["episode_key"]),
                    "dataset": np.asarray(row["dataset"]), "data_sha256": np.asarray(row["sha256"]),
                    "noise_seeds": np.asarray(SEEDS),
                }
            dest = directory / f"{row['dataset']}--{row['episode_key']}.npz"
            temp = dest.with_suffix(f".tmp-{os.getpid()}")
            with temp.open("wb") as f:
                np.savez_compressed(f, **payload)
            os.replace(temp, dest)
            recent.append(time.monotonic() - began)
            progress.update()
            progress.set_postfix(s_ep=round(float(np.mean(recent)), 2))
            if count == 1 or count % 8 == 0:
                write_json(args.output / f"progress-rank{args.rank:04d}.json", {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "completed": len(rows) - len(remaining) + count, "rank_total": len(rows),
                    "seconds_per_episode_recent": float(np.mean(recent)),
                    "eta_seconds": (len(remaining) - count) * float(np.mean(recent)),
                    "latest_dataset": row["dataset"], "kind": args.kind,
                })
            if args.stop_after and count >= args.stop_after and count < len(remaining):
                progress.close()
                print("EVALUATION_ORDERLY_PARTIAL_STOP", count, flush=True)
                return
        progress.close()
    write_json(args.output / f"COMPLETE-rank{args.rank:04d}-of{args.world:04d}.json", {
        "complete": True, "rank": args.rank, "world": args.world, "episodes": len(rows),
        "protocol_sha256": sha(recipe_path),
    })
    print("EVALUATION_RANK_COMPLETE", args.kind, args.rank, len(rows), flush=True)


if __name__ == "__main__":
    main()
