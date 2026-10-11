"""Evaluate all fixed50k interventions on the exact existing27-domain dev cohort."""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from stage3.common import BASE, STORE
from stage3_oxe_eval.prepare import sha, write_json
from stage3_oxe_eval50.evaluate import initial_noise, SEEDS
from stage3_oxe_eval50.policy import CHECKPOINT
from stage3_mechanism.loader import DiagnosticPolicy
from stage3_mechanism.interventions import VARIANTS, encoded_variants

DATA = STORE / "outputs/stage3_oxe_eval/dev27_64ep_01"
BASELINE = STORE / "outputs/stage3_oxe_eval/compare50k_01/stage3"


def balanced_order(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["dataset"], []).append(row)
    return [groups[d][i] for i in range(max(map(len, groups.values())))
            for d in sorted(groups) if i < len(groups[d])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    committed = json.loads((CHECKPOINT / "COMMITTED.json").read_text())
    if not manifest["complete"] or manifest["domain_count"] != 27 or manifest["train_episode_overlap"]:
        raise ValueError("wrong cohort")
    if manifest["train_index_sha256"] != committed["recipe"]["data_index_sha256"]:
        raise ValueError("wrong production training split")
    prior = json.loads((BASELINE / "protocol.json").read_text())
    if prior["data_manifest_sha256"] != sha(DATA / "MANIFEST.json") or prior["seeds"] != list(SEEDS):
        raise ValueError("baseline cohort/noise differs")
    recipe = {
        "schema": "stage3_50k_fixed_weight_mechanism_v1",
        "data_manifest_sha256": sha(DATA / "MANIFEST.json"),
        "checkpoint_manifest_sha256": sha(CHECKPOINT / "COMMITTED.json"),
        "baseline_protocol_sha256": sha(BASELINE / "protocol.json"),
        "variants": list(VARIANTS), "seeds": list(SEEDS),
        "precision": prior["policy_precision"], "intervals": prior["intervals"],
        "source": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
        "dependencies": {str(p.relative_to(BASE)): sha(p) for p in [
            BASE / "stage3_oxe_eval50/policy.py", BASE / "stage3_oxe_eval50/evaluate.py",
            BASE / "stage3_oxe_eval/prepare.py", BASE / "stage3_oxe_eval/metrics.py",
        ]},
        "scope": "Inference interventions; no retraining; no closed-loop or new-method claim",
    }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if json.loads((args.output / "recipe.json").read_text()) != recipe:
            raise ValueError("recipe drift")
    else:
        write_json(args.output / "recipe.json", recipe)
    directory = args.output / "per_episode"
    directory.mkdir(exist_ok=True)
    rows = balanced_order(manifest["episodes"])
    remaining = []
    for row in rows:
        path = directory / f"{row['dataset']}--{row['episode_key']}.npz"
        if path.exists():
            with np.load(path) as saved:
                if str(saved["data_sha256"]) != row["sha256"] or not np.isfinite(saved["predictions"]).all():
                    raise ValueError("invalid resume case")
        else:
            remaining.append(row)
    policy = DiagnosticPolicy()
    write_json(args.output / "policy.json", policy.record)
    progress = tqdm(total=len(rows), initial=len(rows) - len(remaining), desc="Stage3 mechanism27")
    recent = deque(maxlen=40)
    for ordinal, row in enumerate(remaining, start=1):
        began = time.monotonic()
        source = DATA / row["dataset"] / row["file"]
        if sha(source) != row["sha256"] or row["role"] != "dev":
            raise ValueError("input changed/non-dev")
        reference = BASELINE / "per_episode" / f"{row['dataset']}--{row['episode_key']}.npz"
        diagnostics, predictions, stored_conditions = [], [], []
        with np.load(source) as sample, np.load(reference) as original:
            if not np.array_equal(sample["raw_actions"], original["target_raw"]):
                raise ValueError("target identity differs")
            for i, t in enumerate(sample["timestep"]):
                encoded, diagnostic = encoded_variants(
                    policy, sample["frames"][i, -1], sample["frames"][i, :4],
                    str(sample["instructions"][i]),
                )
                noise = initial_noise(row["dataset"], row["episode_key"], t)
                if not np.array_equal(noise.numpy(), original["initial_noise"][i]):
                    raise ValueError("noise identity differs")
                # Keep same three-seed sampling batch as original, not five-way
                # stacked conditions that would change numerical execution.
                values = np.stack([policy.sample(encoded[v], noise) for v in VARIANTS])
                diff = np.abs(values[0] - original["prediction"][i])
                diagnostic["full_replay_max_abs"] = float(diff.max())
                diagnostic["full_replay_rmse"] = float(np.sqrt(np.square(diff).mean()))
                if diff.max() > .002 or diagnostic["full_replay_rmse"] > .0005:
                    raise ValueError("full50k replay drift: " + str(diagnostic))
                diagnostics.append(diagnostic)
                predictions.append(values)
                stored_conditions.append(np.stack([encoded[v]["condition"].float().cpu().numpy()[0] for v in VARIANTS]))
            payload = {
                "predictions": np.stack(predictions),  # window,variant,seed,16,7
                "conditions": np.stack(stored_conditions),
                "target_raw": np.asarray(sample["raw_actions"]), "action_mask": np.asarray(sample["action_mask"]),
                "timestep": np.asarray(sample["timestep"]), "history_padded": np.asarray(row["history_padded"]),
                "action_padded": np.asarray(row["action_padded"]),
                "dataset": np.asarray(row["dataset"]), "episode_key": np.asarray(row["episode_key"]),
                "data_sha256": np.asarray(row["sha256"]), "variants": np.asarray(VARIANTS),
                "diagnostics_json": np.asarray(json.dumps(diagnostics)),
            }
        dest = directory / reference.name
        tmp = dest.with_suffix(f".tmp-{os.getpid()}")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **payload)
        os.replace(tmp, dest)
        recent.append(time.monotonic() - began)
        done = len(rows) - len(remaining) + ordinal
        progress.update()
        progress.set_postfix(s_ep=round(float(np.mean(recent)), 2))
        if ordinal == 1 or ordinal % 8 == 0:
            write_json(args.output / "progress.json", {
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "completed": done, "total": len(rows),
                "seconds_per_episode": float(np.mean(recent)),
                "eta_seconds": (len(rows) - done) * float(np.mean(recent)),
                "latest_dataset": row["dataset"],
            })
        if args.stop_after and done >= args.stop_after and done < len(rows):
            print("ORDERLY_MECHANISM_STOP", done, flush=True)
            progress.close()
            return
    progress.close()
    write_json(args.output / "COMPLETE.json", {
        "complete": True, "episodes": len(rows), "domains": manifest["domain_count"],
        "variants": list(VARIANTS), "recipe_sha256": sha(args.output / "recipe.json"),
    })


if __name__ == "__main__":
    main()
