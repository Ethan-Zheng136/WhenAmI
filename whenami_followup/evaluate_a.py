"""Evaluate existing A checkpoints with strictly causal B inputs; no training.

Uses B's exact seed identity and sampler. A10k/timeline seed42 is checked against
the saved B0-before-any-training predictions as an independent interface test.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.utils import _cast_forward_inputs
from tqdm import tqdm

from stage3.common import BASE, REFERENCE, STORE, DEFAULT_FUTURE
from stage3_oxe_eval.prepare import write_json
from evaluation.past_protocol import normalize_actions
from whenami_timeline_b.data import student_inputs
from whenami_timeline_b_handoff.runtime import build, sha
from whenami_followup.metrics import score, aggregate, paired

DEV = STORE / "outputs/whenami_timeline_a/dev27_16ep_01"
SEEDS = (42, 43, 44)


class DevOnly:
    def __init__(self):
        self.manifest = json.loads((DEV / "MANIFEST.json").read_text())
        if not self.manifest["complete"] or self.manifest["train_overlap"]:
            raise ValueError("bad DEV contract")
        self.clips = self.manifest["clips"]
        self.stats = json.loads((DEFAULT_FUTURE.parents[1] / "dataset_statistics.json").read_text())

    def dev(self, index):
        row = self.clips[index]
        path = DEV / row["file"]
        if sha(path) != row["sha256"]:
            raise ValueError("DEV source changed")
        with np.load(path) as x:
            # Physical prefix: the future node is never sent to an encoder.
            images = x["images"][:2].copy()
            instructions = x["instructions"][:2].tolist()
            times = torch.from_numpy(x["times"][:2].copy())
            action = normalize_actions(x["raw_actions"][:2], self.stats[row["dataset"]]["action"])
            valid = x["action_mask"][:2].copy()
        return {
            "observations": {"images": images, "instructions": instructions, "times": times},
            "labels": {"actions": torch.from_numpy(action), "action_mask": torch.from_numpy(valid)},
            "provenance": row,
        }


def load_a(model, checkpoint, mode, step):
    meta = json.loads((checkpoint / "COMMITTED.json").read_text())
    if not meta.get("complete") or meta["step"] != step or meta["recipe"]["settings"]["mode"] != mode:
        raise ValueError("not the requested A checkpoint")
    for field, where in (("source", BASE), ("reference_source", REFERENCE)):
        for name, expected in meta["recipe"][field].items():
            if sha(where / name) != expected:
                raise ValueError("A source changed: " + name)
    model.timeline.mode = mode
    state = model.state_dict()
    metadata = dcp.FileSystemReader(checkpoint / "model_dcp").read_metadata().state_dict_metadata
    if set(metadata) != {"model." + key for key in state}:
        raise ValueError("A architecture/state keys do not match exactly")
    for key, value in state.items():
        if value.device.type != "cuda" or tuple(value.shape) != tuple(metadata["model." + key].size):
            raise ValueError("not a private, shape-matched accelerator state")
    dcp.load({"model": state}, checkpoint_id=checkpoint / "model_dcp", no_dist=True)
    for key, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite A weight: " + key)
    return {"checkpoint": str(checkpoint), "mode": mode, "step": step,
            "committed_sha256": sha(checkpoint / "COMMITTED.json"),
            "state_entries": len(state), "all_loaded_finite": True,
            "scope": "Model-only A load. No B update, no teacher future input, no optimizer."}


def initial_reference(data):
    root = STORE / "outputs/whenami_timeline_b/training/timeline_b_action_only_A10k_320_20k_01/admission1/development/step-000000"
    report = json.loads((root / "RESULT.json").read_text())
    if not report["complete"] or report["noise_seeds"] != [42]:
        raise ValueError("invalid B0 step0 reference")
    result = {}
    for path in root.glob("rank-*.npz"):
        with np.load(path) as x:
            for i, index in enumerate(x["indices"]):
                row = data.clips[int(index)]
                result[row["dataset"], row["episode_key"]] = (
                    x["predictions"][i, 0].copy(), x["targets"][i].copy(), x["masks"][i].copy()
                )
    if len(result) != len(data.clips):
        raise ValueError("initial-reference cohort incomplete")
    return result


@torch.no_grad()
def sample(model, prefix, episode):
    values, noises = [], []
    def root_call(**kwargs):
        # FSDP's root MixedPrecision casts floating forward arguments, including
        # Prefix fields and tau, to BF16. Autocast alone does NOT reproduce this.
        # Emulate the actual recorded B evaluator, without changing its deployed
        # native sampler or any already exported model package.
        _, cast_kwargs = _cast_forward_inputs(torch.bfloat16, **kwargs)
        return model(**cast_kwargs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = root_call(prefix=prefix, condition_only=True)
        conditioning = {key: encoded[key] for key in ("conditions", "reader_z")}
        ts = torch.linspace(0, 1, 10, device=prefix.times.device)
        for seed in SEEDS:
            identity = int.from_bytes(hashlib.sha256(
                f"Bpilot10k:{episode}:{seed}".encode()).digest()[:8], "little") % (2**63 - 1)
            x = torch.randn((1, 16, 7), generator=torch.Generator(device=prefix.times.device).manual_seed(identity),
                            device=prefix.times.device)
            noises.append(x[0].cpu().numpy())
            for i in range(9):
                v = root_call(noisy_actions=x, tau=ts[i].expand(1), conditioning=conditioning)["velocity"]
                x = x + v.float() * (ts[i + 1] - ts[i])
            values.append(x[0].cpu().numpy())
    return np.stack(values), np.stack(noises)


def summarize(directory, data):
    rows = []
    for row in data.clips:
        path = directory / "per_episode" / f"{row['dataset']}--{row['episode_key']}.npz"
        with np.load(path) as x:
            if str(x["source_sha256"]) != row["sha256"]:
                raise ValueError("stale prediction data")
            scores = score(x["prediction"], x["target"], x["mask"], data.stats[row["dataset"]]["action"])
        rows.append({"dataset": row["dataset"], "episode_key": row["episode_key"],
                     **{k: float(v.mean()) for k, v in scores.items()}})
    result = {"complete": True, "episodes": len(rows), "sampling_seeds": list(SEEDS),
              "causal_only": True, "observed_nodes": 2, **aggregate(rows),
              "scope": "A checkpoint with future physically removed; no B training, not closed-loop success."}
    write_json(directory / "per_episode.json", rows)
    write_json(directory / "RESULT.json", result)
    return result, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = "cuda:0"
    data = DevOnly()
    recipe = {
        "models": [("timeline", 10000), ("timeline", 50000),
                   ("pooled_query", 50000), ("pooled_query", 10000)],
        "dev_sha256": sha(DEV / "MANIFEST.json"),
        "seeds": list(SEEDS), "noise_identity": "Bpilot10k:{episode_key}:{seed}",
        "forward_precision": "Match recorded FSDP root argument casting to BF16; retain FP32 external Euler state/dt.",
        "sources": {f: sha(BASE / f) for f in (
            "whenami_followup/evaluate_a.py", "whenami_followup/metrics.py",
            "whenami_timeline_b_handoff/runtime.py")},
        "scope": "Existing checkpoint evaluation only. No weight updates or new training.",
    }
    if args.resume and json.loads((args.output / "recipe.json").read_text()) != json.loads(json.dumps(recipe)):
        raise ValueError("resume recipe changed")
    write_json(args.output / "recipe.json", recipe)
    reference = initial_reference(data)
    config = json.loads((DEFAULT_FUTURE.parents[1] / "config.json").read_text())
    model, visual, transform, tokenizer, prompt = build(config, device)
    # A has no auxiliary prediction head. Identity adds no state and is not read by action.
    model.forecast_head = torch.nn.Identity()
    summaries, all_rows = {}, {}
    for mode, step in recipe["models"]:
        name = f"{mode}_A{step}"
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        per = directory / "per_episode"
        per.mkdir(exist_ok=True)
        checkpoint = STORE / "outputs/whenami_timeline_joint/training" / (
            f"timeline_joint_{mode}_320_50k_01/main/checkpoints/step-{step:06d}")
        record = load_a(model, checkpoint, mode, step)
        write_json(directory / "LOAD_VERIFIED.json", record)
        start = time.monotonic()
        reference_max = 0.0
        for index in tqdm(range(len(data.clips)), desc=name):
            row = data.clips[index]
            dest = per / f"{row['dataset']}--{row['episode_key']}.npz"
            if not dest.exists():
                case = data.dev(index)
                prefix, label = student_inputs([case], visual, transform, tokenizer, prompt, device)
                prediction, noise = sample(model, prefix, row["episode_key"])
                target, mask = label.actions[0].cpu().numpy(), label.valid[0].cpu().numpy()
                if mode == "timeline" and step == 10000:
                    old, old_target, old_mask = reference[row["dataset"], row["episode_key"]]
                    if not np.array_equal(old_target, target) or not np.array_equal(old_mask, mask):
                        raise ValueError("reference labels/mask mismatch")
                    error = float(np.max(np.abs(prediction[0] - old)))
                    reference_max = max(reference_max, error)
                    if not np.allclose(prediction[0], old, atol=1e-5, rtol=1e-4):
                        write_json(directory / "REFERENCE_MISMATCH.json", {
                            "index": index, "episode": row["episode_key"], "max_abs": error,
                            "scope": "Native A10k versus saved FSDP B0 step0. Inspect before cross-model comparison.",
                        })
                        raise ValueError("A10k causal/initial-B0 numerical parity failed")
                # Scoring validates finiteness before committing the prediction.
                score(prediction, target, mask, data.stats[row["dataset"]]["action"])
                temporary = dest.with_suffix(".tmp")
                with temporary.open("wb") as f:
                    np.savez_compressed(f, prediction=prediction, target=target, mask=mask,
                                        initial_noise=noise, source_sha256=np.asarray(row["sha256"]))
                os.replace(temporary, dest)
            if index == 0:
                write_json(directory / "SMOKE.json", {
                    "complete": True, "first_real_episode_finite": True,
                    "A10k_vs_B0_step0_checked": mode == "timeline" and step == 10000,
                    "scope": "Single episode interface admission, not final evaluation.",
                })
            if index % 10 == 0 or index + 1 == len(data.clips):
                elapsed = time.monotonic() - start
                write_json(args.output / "progress.json", {
                    "utc": datetime.now(timezone.utc).isoformat(), "model": name,
                    "completed": index + 1, "total": len(data.clips),
                    "seconds": elapsed, "remaining_models": len(recipe["models"]) - len(summaries) - 1,
                })
        summary, rows = summarize(directory, data)
        summaries[name], all_rows[name] = summary, rows
        if mode == "timeline" and step == 10000:
            write_json(directory / "INITIAL_B0_REFERENCE.json", {
                "complete": True, "max_abs": reference_max,
                "scope": "All seed42 episodes checked against saved B0-before-training. "
                         "On resume, max covers newly computed episodes; previously committed episodes were already checked.",
            })
        print("A_CAUSAL_DONE", name, summary["macro"], flush=True)
        gc.collect()
    comparisons = {}
    for before, after in (("timeline_A10000", "timeline_A50000"),
                          ("pooled_query_A10000", "pooled_query_A50000"),
                          ("pooled_query_A50000", "timeline_A50000"),
                          ("pooled_query_A10000", "timeline_A10000")):
        a = {(r["dataset"], r["episode_key"]): r for r in all_rows[before]}
        b = {(r["dataset"], r["episode_key"]): r for r in all_rows[after]}
        keys = sorted(a)
        if keys != sorted(b):
            raise ValueError("unpaired A evaluations")
        comparisons[f"{before}_to_{after}"] = {
            k: paired([a[x][k] for x in keys], [b[x][k] for x in keys],
                      [x[0] for x in keys], larger=k == "gripper_accuracy")
            for k in ("mse6", "first_action_mse6", "gripper_accuracy")
        }
    write_json(args.output / "RESULT.json", {
        "complete": True, "models": summaries, "comparisons": comparisons,
        "scope": recipe["scope"], "sampling_seeds": list(SEEDS),
        "notes": ["A-causal is an inference intervention, not retrained B.",
                  "Each A time/pooled pair is matched in budget and information.",
                  "All are existing DEV results, not SIMPLER or independent training seeds."],
    })


if __name__ == "__main__":
    main()
