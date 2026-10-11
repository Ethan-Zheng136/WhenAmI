"""Matched deployment-backend evaluation of A and B, independent of FSDP scores.

Old failed FSDP/native bitwise checks are NOT relaxed or overwritten. All models
here are reevaluated with the unchanged, exported native B inference semantics.
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
from tqdm import tqdm

from stage3.common import STORE, BASE, REFERENCE, DEFAULT_FUTURE
from stage3_oxe_eval.prepare import write_json
from whenami_timeline_b.data import student_inputs
from whenami_timeline_b_handoff.runtime import build, load_release, sample as public_sample, sha
from whenami_followup.evaluate_a import DevOnly
from whenami_followup.metrics import score, aggregate, paired

MODELS = (
    ("A50k_timeline", "A", "timeline", 50000),
    ("A50k_pooled", "A", "pooled_query", 50000),
    ("A10k_timeline", "A", "timeline", 10000),
    ("B0_20k", "B", "action_only", 20000),
    ("B1_20k", "B", "condition_align", 20000),
    ("B0_10k", "B", "action_only", 10000),
    ("B1_10k", "B", "condition_align", 10000),
)
SEEDS = (42, 43, 44)


def load(model, kind, variant, step):
    if kind == "B":
        model.timeline.mode = "timeline"
        package = "B10k_20261005_02" if step == 10000 else "B20k_20261006_01"
        root = STORE / "outputs/whenami_timeline_b/nvidia_handoff" / package / variant
        meta = load_release(model, root)
        return {"kind": kind, "variant": variant, "step": step, "release": str(root),
                "manifest_sha256": sha(root / "MODEL.json"), "state_entries": meta["state_entries"]}
    checkpoint = STORE / "outputs/whenami_timeline_joint/training" / (
        f"timeline_joint_{variant}_320_50k_01/main/checkpoints/step-{step:06d}")
    meta = json.loads((checkpoint / "COMMITTED.json").read_text())
    if not meta["complete"] or meta["step"] != step or meta["recipe"]["settings"]["mode"] != variant:
        raise ValueError("wrong A checkpoint")
    for field, directory in (("source", BASE), ("reference_source", REFERENCE)):
        for name, expected in meta["recipe"][field].items():
            if sha(directory / name) != expected:
                raise ValueError("A source mismatch")
    model.timeline.mode = variant
    state = model.state_dict()
    auxiliary = {k for k in state if k.startswith("forecast_head.")}
    if auxiliary != {"forecast_head.weight", "forecast_head.bias"}:
        raise ValueError("unexpected auxiliary keys")
    core = {k: v for k, v in state.items() if k not in auxiliary}
    metadata = dcp.FileSystemReader(checkpoint / "model_dcp").read_metadata().state_dict_metadata
    if set(metadata) != {"model." + k for k in core}:
        raise ValueError("A model keys mismatch")
    for k, value in core.items():
        if value.device.type != "cuda" or tuple(value.shape) != tuple(metadata["model." + k].size):
            raise ValueError("non-private or mismatched A tensor")
    dcp.load({"model": core}, checkpoint_id=checkpoint / "model_dcp", no_dist=True)
    for k, value in core.items():
        if not torch.isfinite(value).all():
            raise ValueError("nonfinite A tensor: " + k)
    with torch.no_grad():
        torch.nn.init.eye_(model.forecast_head.weight)
        torch.nn.init.zeros_(model.forecast_head.bias)
    return {"kind": kind, "variant": variant, "step": step, "checkpoint": str(checkpoint),
            "manifest_sha256": sha(checkpoint / "COMMITTED.json"), "state_entries": len(core),
            "auxiliary_note": "Only the unused 32x32 auxiliary head is absent from A. All action-path weights loaded."}


@torch.no_grad()
def predict(model, prefix, episode):
    device = prefix.times.device
    noises, predictions = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        encoded = model(prefix=prefix, condition_only=True)
        condition = {k: encoded[k] for k in ("conditions", "reader_z")}
        ts = torch.linspace(0, 1, 10, dtype=torch.float32, device=device)
        for seed in SEEDS:
            identity = int.from_bytes(hashlib.sha256(
                f"Bpilot10k:{episode}:{seed}".encode()).digest()[:8], "little") % (2**63 - 1)
            x = torch.randn((1, 16, 7), dtype=torch.float32, device=device,
                            generator=torch.Generator(device=device).manual_seed(identity))
            noises.append(x.cpu().numpy()[0])
            for i in range(9):
                velocity = model(noisy_actions=x, tau=ts[i].expand(1),
                                 conditioning=condition)["velocity"]
                x = x + velocity.float() * (ts[i + 1] - ts[i])
            predictions.append(x.cpu().numpy()[0])
    return np.stack(predictions), np.stack(noises)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    data = DevOnly()
    recipe = {
        "models": MODELS, "seeds": SEEDS,
        "native_runtime_sha256": sha(BASE / "whenami_timeline_b_handoff/runtime.py"),
        "source_sha256": sha(Path(__file__)),
        "metrics_sha256": sha(BASE / "whenami_followup/metrics.py"),
        "dev_loader_sha256": sha(BASE / "whenami_followup/evaluate_a.py"),
        "dev_manifest_sha256": sha(STORE / "outputs/whenami_timeline_a/dev27_16ep_01/MANIFEST.json"),
        "precision": "Unchanged native deployment: FP32 external x/tau/dt; BF16 parameters and autocast. No FSDP input-cast emulation.",
        "scope": "All A and B freshly scored in the SAME native backend. Do not merge with prior FSDP scores.",
    }
    canonical = json.loads(json.dumps(recipe))
    if args.resume and json.loads((args.output / "recipe.json").read_text()) != canonical:
        raise ValueError("resume recipe drift")
    write_json(args.output / "recipe.json", recipe)
    config = json.loads((DEFAULT_FUTURE.parents[1] / "config.json").read_text())
    model, visual, transform, tokenizer, prompt = build(config)
    summaries, records = {}, {}
    for model_index, (name, kind, variant, step) in enumerate(MODELS):
        directory = args.output / name
        per = directory / "per_episode"
        per.mkdir(parents=True, exist_ok=True)
        record = load(model, kind, variant, step)
        write_json(directory / "LOAD_VERIFIED.json", record)
        began = time.monotonic()
        for index in tqdm(range(len(data.clips)), desc="native " + name):
            row = data.clips[index]
            path = per / f"{row['dataset']}--{row['episode_key']}.npz"
            if not path.exists():
                case = data.dev(index)
                prefix, label = student_inputs([case], visual, transform, tokenizer, prompt, "cuda:0")
                prediction, noise = predict(model, prefix, row["episode_key"])
                if index == 0:
                    reference = public_sample(model, prefix, noise[0:1])["normalized_action_unclipped"][0]
                    maximum = float(np.max(np.abs(reference - prediction[0])))
                    if not np.array_equal(reference, prediction[0]):
                        raise ValueError("new evaluator differs from unchanged native deployment sampler")
                    write_json(directory / "NATIVE_SAMPLER_VERIFIED.json", {
                        "complete": True, "max_abs": maximum,
                        "scope": "Same-native-backend sampler equality. NOT FSDP/native equality.",
                    })
                target, mask = label.actions[0].cpu().numpy(), label.valid[0].cpu().numpy()
                score(prediction, target, mask, data.stats[row["dataset"]]["action"])
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as f:
                    np.savez_compressed(f, prediction=prediction, target=target, mask=mask,
                                        initial_noise=noise, source_sha256=np.asarray(row["sha256"]))
                os.replace(temporary, path)
            if index % 10 == 0 or index + 1 == len(data.clips):
                elapsed = time.monotonic() - began
                write_json(args.output / "progress.json", {
                    "utc": datetime.now(timezone.utc).isoformat(), "model": name,
                    "completed": index + 1, "total": len(data.clips),
                    "model_index": model_index, "model_count": len(MODELS),
                    "elapsed_seconds": elapsed,
                    "eta_current_model_seconds": elapsed / (index + 1) * (len(data.clips) - index - 1),
                })
        rows = []
        for row in data.clips:
            path = per / f"{row['dataset']}--{row['episode_key']}.npz"
            with np.load(path) as values:
                if str(values["source_sha256"]) != row["sha256"]:
                    raise ValueError("DEV data changed")
                if model_index:
                    first = args.output / MODELS[0][0] / "per_episode" / path.name
                    with np.load(first) as previous:
                        if not all(np.array_equal(values[k], previous[k]) for k in ("target", "mask", "initial_noise")):
                            raise ValueError("unpaired data/noise")
                metrics = score(values["prediction"], values["target"], values["mask"],
                                data.stats[row["dataset"]]["action"])
            rows.append({"dataset": row["dataset"], "episode_key": row["episode_key"],
                         **{k: float(v.mean()) for k, v in metrics.items()}})
        report = {"complete": True, "model": record, "episodes": len(rows), "seeds": SEEDS,
                  **aggregate(rows), "scope": recipe["scope"]}
        write_json(directory / "RESULT.json", report)
        write_json(directory / "episodes.json", rows)
        summaries[name], records[name] = report, rows
        print("NATIVE_DONE", name, report["macro"], flush=True)
        gc.collect()
    comparisons = {}
    for a, b in (
        ("A10k_timeline", "A50k_timeline"), ("A50k_pooled", "A50k_timeline"),
        ("A10k_timeline", "B0_20k"), ("A10k_timeline", "B1_20k"),
        ("B0_10k", "B0_20k"), ("B1_10k", "B1_20k"),
        ("B0_20k", "B1_20k"),
    ):
        comparisons[f"{a}_to_{b}"] = {
            metric: paired([r[metric] for r in records[a]], [r[metric] for r in records[b]],
                           [r["dataset"] for r in records[a]], larger=metric == "gripper_accuracy")
            for metric in ("mse6", "first_action_mse6", "gripper_accuracy")
        }
    write_json(args.output / "RESULT.json", {
        "complete": True, "models": summaries, "comparisons": comparisons,
        "scope": recipe["scope"],
        "notes": ["Different A training lengths are a performance diagnostic, not equal-budget comparisons.",
                  "Only A50k pooled/time is the matched architecture pair here.",
                  "A/B warm starts differ by design; B20k still inherits A10k.",
                  "No sealed test, simulator, optimizer, or true future model inputs used."],
    })


if __name__ == "__main__":
    main()
