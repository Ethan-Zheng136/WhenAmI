"""Pure-noise action evaluation of finished timeline checkpoints.

All four modes consume identical frozen causal features and noise. Prediction
functions accept NO action labels. Outputs are separate from production runs.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from stage3.common import BASE, check_endpoint
from whenami_timeline.backbone import FrozenPastBackbone
from whenami_timeline.data import RawTimelines, feature_batch
from whenami_timeline.model import Timeline, TimelineConfig


ROOT = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline")
INDEX = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json")
VARIANTS = ("past", "direct_history", "revision", "revision_no_update")
NOISE_SEEDS = (42, 43, 44)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def noise_seed(dataset, key, event_time, seed):
    value = f"timeline-actions01:{dataset}:{key}:{int(event_time)}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def make_noise(provenance, seeds, device):
    values = []
    # Ordering: case, sampling seed, anchor.
    for case in provenance:
        for seed in seeds:
            for event_time in case["source_times"]:
                generator = torch.Generator(device=device).manual_seed(
                    noise_seed(case["dataset"], case["episode_key"], event_time, seed))
                values.append(torch.randn(16, 7, generator=generator, device=device, dtype=torch.float32))
    return torch.stack(values)


@contextmanager
def disable_old_revision(model):
    handle = model.relevance_gate.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def sample_actions(reader, z, condition, initial_noise, intervals=9):
    """No labels/target actions; same sampler and noise for all variants."""
    if intervals < 1 or z.shape[0] != condition.shape[0] or len(z) != len(initial_noise):
        raise ValueError("sampler batch/interval mismatch")
    device = initial_noise.device
    x = initial_noise.clone()
    ts = torch.linspace(0, 1, intervals + 1, device=device, dtype=torch.float32)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for i in range(intervals):
            velocity = reader(x, ts[i].expand(len(x)), z, per_token=condition)
            x = x + velocity.float() * (ts[i + 1] - ts[i])
    if not torch.isfinite(x).all():
        raise FloatingPointError("nonfinite sampled actions")
    return x


def per_anchor_metrics(predictions, targets, mask, unused=None):
    """[cases,seeds,anchors,16,7] -> per-case/seed/anchor errors.

    Main errors are in the stored training normalization coordinates, with
    clamped predictions, not raw-unit metrics pooled across embodiments.
    """
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if p.ndim != 5 or p.shape[-2:] != (16, 7) or y.shape != p.shape[:1] + p.shape[2:]:
        raise ValueError("prediction/target shapes")
    if valid.shape != y.shape[:-1] or not valid.any(-1).all() or not np.isfinite(p).all():
        raise ValueError("invalid mask/predictions")
    if not np.isfinite(y).all():
        raise ValueError("invalid targets")
    clipped = np.clip(p, -1, 1)
    if unused is not None:
        unused = np.asarray(unused, dtype=bool)
        if unused.shape != (len(p), 7):
            raise ValueError("unused action dimensions")
        clipped = np.where(unused[:, None, None, None, :], 0., clipped)
        p = np.where(unused[:, None, None, None, :], 0., p)
    error = clipped[..., :6] - y[:, None, ..., :6]
    raw_error = p[..., :6] - y[:, None, ..., :6]
    weight = valid[:, None]
    denominator = weight.sum(-1)
    def masked_mean(value):
        return (value * weight).sum(-1) / denominator
    return {
        "mse6": masked_mean(np.square(error).mean(-1)),
        "unclipped_mse6": masked_mean(np.square(raw_error).mean(-1)),
        "mae6": masked_mean(np.abs(error).mean(-1)),
        "first_action_mse6": np.square(error[..., 0, :]).mean(-1),
        "gripper_accuracy": masked_mean((p[..., 6] >= .5) == (y[:, None, ..., 6] >= .5)),
    }


def summarize(metric_arrays, datasets):
    domains = sorted(set(datasets))
    by_domain = {}
    for name in domains:
        select = datasets == name
        by_domain[name] = {
            "episodes": int(select.sum()),
            **{k: float(v[select].mean()) for k, v in metric_arrays.items()},
            "by_anchor": [
                {k: float(v[select, :, i].mean()) for k, v in metric_arrays.items()}
                for i in range(3)
            ],
        }
    return {
        "domains": by_domain,
        "macro": {k: float(np.mean([by_domain[d][k] for d in domains])) for k in metric_arrays},
        "macro_by_anchor": [
            {k: float(np.mean([by_domain[d]["by_anchor"][i][k] for d in domains])) for k in metric_arrays}
            for i in range(3)
        ],
        "micro": {k: float(v.mean()) for k, v in metric_arrays.items()},
        "macro_mse_by_noise_seed": [
            float(np.mean([metric_arrays["mse6"][datasets == d, j].mean() for d in domains]))
            for j in range(metric_arrays["mse6"].shape[1])
        ],
    }


def paired_summary(a, b, datasets, repeats=2000):
    """Positive reduction means b beats a; resample episodes within domain."""
    domains = sorted(set(datasets))
    av, bv = a.mean((1, 2)), b.mean((1, 2))
    domain_a, domain_b = [], []
    rng = np.random.default_rng(20261003)
    boot_a, boot_b = np.zeros(repeats), np.zeros(repeats)
    for d in domains:
        ids = np.flatnonzero(datasets == d)
        domain_a.append(av[ids].mean())
        domain_b.append(bv[ids].mean())
        draws = rng.integers(0, len(ids), size=(repeats, len(ids)))
        boot_a += av[ids][draws].mean(1) / len(domains)
        boot_b += bv[ids][draws].mean(1) / len(domains)
    before, after = float(np.mean(domain_a)), float(np.mean(domain_b))
    return {
        "before_macro_mse6": before, "after_macro_mse6": after,
        "relative_reduction_percent": 100 * (1 - after / before),
        "paired_episode_stratified_95pct_reduction_interval":
            np.quantile(100 * (1 - boot_b / boot_a), [.025, .975]).tolist(),
        "domains_favorable": int(sum(y < x for x, y in zip(domain_a, domain_b))),
        "domain_count": len(domains),
        "scope": "Descriptive fixed-model/three-inference-seed interval, not training-seed uncertainty.",
    }


def load_completed(mode, device):
    run = ROOT / f"{mode}_seed42_32ppu_5k_01"
    recipe = json.loads((run / "recipe.json").read_text())
    done = json.loads((run / "COMPLETE.json").read_text())
    cp = run / "checkpoints/step-005000"
    meta = json.loads((cp / "COMMITTED.json").read_text())
    if not done["complete"] or done["steps"] != 5000 or not meta["complete"] or meta["step"] != 5000 or meta["recipe"] != recipe:
        raise ValueError("not a valid completed final5000 checkpoint")
    for name, expected in recipe["source"].items():
        if sha(BASE / "whenami_timeline" / name) != expected:
            raise ValueError("trained source mismatch: " + name)
    for field in ("science_source", "reference_source"):
        for name, expected in recipe[field].items():
            if sha(BASE / name) != expected:
                raise ValueError("trained dependency changed: " + name)
    check_endpoint(recipe["frozen_endpoint"])
    weights = torch.load(cp / "timeline.pt", map_location="cpu", weights_only=True)
    if not all(torch.isfinite(t).all() for t in weights["model"].values() if torch.is_floating_point(t)):
        raise ValueError("nonfinite learned timeline state")
    model = Timeline(TimelineConfig(**recipe["config"]), mode=mode).to(device).eval()
    model.load_state_dict(weights["model"], strict=True)
    model.requires_grad_(False)
    return model, recipe, {"path": str(cp / "timeline.pt"), "sha256": sha(cp / "timeline.pt")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Resume completed per-trajectory predictions only")
    p.add_argument("--stop-after-cases", type=int, default=0,
                   help="Orderly partial stop after this many total completed trajectories")
    args = p.parse_args()
    if args.stop_after_cases < 0:
        p.error("stop-after-cases must be nonnegative")
    if args.resume:
        if not (args.output / "manifest.json").exists() or (args.output / "RESULT.json").exists():
            raise ValueError("resume requires an incomplete existing evaluation")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    direct, recipe, direct_cp = load_completed("direct_history", device)
    revision, other, rev_cp = load_completed("revision", device)
    def comparable(r):
        r = json.loads(json.dumps(r))
        r["settings"].pop("mode"); r["settings"].pop("output")
        return r
    if comparable(recipe) != comparable(other):
        raise ValueError("unmatched experiment recipes")
    source = Path(__file__).parent
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(source / "PLAN_FINAL_5K.md"),
        "source": {f.name: sha(f) for f in source.glob("*.py")},
        "checkpoints": {"direct_history": direct_cp, "revision": rev_cp},
        "data_index": str(INDEX), "data_index_sha256": sha(INDEX),
        "noise_seeds": NOISE_SEEDS, "euler_intervals": 9,
        "variants": VARIANTS, "selection_seed": 20261003, "max_episodes_per_domain": 16,
        "precision": "FP32 timeline/noise/integration; BF16 frozen backbone/DiT",
        "main_metric": "clamped-normalized valid-step MSE6, episode then domain macro",
        "scope": "Expanded internal development from original TRAIN; not sealed or closed-loop.",
    }
    if args.resume:
        original = json.loads((args.output / "manifest.json").read_text())
        manifest["created_utc"] = original["created_utc"]
        # JSON normalizes tuples to lists; compare serialized representations.
        if json.loads(json.dumps(manifest)) != original:
            raise ValueError("evaluation recipe/source/checkpoint drift")
    else:
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        snap = args.output / "source_snapshot"; snap.mkdir()
        for f in source.glob("*.py"): shutil.copy2(f, snap / f.name)
        shutil.copy2(source / "PLAN_FINAL_5K.md", snap)
    per_case = args.output / "per_case"; per_case.mkdir(exist_ok=True)
    started = time.monotonic()
    print("LOAD_SHARED_FROZEN_BACKBONE", flush=True)
    backbone = FrozenPastBackbone(device)
    if backbone.endpoint != recipe["frozen_endpoint"]:
        raise ValueError("different base policy")
    data = RawTimelines(INDEX, role="internal_dev")
    selected = []
    for domain in data.domains:
        used = set()
        for ordinal in range(512):
            entry, times = data.select(20261003, 1, 0, ordinal, domain=domain)
            if entry["episode_key"] in used: continue
            used.add(entry["episode_key"]); selected.append((entry, times))
            if len(used) >= min(16, len(data.episodes[domain])): break
        print("SELECTED", domain, len(used), flush=True)
    case_records, labels_all, mask_all, unused_all = [], [], [], []
    predictions = {v: [] for v in VARIANTS}
    gate_means = []
    for case_index, (entry, times) in enumerate(tqdm(selected, desc="Timeline final actions", mininterval=5)):
        saved = per_case / f"case-{case_index:05d}.npz"
        if saved.exists():
            with np.load(saved) as checkpoint:
                provenance = json.loads(str(checkpoint["provenance_json"]))
                if (provenance["dataset"] != entry["dataset"] or provenance["episode_key"] != entry["episode_key"] or
                        provenance["source_times"] != times.tolist()):
                    raise ValueError("saved case identity changed")
                for variant in VARIANTS:
                    prediction = checkpoint["prediction_" + variant]
                    if prediction.shape != (len(NOISE_SEEDS), 3, 16, 7) or not np.isfinite(prediction).all():
                        raise ValueError("invalid cached predictions")
                    predictions[variant].append(prediction.copy())
                labels_all.append(checkpoint["targets"].copy())
                mask_all.append(checkpoint["mask"].copy())
                unused_all.append(checkpoint["unused"].copy())
                gate_means.append(checkpoint["gate_means"].tolist())
                case_records.append(provenance)
            continue
        raw, labels, provenance = data.materialize(entry, times, backbone.transform, backbone.statistics)
        encoded = backbone.encode(raw)
        cases = [{"policy": encoded, "labels": labels}]
        inputs, _ = feature_batch(cases, [0], device)
        with torch.no_grad():
            out_direct = direct(**inputs)
            out_revision = revision(**inputs)
            with disable_old_revision(revision):
                no_update = revision(**inputs)
            # The initial node has no older bank; intervention must not alter it.
            torch.testing.assert_close(out_revision["conditions"][:, 0], no_update["conditions"][:, 0], rtol=0, atol=0)
            conditions = {
                "past": inputs["base_condition"], "direct_history": out_direct["conditions"],
                "revision": out_revision["conditions"], "revision_no_update": no_update["conditions"],
            }
            initial_noise = make_noise([provenance], NOISE_SEEDS, device)
            z = inputs["z"][:, None].expand(-1, len(NOISE_SEEDS), -1, -1, -1).flatten(0, 2)
            # One reader batch for all variants eliminates sampler/config differences.
            condition = torch.cat([
                conditions[v][:, None].expand(-1, len(NOISE_SEEDS), -1, -1, -1).flatten(0, 2)
                for v in VARIANTS
            ])
            generated = sample_actions(backbone.reader, z.repeat(len(VARIANTS), 1, 1),
                                       condition, initial_noise.repeat(len(VARIANTS), 1, 1))
            for name, prediction in zip(VARIANTS, generated.chunk(len(VARIANTS))):
                predictions[name].append(prediction.reshape(len(NOISE_SEEDS), 3, 16, 7).cpu().numpy())
            gate_means.append([float(g.mean()) for g in out_revision["revision_gates"]])
        labels_all.append(labels["actions"].numpy())
        mask_all.append(labels["action_mask"].numpy())
        stats = backbone.statistics[entry["dataset"]]["action"]
        unused_all.append(np.asarray(stats["min"]) == np.asarray(stats["max"]))
        case_records.append(provenance)
        # Publish each completed trajectory atomically. Interrupted inference
        # can resume without changing samples or repeating completed predictions.
        temporary = saved.with_suffix(".incomplete")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{"prediction_" + v: predictions[v][-1] for v in VARIANTS},
                                targets=labels_all[-1], mask=mask_all[-1], unused=unused_all[-1],
                                gate_means=np.asarray(gate_means[-1]),
                                provenance_json=np.asarray(json.dumps(provenance)))
        os.replace(temporary, saved)
        if (case_index + 1) % 27 == 0:
            print("ACTION_CASES_COMPLETE", case_index + 1, len(selected), flush=True)
        if args.stop_after_cases and case_index + 1 >= args.stop_after_cases and case_index + 1 < len(selected):
            print("ORDERLY_EVAL_PARTIAL_STOP", case_index + 1, flush=True)
            return
    backbone.assert_frozen()
    targets, mask, unused = np.stack(labels_all), np.stack(mask_all), np.stack(unused_all)
    dataset = np.asarray([c["dataset"] for c in case_records])
    keys = np.asarray([c["episode_key"] for c in case_records])
    metrics = {}; summary = {}
    for v in VARIANTS:
        prediction = np.stack(predictions[v])
        metrics[v] = per_anchor_metrics(prediction, targets, mask, unused)
        summary[v] = summarize(metrics[v], dataset)
        np.savez_compressed(args.output / f"{v}.npz", predictions=prediction, targets=targets,
                            valid_action_mask=mask, dataset=dataset, episode_key=keys,
                            source_times=np.asarray([c["source_times"] for c in case_records]),
                            noise_seeds=np.asarray(NOISE_SEEDS), unused_dimensions=unused,
                            **{f"metric_{k}": value for k, value in metrics[v].items()})
    pairs = {}
    for a, b in (("past", "direct_history"), ("past", "revision"),
                 ("direct_history", "revision"), ("revision_no_update", "revision")):
        pairs[f"{a}_to_{b}"] = paired_summary(metrics[a]["mse6"], metrics[b]["mse6"], dataset)
    (args.output / "cases.json").write_text(json.dumps(case_records, indent=2))
    np.save(args.output / "revision_gate_means.npy", np.asarray(gate_means))
    check_endpoint(backbone.endpoint)
    result = {
        "complete": True, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "episodes": len(case_records), "anchors": 3 * len(case_records), "domains": len(set(dataset)),
        "noise_seeds": NOISE_SEEDS, "training_seeds": [42],
        "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "metrics": summary, "comparisons": pairs,
        "limitations": [
            "One training seed; three noise seeds are NOT three independently trained models.",
            "Internal development split, potential pretraining exposure, not a sealed-test score.",
            "At most16 unique episodes per domain; domain counts may be smaller.",
            "Gate removal is an inference intervention, not a trained no-revision baseline.",
            "Clipped normalized mask-aware metrics differ from earlier full-window/raw protocols.",
            "No robot closed-loop performance claim.",
        ],
    }
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print("TIMELINE_SAMPLED_ACTION_RESULTS", json.dumps({k: summary[k]["macro"] for k in summary}), flush=True)


if __name__ == "__main__":
    main()
