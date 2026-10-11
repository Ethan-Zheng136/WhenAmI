"""Independently recompute every saved prediction and paired multi-domain results."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from stage3.common import DEFAULT_PAST, STORE
from stage3_oxe_eval.metrics import episode_metrics, summarize_raw
from stage3_oxe_eval50.policy import KINDS, WOG
from stage3_oxe_eval.prepare import sha, write_json


def paired(a, b, names, repeats=2000):
    domains = sorted(set(names.tolist()))
    rng = np.random.default_rng(20261003)
    boot_a, boot_b = np.zeros(repeats), np.zeros(repeats)
    da, db = [], []
    for domain in domains:
        ids = np.flatnonzero(names == domain)
        draws = rng.integers(0, len(ids), size=(repeats, len(ids)))
        boot_a += a[ids][draws].mean(1) / len(domains)
        boot_b += b[ids][draws].mean(1) / len(domains)
        da.append(a[ids].mean()); db.append(b[ids].mean())
    before, after = float(np.mean(da)), float(np.mean(db))
    return {
        "before_macro_mse6": before, "after_macro_mse6": after,
        "relative_reduction_percent": 100 * (1 - after / before),
        "paired_episode_stratified_95pct_reduction_interval":
            np.quantile(100 * (1 - boot_b / boot_a), [.025, .975]).tolist(),
        "domains_favorable": int(sum(y < x for x, y in zip(da, db))),
        "domain_count": len(domains),
        "scope": "fixed models and three inference seeds; not training-seed uncertainty",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True, help="directory containing wog_a/past/stage3")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data / "MANIFEST.json").read_text())
    stats = json.loads((DEFAULT_PAST.parents[1] / "dataset_statistics.json").read_text())
    wog_stats = json.loads((WOG.parents[1] / "dataset_statistics.json").read_text())
    inventory = json.loads((STORE / "outputs/stage3_joint/oxe27_inventory_01/inventory.json").read_text())
    records = {}
    protocols = {k: json.loads((args.runs / k / "protocol.json").read_text()) for k in KINDS}
    comparable = ("schema", "data_manifest_sha256", "limit_per_domain", "episode_count", "seeds", "intervals", "initial_noise", "policy_precision")
    reference = {k: protocols["stage3"][k] for k in comparable}
    for kind in KINDS:
        if {k: protocols[kind][k] for k in comparable} != reference:
            raise ValueError("model evaluation protocols do not match")
        if protocols[kind]["data_manifest_sha256"] != sha(args.data / "MANIFEST.json"):
            raise ValueError("sample manifest changed")
    selected, counts = [], {}
    for row in manifest["episodes"]:
        d = row["dataset"]; counts[d] = counts.get(d, 0) + 1
        if not reference["limit_per_domain"] or counts[d] <= reference["limit_per_domain"]:
            selected.append(row)
    names = np.asarray([r["dataset"] for r in selected])
    comparison_stats = {}
    for kind in KINDS:
        per_episode, moments, window_values, hashes = [], [], [], []
        model_stats = wog_stats if kind == "wog_a" else stats
        statistics_path = (WOG if kind == "wog_a" else DEFAULT_PAST).parents[1] / "dataset_statistics.json"
        for policy_record in (args.runs / kind).glob("policy-rank*.json"):
            if json.loads(policy_record.read_text())["statistics_sha256"] != sha(statistics_path):
                raise ValueError("normalization statistics changed after inference")
        for row in selected:
            path = args.runs / kind / "per_episode" / f"{row['dataset']}--{row['episode_key']}.npz"
            with np.load(path) as data:
                if str(data["episode_key"]) != row["episode_key"] or str(data["data_sha256"]) != row["sha256"]:
                    raise ValueError("prediction identity mismatch")
                with np.load(args.data / row["dataset"] / row["file"]) as original:
                    if not np.array_equal(data["target_raw"], original["raw_actions"]) or not np.array_equal(data["action_mask"], original["action_mask"]):
                        raise ValueError("prediction target/mask changed")
                if kind != "wog_a":
                    paired_path = args.runs / "wog_a/per_episode" / path.name
                    with np.load(paired_path) as first:
                        for key in ("initial_noise", "timestep", "noise_seeds", "target_raw", "action_mask"):
                            if not np.array_equal(data[key], first[key]):
                                raise ValueError("unpaired inputs/noise/targets: " + key)
                value = episode_metrics(data["prediction"], data["target_raw"], data["action_mask"],
                                        model_stats[row["dataset"]]["action"], stats[row["dataset"]]["action"])
                per_episode.append(value["mean"]); moments.append(value["moments"])
                window_values.append(value["per_window_seed"])
            hashes.append({"file": str(path), "sha256": sha(path)})
        by_domain = {}
        core = ("mse6", "unclipped_mse6", "mae6", "first_action_mse6", "gripper_accuracy")
        for name in sorted(set(names.tolist())):
            ids = np.flatnonzero(names == name)
            by_domain[name] = {
                "episodes": len(ids), "windows": sum(selected[i]["windows"] for i in ids),
                "small_domain_below_16_episodes": len(ids) < 16,
                **{k: float(np.mean([per_episode[i][k] for i in ids])) for k in core},
                **summarize_raw([moments[i] for i in ids]),
                "active_continuous_dimensions": int(np.sum(np.asarray(stats[name]["action"]["min"])[:6] != np.asarray(stats[name]["action"]["max"])[:6])),
                "gripper_varies_in_training_statistics": stats[name]["action"]["min"][6] != stats[name]["action"]["max"][6],
            }
        macro = {k: float(np.mean([d[k] for d in by_domain.values()])) for k in core}
        micro = {k: float(np.mean([d[k] for d in per_episode])) for k in core}
        mixture = {k: float(sum(inventory["sampling_probabilities"][d] * by_domain[d][k] for d in by_domain)) for k in core}
        # Report boundary/non-boundary strata without hiding the primary all-case score.
        strata = {}
        for label, field, value in (("real_full_history", "history_padded", False),
                                    ("history_boundary_repeat", "history_padded", True),
                                    ("complete_action_window", "action_padded", False),
                                    ("masked_action_tail", "action_padded", True)):
            per_domain = {}
            for name in by_domain:
                ep = []
                for i in np.flatnonzero(names == name):
                    keep = np.asarray(selected[i][field]) == value
                    if keep.any():
                        ep.append(float(window_values[i]["mse6"][keep].mean()))
                if ep:
                    per_domain[name] = {"episodes": len(ep), "mse6": float(np.mean(ep))}
            strata[label] = {"domains": per_domain, "macro_mse6": float(np.mean([v["mse6"] for v in per_domain.values()])) if per_domain else None}
        records[kind] = {
            "macro": macro, "episode_micro": micro, "training_mixture_weighted": mixture,
            "domains": by_domain, "strata": strata,
            "prediction_files": hashes, "episodes": per_episode,
        }
        comparison_stats[kind] = np.asarray([d["mse6"] for d in per_episode])
    comparisons = {
        f"{a}_to_{b}": paired(comparison_stats[a], comparison_stats[b], names)
        for a, b in (("wog_a", "past"), ("wog_a", "stage3"), ("past", "stage3"))
    }
    result = {
        "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_manifest_sha256": sha(args.data / "MANIFEST.json"), "domain_count": len(set(names)),
        "episodes": len(selected), "windows": sum(r["windows"] for r in selected),
        "inference_seeds": reference["seeds"], "models": records, "comparisons": comparisons,
        "baseline_reuse": "Fixed WoG-A/Past predictions from compare35k_01; exact data/noise/masks checked per episode",
        "analysis_source_sha256": {name: sha((Path(__file__).parent if name == "summarize.py" else Path(__file__).parent.parent / "stage3_oxe_eval") / name) for name in ("summarize.py", "metrics.py")},
        "notes": ["Offline action errors, not task success.", "Existing Stage3 dev, base-pretraining exposure possible.",
                  "Unequal pretraining histories/information; this is performance comparison, not causal mechanism isolation.",
                  "Domain-macro primary; episode-stratified descriptive confidence interval, not independent training runs.",
                  "Small-domain uncertainty and quota-bounded source-shard sampling limit generalization claims."],
    }
    write_json(args.output / "RESULT.json", result)
    lines = ["# OXE27：WoG-A / Past / Stage3-50k 离线动作对比", "",
             f"{result['domain_count']}域，{result['episodes']}条episode，{result['windows']}个窗口，3个推理seed。",
             "现有Stage3开发划分、零Stage3训练episode重叠；不保证基础预训练未见。不是闭环成功率。", "",
             "| 模型 | macro MSE6 ↓ | 首动作 MSE6 ↓ | gripper ↑ |",
             "|---|---:|---:|---:|"]
    for kind in KINDS:
        m = records[kind]["macro"]
        lines.append(f"| {kind} | {m['mse6']:.7f} | {m['first_action_mse6']:.7f} | {m['gripper_accuracy']:.5f} |")
    lines += ["", "## 全部数据域", "", "| 域 | episodes | WoG-A MSE6 | Past MSE6 | Stage3 MSE6 |", "|---|---:|---:|---:|---:|"]
    for name in sorted(set(names.tolist())):
        d = records["stage3"]["domains"][name]
        values = [records[k]["domains"][name]["mse6"] for k in KINDS]
        lines.append(f"| {name} | {d['episodes']} | " + " | ".join(f"{v:.7f}" for v in values) + " |")
    lines += ["", "## 配对比较：正数表示后者误差降低", ""]
    for name, v in comparisons.items():
        lo, hi = v["paired_episode_stratified_95pct_reduction_interval"]
        lines.append(f"- {name}: {v['relative_reduction_percent']:+.3f}%；95%区间[{lo:+.3f}, {hi:+.3f}]%；有利域{v['domains_favorable']}/27。")
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"macro": {k: records[k]["macro"] for k in KINDS}, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
