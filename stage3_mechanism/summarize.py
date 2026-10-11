"""Recompute metrics and paired episode diagnostics; no causal-training claims."""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from stage3.common import DEFAULT_PAST
from stage3_oxe_eval.metrics import episode_metrics
from stage3_oxe_eval.summarize import paired
from stage3_oxe_eval.prepare import write_json, sha
from stage3_mechanism.evaluate import DATA
from stage3_mechanism.interventions import VARIANTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    complete = (args.run / "COMPLETE.json").exists()
    if not complete and not args.allow_partial:
        raise ValueError("wait until all cases finish")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    stats = json.loads((DEFAULT_PAST.parents[1] / "dataset_statistics.json").read_text())
    names, summaries, changes, diagnostics, strata = [], {v: [] for v in VARIANTS}, [], [], []
    hashes = []
    for row in manifest["episodes"]:
        path = args.run / "per_episode" / f"{row['dataset']}--{row['episode_key']}.npz"
        if not path.exists():
            if complete:
                raise ValueError("missing completed case")
            continue
        with np.load(path) as case:
            if str(case["data_sha256"]) != row["sha256"] or tuple(case["variants"]) != VARIANTS:
                raise ValueError("case identity changed")
            names.append(row["dataset"])
            diagnostics.extend(json.loads(str(case["diagnostics_json"])))
            for j, v in enumerate(VARIANTS):
                value = episode_metrics(case["predictions"][:, j], case["target_raw"], case["action_mask"],
                                        stats[row["dataset"]]["action"], stats[row["dataset"]]["action"])
                summaries[v].append(value["mean"])
            # Change in sampled action output, not error. Keep valid windows only.
            diff = np.abs(case["predictions"] - case["predictions"][:, :1])
            valid = case["action_mask"][:, None, None, :, None]
            denominator = case["action_mask"].sum(-1)[:, None, None] * 7
            per = np.where(valid, diff, 0.).sum((-1, -2)) / denominator
            changes.append(per.mean((0, 2)))
            strata.append({"dataset": row["dataset"], "real_history_windows": int((~case["history_padded"]).sum())})
        hashes.append({"name": path.name, "sha256": sha(path)})
    if not names:
        raise ValueError("no completed cases")
    names = np.asarray(names)
    domains = sorted(set(names.tolist()))
    core = ("mse6", "unclipped_mse6", "mae6", "first_action_mse6", "gripper_accuracy")
    models = {}
    for v in VARIANTS:
        by_domain = {
            d: {"episodes": int((names == d).sum()),
                **{k: float(np.mean([summaries[v][i][k] for i in np.flatnonzero(names == d)])) for k in core}}
            for d in domains
        }
        models[v] = {"domains": by_domain,
                     "macro": {k: float(np.mean([by_domain[d][k] for d in domains])) for k in core}}
    comparison = {
        f"{v}_to_full": paired(np.asarray([x["mse6"] for x in summaries[v]]),
                              np.asarray([x["mse6"] for x in summaries["full"]]), names)
        for v in VARIANTS if v != "full"
    }
    # Positive interaction means K reduces error more with real than repeated history.
    interaction = np.asarray([summaries["no_k"][i]["mse6"] - summaries["full"][i]["mse6"]
                              - summaries["repeat_history_no_k"][i]["mse6"] + summaries["repeat_history"][i]["mse6"]
                              for i in range(len(names))])
    result = {
        "complete": complete, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "domains": len(domains), "episodes": len(names), "variants": models,
        "comparisons_positive_means_full_better": comparison,
        "k_history_error_interaction_macro": float(np.mean([interaction[names == d].mean() for d in domains])),
        "valid_action_mean_absolute_change_vs_full": dict(zip(VARIANTS, np.asarray(changes).mean(0).tolist())),
        "latent_diagnostic_means": {k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]},
        "maximum_full_replay_abs_error": max(d["full_replay_max_abs"] for d in diagnostics),
        "prediction_hashes": hashes,
        "scope": "Fixed-model inference intervention; not retraining contribution, not closed-loop success.",
    }
    write_json(args.output / "RESULT.json", result)
    lines = ["# Stage3-50k固定权重机制诊断", "",
             f"完整：{complete}；{len(domains)}域，{len(names)}条episode；不是重训消融或闭环成功率。", "",
             "| 条件 | macro MSE6↓ | 首动作MSE6↓ | gripper↑ |", "|---|---:|---:|---:|"]
    for v in VARIANTS:
        m = models[v]["macro"]
        lines.append(f"| {v} | {m['mse6']:.7f} | {m['first_action_mse6']:.7f} | {m['gripper_accuracy']:.5f} |")
    lines += ["", "## 配对比较：正数表示原full优于干预", ""]
    for name, item in comparison.items():
        lines.append(f"- {name}: {item['relative_reduction_percent']:+.3f}%；"
                     f"区间{item['paired_episode_stratified_95pct_reduction_interval']}；"
                     f"{item['domains_favorable']}/{item['domain_count']}域有利。")
    lines += ["", "## 全部域MSE6", "", "|域|episodes|"+"|".join(VARIANTS)+"|",
              "|---|---:|"+"---:|"*len(VARIANTS)]
    for d in domains:
        lines.append(f"|{d}|{models['full']['domains'][d]['episodes']}|"+
                     "|".join(f"{models[v]['domains'][d]['mse6']:.7f}" for v in VARIANTS)+"|")
    (args.output / "RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"complete": complete, "episodes": len(names),
                      "macro": {v: models[v]["macro"] for v in VARIANTS},
                      "comparisons": comparison}, indent=2))


if __name__ == "__main__":
    main()
