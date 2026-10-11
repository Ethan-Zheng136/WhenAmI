"""Fixed27-domain development action sampling. Cache ONLY frozen backbone outputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from evaluation.past_protocol import normalize_actions
from stage3_oxe_eval.prepare import sha, write_json
from whenami_timeline_a.data import stack_cases
from whenami_timeline_a.objectives import action_loss, sample_actions
from whenami_timeline_a import checkpoint


def dataset_cases(root, foundation, limit_per_domain=0):
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if not manifest["complete"] or manifest["domains"] != 27 or manifest["train_overlap"]:
        raise ValueError("requires audited27-domain dev clips")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    counts, selected = {}, []
    for row in manifest["clips"]:
        name = row["dataset"]
        counts[name] = counts.get(name, 0) + 1
        if not limit_per_domain or counts[name] <= limit_per_domain:
            selected.append(row)
    cases = []
    for i, row in enumerate(selected):
        if i % world != rank:
            continue
        path = root / row["file"]
        if sha(path) != row["sha256"]:
            raise ValueError("development raw clip changed")
        with np.load(path) as value:
            obs = {"images": value["images"], "instructions": value["instructions"].tolist(),
                   "times": torch.from_numpy(value["times"].copy())}
            inputs = foundation.encode(obs)
            labels = {
                "actions": torch.from_numpy(normalize_actions(
                    value["raw_actions"], foundation.statistics[row["dataset"]]["action"])),
                "action_mask": torch.from_numpy(value["action_mask"].copy()),
            }
        cases.append({"inputs": inputs, "labels": labels, "provenance": row})
    return cases, len(selected), manifest


def evaluate(model, cases, foundation, output, total):
    output = Path(output)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    # Evaluating must not change dropout or sampler RNG of the next train step.
    rng = checkpoint.rng_state()
    core = checkpoint.bare(model)
    was_training = core.training
    core.eval()
    results = []
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for case in cases:
                x, labels = stack_cases([case], foundation.device)
                row = case["provenance"]
                seed = int.from_bytes(hashlib.sha256(
                    ("timeline-a-dev-noise42/" + row["episode_key"]).encode()).digest()[:8],
                    "little") % (2**63 - 1)
                g = torch.Generator(device=foundation.device).manual_seed(seed)
                noise = torch.randn((*x["times"].shape, 16, 7), generator=g,
                                    device=foundation.device, dtype=torch.float32)
                score = {}
                for mode in ("full", "masked"):
                    visible = torch.ones_like(x["times"], dtype=torch.bool)
                    if mode == "masked":
                        visible[:, x["times"].shape[1] // 2] = False
                    encoded = core(**x, visible=visible)
                    fm, _ = action_loss(core, encoded, labels["actions"], labels["action_mask"],
                                        seed, draws=2)
                    action = sample_actions(core, encoded, noise)
                    # Match training's unused-dimension convention.
                    stats = foundation.statistics[row["dataset"]]["action"]
                    unused = torch.tensor(np.asarray(stats["min"]) == np.asarray(stats["max"]),
                                          device=action.device)
                    p = torch.where(unused, 0., action.clamp(-1, 1))
                    mask = labels["action_mask"]
                    error = (p[..., :6] - labels["actions"][..., :6]).float().square().mean(-1)
                    node_mse = (error * mask).sum(-1) / mask.sum(-1)
                    grip = (((action[..., 6] >= .5) == (labels["actions"][..., 6] >= .5)) * mask).sum(-1) / mask.sum(-1)
                    item = {"action_fm": float(fm), "mse6": float(node_mse.mean()),
                            "first_action_mse6": float(error[..., 0].mean()),
                            "gripper_accuracy": float(grip.mean())}
                    if mode == "masked":
                        item["hidden_node_mse6"] = float(node_mse[~visible].mean())
                    score[mode] = item
                results.append({"dataset": row["dataset"], "episode_key": row["episode_key"], **score})
        write_json(output / f"rank-{rank:05d}.json", results)
        if world > 1:
            dist.barrier()
        if rank == 0:
            merged = []
            for r in range(world):
                merged.extend(json.loads((output / f"rank-{r:05d}.json").read_text()))
            if len(merged) != total or len({(x["dataset"], x["episode_key"]) for x in merged}) != total:
                raise ValueError("missing/duplicate dev cases")
            domains = {}
            for d in sorted({x["dataset"] for x in merged}):
                rows = [r for r in merged if r["dataset"] == d]
                domains[d] = {
                    "episodes": len(rows),
                    **{v: {k: float(np.mean([x[v][k] for x in rows])) for k in rows[0][v]}
                       for v in ("full", "masked")},
                }
            macro = {v: {k: float(np.mean([x[v][k] for x in domains.values()]))
                         for k in next(iter(domains.values()))[v]} for v in ("full", "masked")}
            write_json(output / "RESULT.json", {
                "complete": True, "episodes": total, "domains": domains, "macro": macro,
                "sampling": "pure Gaussian,9 Euler intervals, one fixed identity-based inference seed42",
                "scope": "Phase-A privileged development; not deployed student or closed-loop success.",
            })
            print("TIMELINE_A_DEV", output.name, macro, flush=True)
        if world > 1:
            dist.barrier()
    finally:
        core.train(was_training)
        checkpoint.set_rng(rng)
