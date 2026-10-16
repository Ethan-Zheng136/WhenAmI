"""Same held-out DEV identities/noise for causal and hindsight timeline nodes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from stage3_scale.checkpoint import rng_state, restore_rng
from stage3_oxe_eval.prepare import write_json
from whenami_unified.data import batch, executed


@torch.no_grad()
def evaluate(model, visual, data, transform, tokenizer, prompt, device, out,
             student=False, seeds=(42, 43, 44), limit=0):
    out = Path(out)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    count = min(limit, len(data.clips)) if limit else len(data.clips)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    rng, was = rng_state(), model.training
    model.eval()
    rows, predictions = [], []
    try:
        for j in range((count+world-1)//world):
            ordinal = j*world+rank
            case = data.dev(min(ordinal, count-1))
            evidence, labels = batch([case], visual, transform, tokenizer, prompt, device)
            cmd = executed([case], 3, device) if student else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc = model(evidence=evidence, executed=cmd, condition_only=True)
            # No future exists after anchor2 in existing DEV; anchor1 can use2.
            specs = (("causal_last", 2, 2),) if student else (
                ("causal_last", 2, 2), ("causal_middle", 1, 1),
                ("hindsight_middle", 1, 2))
            scores = {}
            for name, target, version in specs:
                values = []
                conditioning = {"conditions": enc["node_versions"][version][:, target],
                                "reader_z": enc["node_z_versions"][version][:, target]}
                for seed in seeds:
                    identity = f"force-timeline-v1:{case['provenance']['episode_key']}:{target}:{seed}"
                    s = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2**63-1)
                    x = torch.randn((1, 16, 7), generator=torch.Generator(device=device).manual_seed(s), device=device)
                    ts = torch.linspace(0, 1, 10, device=device)
                    for k in range(9):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            v = model(conditioning=conditioning, noisy_actions=x, tau=ts[k].expand(1))["velocity"]
                        x += v.float()*(ts[k+1]-ts[k])
                    if not torch.isfinite(x).all():
                        raise FloatingPointError("DEV action sampling")
                    values.append(x[0])
                predicted = torch.stack(values)
                stats = data.stats[case["provenance"]["dataset"]]["action"]
                unused = torch.tensor(np.asarray(stats["min"]) == np.asarray(stats["max"]), device=device)
                clip = torch.where(unused, 0., predicted.clamp(-1, 1))
                gt, mask = labels.actions[0, target], labels.valid[0, target]
                err = (clip[:, :, :6]-gt[:, :6]).square().mean(-1)
                scores[name] = {"mse6": float(((err*mask).sum(-1)/mask.sum()).mean()),
                                "first_action_mse6": float(err[:, 0].mean()),
                                "gripper_accuracy": float(((((predicted[:, :, 6] >= .5) == (gt[:, 6] >= .5))*mask).sum(-1).float()/mask.sum()).mean())}
                if ordinal < count:
                    predictions.append({"index": ordinal, "mode": name,
                                        "prediction": predicted.cpu().numpy(),
                                        "target": gt.cpu().numpy(), "mask": mask.cpu().numpy()})
            if ordinal < count:
                rows.append({"dataset": case["provenance"]["dataset"],
                             "episode_key": case["provenance"]["episode_key"], **scores})
        write_json(out/f"rank-{rank:05d}.json", rows)
        # Small, auditable per-case sampler output, never interpreted as success rate.
        torch.save(predictions, out/f"predictions-{rank:05d}.pt")
        if world > 1:
            dist.barrier()
        if rank == 0:
            merged = [r for p in sorted(out.glob("rank-*.json")) for r in json.loads(p.read_text())]
            if len(merged) != count or len({(r["dataset"], r["episode_key"]) for r in merged}) != count:
                raise ValueError("missing/duplicate DEV identities")
            domains = {}
            keys = ("mse6", "first_action_mse6", "gripper_accuracy")
            for domain in sorted({r["dataset"] for r in merged}):
                subset = [r for r in merged if r["dataset"] == domain]
                domains[domain] = {"episodes": len(subset), **{
                    mode: {key: float(np.mean([r[mode][key] for r in subset])) for key in keys}
                    for mode in scores}}
            macro = {mode: {key: float(np.mean([r[mode][key] for r in domains.values()])) for key in keys}
                     for mode in scores}
            write_json(out/"RESULT.json", {"complete": True, "student": student, "episodes": count,
                "seeds": list(seeds), "domains": domains, "macro": macro,
                "protocol": "force_timeline_fixed_dev3_v1", "no_future_student": True,
                "all_reader_conditions_from_node": True, "scope": "offline DEV, NOT SIMPLER success"})
            print("TIMELINE_DEV", macro, flush=True)
        if world > 1:
            dist.barrier()
    finally:
        model.train(was)
        visual.eval()
        restore_rng(rng)
