"""DEV mse6 (same protocol/noise identities as timeline_force) + present-token probe metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from stage3_scale.checkpoint import rng_state, restore_rng
from stage3_oxe_eval.prepare import write_json
from whenami_present.data import batch, executed, canonical_times


def _world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _gather(rows, out, tag):
    rank, world = _world()
    write_json(out/f"{tag}-rank-{rank:05d}.json", rows)
    if world > 1:
        dist.barrier()
    return [r for p in sorted(out.glob(f"{tag}-rank-*.json")) for r in json.loads(p.read_text())]


def sample(model, conditioning, key, device, steps=10):
    s = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63-1)
    x = torch.randn((1, 16, 7), generator=torch.Generator(device=device).manual_seed(s), device=device)
    ts = torch.linspace(0, 1, steps, device=device)
    for k in range(steps-1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(conditioning=conditioning, noisy_actions=x, tau=ts[k].expand(1))["velocity"]
        x += v.float()*(ts[k+1]-ts[k])
    if not torch.isfinite(x).all():
        raise FloatingPointError("DEV sampling")
    return x[0]


def encode(model, case, visual, transform, tokenizer, prompt, device, student, specs, gap):
    evidence, labels = batch([case], visual, transform, tokenizer, prompt, device)
    cmd = executed([case], 3, device) if student else None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if student:
            res = model(evidence=evidence, executed=cmd, presents=(0, 1, 2),
                        imagine_times=canonical_times(evidence.times, (0, 1, 2), gap))
        else:
            # causal presents: no future (fwd-only); hindsight_middle: present1 + real anchor2.
            res = model(evidence=evidence, presents=(0, 1, 2), future_keep=None)
            if "hindsight_middle" in specs:
                res["present"]["h1"] = model(evidence=evidence, presents=(1,),
                    future_keep=torch.ones(1, dtype=torch.bool, device=device))["present"][1]
    return evidence, labels, res


@torch.no_grad()
def evaluate(model, visual, data, transform, tokenizer, prompt, device, out, student=False,
             seeds=(42, 43, 44), limit=0, probe=None, gap=4):
    out = Path(out)
    rank, world = _world()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    rng, was = rng_state(), model.training
    model.eval()
    specs = {"causal_last": (2, 2), "causal_middle": (1, 1)}
    if not student:
        specs["hindsight_middle"] = ("h1", 1)
    try:
        count = min(limit, len(data.clips)) if limit else len(data.clips)
        rows = []
        for j in range((count+world-1)//world):
            ordinal = j*world+rank
            case = data.dev(min(ordinal, count-1))
            evidence, labels, res = encode(model, case, visual, transform, tokenizer, prompt, device,
                                           student, specs, gap)
            scores = {}
            for name, (key, target) in specs.items():
                o = res["present"][key]
                cond = {"conditions": o["condition"][:, 0], "reader_z": o["reader_z"]}
                pred = torch.stack([sample(model, cond, f"present-v1:{case['provenance']['episode_key']}:{target}:{s}", device)
                                    for s in seeds])
                stats = data.stats[case["provenance"]["dataset"]]["action"]
                unused = torch.tensor(np.asarray(stats["min"]) == np.asarray(stats["max"]), device=device)
                clip = torch.where(unused, 0., pred.clamp(-1, 1))
                gt, mask = labels.actions[0, target], labels.valid[0, target]
                err = (clip[:, :, :6]-gt[:, :6]).square().mean(-1)
                scores[name] = {"mse6": float(((err*mask).sum(-1)/mask.sum()).mean()),
                                "first_action_mse6": float(err[:, 0].mean()),
                                "w_fwd": float(o["w_fwd"].mean())}
            taus = [float(res["present"][c]["tau"][0]) for c in (0, 1, 2)]
            if ordinal < count:
                rows.append({"dataset": case["provenance"]["dataset"], "episode_key": case["provenance"]["episode_key"],
                             "tau": taus, "tau_order": float(taus[0] < taus[1] < taus[2]), **scores})
        merged = _gather(rows, out, "dev")
        probe_result = run_probe(model, probe, visual, transform, tokenizer, prompt, device, out, student, gap) if probe else None
        if rank == 0:
            if len(merged) != count:
                raise ValueError("missing DEV rows")
            domains = {}
            for d in sorted({r["dataset"] for r in merged}):
                sub = [r for r in merged if r["dataset"] == d]
                domains[d] = {m: {k: float(np.mean([r[m][k] for r in sub])) for k in ("mse6", "first_action_mse6", "w_fwd")}
                              for m in specs}
                domains[d]["tau_order"] = float(np.mean([r["tau_order"] for r in sub]))
            macro = {m: {k: float(np.mean([v[m][k] for v in domains.values()])) for k in ("mse6", "first_action_mse6", "w_fwd")}
                     for m in specs}
            macro["dev_tau_order_acc"] = float(np.mean([v["tau_order"] for v in domains.values()]))
            write_json(out/"RESULT.json", {"complete": True, "student": student, "episodes": count,
                "domains": domains, "macro": macro, "probe": probe_result,
                "protocol": "present_token_dev3_v1", "scope": "offline DEV mse6 + probe; NOT SIMPLER"})
            print("PRESENT_DEV", json.dumps(macro), "PROBE", json.dumps(probe_result), flush=True)
        if world > 1:
            dist.barrier()
    finally:
        model.train(was)
        visual.eval()
        restore_rng(rng)


def r2(y, x):
    y, x = np.asarray(y), np.asarray(x)
    return float(1-((y-x)**2).sum()/max(((y-y.mean())**2).sum(), 1e-12))


def run_probe(model, probe, visual, transform, tokenizer, prompt, device, out, student, gap):
    rank, world = _world()
    rows = []
    for j in range((len(probe)+world-1)//world):
        i = j*world+rank
        if i >= len(probe):
            break
        case = probe.case(i)
        evidence, _, res = encode(model, case, visual, transform, tokenizer, prompt, device, student, {}, gap)
        o = res["present"][2]
        vis = evidence.frames[2].patches.float().mean(1)[0]
        rows.append({"i": i, "episode": case["provenance"]["probe_episode"], "progress": case["provenance"]["progress"],
                     "tau": float(o["tau"][0]), "phase": int(o["logits"][0].argmax()),
                     "visual": vis.cpu().numpy().tolist()[:256], "w_fwd": float(o["w_fwd"].mean())})
    merged = _gather(rows, out, "probe")
    if rank != 0:
        return None
    y, t = [r["progress"] for r in merged], [r["tau"] for r in merged]
    pairs, correct, rho = 0, 0, []
    by = {}
    for r in merged:
        by.setdefault(r["episode"], []).append(r)
    sims = []
    for ep in by.values():
        for a in range(len(ep)):
            for b in range(a+1, len(ep)):
                u, v = np.asarray(ep[a]["visual"]), np.asarray(ep[b]["visual"])
                sims.append((float(u@v/(np.linalg.norm(u)*np.linalg.norm(v)+1e-9)), ep[a], ep[b]))
        if len(ep) > 2 and np.ptp([r["tau"] for r in ep]) > 1e-6:
            rho.append(float(np.corrcoef(np.argsort(np.argsort([r["progress"] for r in ep])),
                                         np.argsort(np.argsort([r["tau"] for r in ep])))[0, 1]))
    if sims:
        thr = float(np.quantile([s for s, _, _ in sims], .8))
        for s, a, b in sims:
            if s >= thr and abs(a["progress"]-b["progress"]) >= .3:
                pairs += 1
                correct += int((a["tau"]-b["tau"])*(a["progress"]-b["progress"]) > 0)
    phase_by_third = {}
    for r in merged:
        phase_by_third.setdefault(min(2, int(r["progress"]*3)), []).append(r["phase"])
    return {"windows": len(merged), "episodes": len(by), "progress_R2_head": r2(y, t),
            "within_episode_spearman": float(np.nanmean(rho)) if rho else None,
            "aliasing_pairs": pairs, "aliasing_pair_accuracy": correct/pairs if pairs else None,
            "phase_hist_by_progress_third": {k: np.bincount(v).tolist() for k, v in sorted(phase_by_third.items())},
            "w_fwd_mean": float(np.mean([r["w_fwd"] for r in merged])),
            "scope": "train-index episodes (not held-out); causal 3-anchor windows gap"+str(gap)}
