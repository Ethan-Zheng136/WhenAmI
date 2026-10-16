"""[hide variant: frame2 visible=False, the in-distribution hidden-node path used in training] Counterfactual-future DEV test for timeline_force teacher ckpts (read-only on ckpt).

Same model/loader/scoring as whenami_timeline_force.evaluate (force_timeline_fixed_dev3_v1),
but frame index 2 (the 'future' for hindsight_middle = target1 version2) is replaced by:
  true   : real frame 2
  swap   : frame 2 of the NEXT dev episode of the same domain (cyclic)
  repeat : frame 1 (the current frame) repeated
Instruction, event times and labels are always the episode's own. Single PPU, param_bf16
(as whenami_timeline_force.inference), so 'true' is re-measured here as the reference.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from stage3_oxe_eval50.policy import param_bf16
from whenami_unified.data import Inputs, batch
from whenami_timeline_force.model import build_policy
from whenami_timeline_force.loading import load_weights

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--limit", type=int, default=0)
a = p.parse_args()
device = torch.device("cuda", 0)
torch.cuda.set_device(device)
model, visual, transform, tokenizer, prompt = build_policy(False, "revise", 42)
model = param_bf16(model).to(device).requires_grad_(False).eval()
model.vlm.vision_backbone_requires_grad = False
visual = visual.to(device).requires_grad_(False).eval()
loaded = load_weights(model, a.checkpoint)
if loaded["new_entries"]:
    raise ValueError("incomplete teacher weights: %s" % loaded["new_entries"][:5])
data = Inputs()
count = min(a.limit, len(data.clips)) if a.limit else len(data.clips)
by_domain = {}
for i, row in enumerate(data.clips):
    by_domain.setdefault(row["dataset"], []).append(i)
SPECS = {"true": (("causal_last", 2, 2), ("causal_middle", 1, 1), ("hindsight_middle", 1, 2)),
         "hide": (("causal_middle", 1, 1), ("hindsight_middle", 1, 2))}
rows, t0 = [], time.time()
with torch.no_grad():
    for idx in range(count):
        case = data.dev(idx)
        dom = case["provenance"]["dataset"]
        peers = by_domain[dom]
        donor_idx = peers[(peers.index(idx)+1) % len(peers)]
        donor = data.dev(donor_idx)
        row = {"index": idx, "dataset": dom, "episode_key": case["provenance"]["episode_key"],
               "donor_episode_key": donor["provenance"]["episode_key"]}
        for cond, specs in SPECS.items():
            imgs = np.array(case["observations"]["images"], copy=True)
            if cond == "swap":
                imgs[2] = np.asarray(donor["observations"]["images"])[2] if imgs.shape[1:] == np.asarray(donor["observations"]["images"]).shape[1:] else imgs[2]
                if imgs.shape[1:] != np.asarray(donor["observations"]["images"]).shape[1:]:
                    raise ValueError("donor image shape differs in %s" % dom)
            elif cond == "repeat":
                imgs[2] = imgs[1]
            c = {"observations": {**case["observations"], "images": imgs},
                 "labels": case["labels"], "provenance": case["provenance"]}
            evidence, labels = batch([c], visual, transform, tokenizer, prompt, device, hide=(2 if cond == "hide" else None))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc = model(evidence=evidence, executed=None, condition_only=True)
            scores = {}
            for name, target, version in specs:
                values = []
                conditioning = {"conditions": enc["node_versions"][version][:, target],
                                "reader_z": enc["node_z_versions"][version][:, target]}
                for seed in (42, 43, 44):
                    identity = f"force-timeline-v1:{case['provenance']['episode_key']}:{target}:{seed}"
                    s = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little") % (2**63-1)
                    x = torch.randn((1, 16, 7), generator=torch.Generator(device=device).manual_seed(s), device=device)
                    ts = torch.linspace(0, 1, 10, device=device)
                    for k in range(9):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            v = model(conditioning=conditioning, noisy_actions=x, tau=ts[k].expand(1))["velocity"]
                        x += v.float()*(ts[k+1]-ts[k])
                    values.append(x[0])
                predicted = torch.stack(values)
                stats = data.stats[dom]["action"]
                unused = torch.tensor(np.asarray(stats["min"]) == np.asarray(stats["max"]), device=device)
                clip = torch.where(unused, 0., predicted.clamp(-1, 1))
                gt, mask = labels.actions[0, target], labels.valid[0, target]
                err = (clip[:, :, :6]-gt[:, :6]).square().mean(-1)
                scores[name] = {"mse6": float(((err*mask).sum(-1)/mask.sum()).mean()),
                                "first_action_mse6": float(err[:, 0].mean()),
                                "gripper_accuracy": float(((((predicted[:, :, 6] >= .5) == (gt[:, 6] >= .5))*mask).sum(-1).float()/mask.sum()).mean())}
            row[cond] = scores
        rows.append(row)
        if idx % 20 == 0:
            print(f"CF {idx+1}/{count} {time.time()-t0:.0f}s", flush=True)
keys = ("mse6", "first_action_mse6", "gripper_accuracy")
domains = {}
for dom in sorted({r["dataset"] for r in rows}):
    sub = [r for r in rows if r["dataset"] == dom]
    domains[dom] = {"episodes": len(sub), **{f"{cond}/{m}": {k: float(np.mean([r[cond][m][k] for r in sub])) for k in keys}
                    for cond, specs in SPECS.items() for m, _, _ in specs}}
modes = [f"{cond}/{m}" for cond, specs in SPECS.items() for m, _, _ in specs]
macro = {m: {k: float(np.mean([d[m][k] for d in domains.values()])) for k in keys} for m in modes}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps({"checkpoint": str(a.checkpoint), "episodes": count, "macro": macro,
    "domains": domains, "rows": rows, "loaded": loaded, "precision": "param_bf16 single PPU",
    "seconds": time.time()-t0, "scope": "offline DEV counterfactual, NOT SIMPLER"}, indent=1))
print("CF_MACRO", json.dumps(macro), flush=True)
