"""Fixed PLAN_03 matched-model CPU diagnostic; no robot/production training."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F
from tqdm import tqdm

from whenami_revision.model import EvidenceTimePolicy
from whenami_revision.synthetic import prefix
from whenami_revision.delayed_evidence import delayed_groups
from whenami_revision.objectives import diagnostic_loss


SETTINGS = [("in_distribution", 12, .10), ("length24", 24, .10), ("length32", 32, .10),
            ("noisy25", 12, .25), ("long_clean", 24, .0)]


@torch.no_grad()
def measure(model, name, length, noise):
    model.eval()
    x, y = delayed_groups(128, length, 97532, split="diagnostic", noise=noise)
    # Chunk inference avoids making context-length memory usage a hidden obstacle.
    outputs, off, predicted_first = [], [], []
    for i in range(0, len(y["branch"]), 48):
        batch = {k: v[i:i+48] for k, v in x.items()}
        out = model(**batch)
        outputs.append({k: out[k] for k in ("decision_logits", "event_logits", "filtered_event_logits")})
        off.append(model(**batch, revise=False)["decision_logits"])
        predicted_first.append(model(**prefix(batch, 1))["event_logits"][:, 0].softmax(-1)[:, 1])
    logits = torch.cat([o["decision_logits"] for o in outputs])
    event_logits = torch.cat([o["event_logits"] for o in outputs])
    event_filtered = torch.cat([o["filtered_event_logits"] for o in outputs])
    p0 = event_logits[:, 0].softmax(-1)[:, 1]
    f0 = event_filtered[:, 0].softmax(-1)[:, 1]
    current_p = event_logits[:, -1].softmax(-1)[:, 1]
    prior = torch.cat(predicted_first)
    posterior = y["bayes_branch"][:, -1]
    bayes_old = posterior[:, 1:].sum(1)
    bayes_current = posterior[:, 2]
    true_old = y["states"][:, 0]
    target = y["decisions"][:, -1]
    return {
        "setting": name, "length": length, "noise_flip_probability": noise, "cases": len(target),
        "decision_accuracy": float((logits.argmax(-1) == target).float().mean()),
        "decision_ce": float(F.cross_entropy(logits, target)),
        "bayes_branch_rule_accuracy_on_same_cases": float(
            (posterior.argmax(-1) + 1 == target).float().mean()),
        "old_event_accuracy": float(((p0 >= .5) == true_old).float().mean()),
        "old_event_brier": float((p0 - true_old).square().mean()),
        "old_filtered_event_brier": float((f0 - true_old).square().mean()),
        "old_event_probability_mse_to_bayes": float((p0 - bayes_old).square().mean()),
        "current_state_brier": float((current_p - y["states"][:, -1]).square().mean()),
        "current_probability_mse_to_bayes": float((current_p - bayes_current).square().mean()),
        "prior_acquired_probability": float(prior.mean()),
        "prior_paired_max_difference": float((prior.reshape(-1, 3) - prior.reshape(-1, 3)[:, :1]).abs().max()),
        "disable_revision_decision_accuracy": float(
            (torch.cat(off).argmax(-1) == target).float().mean()),
    }


@torch.no_grad()
def unrelated_intervention(model):
    x, y = delayed_groups(128, 24, 97532, split="diagnostic", noise=.1)
    changed, targets = delayed_groups(128, 24, 97532, split="diagnostic", noise=.1,
                                      flip_unrelated=True)
    for key in ("times", "executed_actions", "executed_mask"):
        assert torch.equal(x[key], changed[key])
    assert torch.equal(y["states"], targets["states"])
    visible_relevant = y["relevant"]
    assert torch.equal(x["observations"][visible_relevant], changed["observations"][visible_relevant])
    original, altered = [], []
    for i in range(0, len(y["branch"]), 48):
        a = model(**{k: v[i:i+48] for k, v in x.items()})
        b = model(**{k: v[i:i+48] for k, v in changed.items()})
        original.append((a["decision_logits"].argmax(-1), a["event_logits"][:, 0].softmax(-1)[:, 1]))
        altered.append((b["decision_logits"].argmax(-1), b["event_logits"][:, 0].softmax(-1)[:, 1]))
    da, db = torch.cat([p[0] for p in original]), torch.cat([p[0] for p in altered])
    pa, pb = torch.cat([p[1] for p in original]), torch.cat([p[1] for p in altered])
    return {
        "cases": len(da), "decision_changed_fraction": float((da != db).float().mean()),
        "old_event_mean_abs_probability_change": float((pa - pb).abs().mean()),
        "old_event_max_abs_probability_change": float((pa - pb).abs().max()),
        "note": "Only explicitly unrelated sensor bits changed. Paired intervention, not a robot evaluation.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("incremental", "direct_history"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).parent
    recipe = {
        "mode": args.mode, "seed": args.seed, "steps": 1200, "batch": 48, "hidden": 64,
        "lr": .003, "lengths": [8, 12, 16], "noise": .1, "device": "cpu",
        "synthetic_only": True, "settings": SETTINGS,
        "source": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("*.py")},
        "plan_sha256": hashlib.sha256((source / "PLAN_03_DELAYED_EVIDENCE.md").read_bytes()).hexdigest(),
    }
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2))
    snapshot = args.output / "source_snapshot"; snapshot.mkdir()
    for name in recipe["source"]:
        shutil.copy2(source / name, snapshot / name)
    shutil.copy2(source / "PLAN_03_DELAYED_EVIDENCE.md", snapshot)
    torch.manual_seed(args.seed)
    model = EvidenceTimePolicy(hidden=64, mode=args.mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.)
    start = time.monotonic()
    with (args.output / "train.jsonl").open("w") as log:
        for step in tqdm(range(1, 1201), desc=f"Delayed {args.mode} seed{args.seed}", mininterval=5):
            model.train()
            length = [8, 12, 16][(step - 1) % 3]
            x, y = delayed_groups(16, length, args.seed, first_group=(step - 1) * 16, noise=.1)
            cutoff = [max(1, length // 3), max(2, 2 * length // 3), length, length][(step-1) % 4]
            x = prefix(x, cutoff)
            optimizer.zero_grad()
            out = model(**x)
            loss, metrics = diagnostic_loss(out, x["observations"], y["states"][:, :cutoff], y["decisions"][:, :cutoff])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite delayed diagnostic loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite delayed diagnostic gradient")
            optimizer.step()
            log.write(json.dumps({"step": step, "length": length, "cutoff": cutoff,
                                  "loss": float(loss.detach()), "grad_norm": float(norm),
                                  **{k: float(v) for k, v in metrics.items()}}) + "\n")
    train_seconds = time.monotonic() - start
    torch.save({"model": model.state_dict(), "recipe": recipe}, args.output / "final.pt")
    results = [measure(model, *spec) for spec in SETTINGS]
    intervention = unrelated_intervention(model)
    (args.output / "RESULT.json").write_text(json.dumps({
        "complete": True, "synthetic_only": True, "mode": args.mode, "seed": args.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_seconds": train_seconds, "results": results, "unrelated_intervention": intervention,
        "claims": "Stress diagnosis only, no robotics/novelty/superiority conclusion without further tests.",
    }, indent=2, allow_nan=False))
    print("DELAYED_DIAGNOSTIC_COMPLETE", args.mode, args.seed, json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
