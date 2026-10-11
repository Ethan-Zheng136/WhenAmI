"""Fixed-plan CPU learnability diagnostic; never an OXE/closed-loop result."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from tqdm import tqdm

from whenami_revision.model import EvidenceTimePolicy
from whenami_revision.synthetic import paired_branches, prefix
from whenami_revision.objectives import diagnostic_loss


@torch.no_grad()
def diagnose(model):
    model.eval()
    x, labels = paired_branches(128, 97531, split="diagnostic")
    output = model(**x)
    first = model(**prefix(x, 1))
    no_revision = model(**x, revise=False)
    truth = labels["states"][:, 0]
    initial_p = first["event_logits"][:, 0].softmax(-1)[:, 1]
    revised_p = output["event_logits"][:, 0].softmax(-1)[:, 1]
    decision = output["decision_logits"].argmax(-1)
    target = labels["decisions"][:, -1]
    after_reveal = model(**prefix(x, 2))["event_logits"][:, 0].argmax(-1)
    initially_correct = after_reveal == truth
    overturned = initially_correct & (output["event_logits"][:, 0].argmax(-1) != truth)
    report = {
        "synthetic_only": True, "groups": 128, "cases": len(truth),
        "current_decision_accuracy": float((decision == target).float().mean()),
        "old_event_accuracy_after_reveal": float(initially_correct.float().mean()),
        "old_event_accuracy_after_all_evidence": float(
            (output["event_logits"][:, 0].argmax(-1) == truth).float().mean()),
        "old_event_brier_before_reveal": float((initial_p - truth).square().mean()),
        "old_event_brier_after_all_evidence": float((revised_p - truth).square().mean()),
        "false_old_event_revision_after_reveal": float(overturned.float().sum() / initially_correct.sum().clamp_min(1)),
        "unobservable_prefix_mean_acquired_probability": float(initial_p.mean()),
        "unobservable_prefix_paired_probability_max_difference": float(
            (initial_p.reshape(-1, 3) - initial_p.reshape(-1, 3)[:, :1]).abs().max()),
        "no_revision_intervention_decision_accuracy": float(
            (no_revision["decision_logits"].argmax(-1) == target).float().mean()),
        "no_revision_intervention_changed_decision_fraction": float(
            (no_revision["decision_logits"].argmax(-1) != decision).float().mean()),
        "intervention_not_a_retrained_baseline": True, "branches": {},
    }
    for branch, name in enumerate(("empty_attempt", "grasp_then_drop", "grasp_and_hold")):
        mask = labels["branch"] == branch
        report["branches"][name] = {
            "decision_accuracy": float((decision[mask] == target[mask]).float().mean()),
            "original_grasp_acquired_probability_after_later_evidence": float(revised_p[mask].mean()),
            "current_holding_probability": float(output["event_logits"][mask, -1].softmax(-1)[:, 1].mean()),
        }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    # Fixed defaults implement PLAN_01; overrides are recorded, never disguised.
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--mode", choices=("incremental", "direct_history"), default="incremental")
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    recipe = {
        "task": "synthetic_three_branch_learnability_not_robot_evaluation",
        "steps": args.steps, "seeds": args.seeds, "batch": 48, "hidden": 64, "mode": args.mode,
        "lr": .003, "loss": "decisionCE +0.5*stateCE +0.1*predictionMSE",
        "device": "cpu", "scope": "No superiority over memory baseline claimed",
        "source_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in Path(__file__).parent.glob("*.py")},
        "plan_sha256": hashlib.sha256(Path(__file__).with_name("PLAN_01.md").read_bytes()).hexdigest(),
        "matched_control_plan_sha256": hashlib.sha256(Path(__file__).with_name("PLAN_02.md").read_bytes()).hexdigest(),
    }
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2))
    source = args.output / "source_snapshot"; source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source / path.name)
    for name in ("PLAN_01.md", "PLAN_02.md"):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    reports = []
    for seed in args.seeds:
        run = args.output / f"seed{seed}"; run.mkdir()
        torch.manual_seed(seed)
        model = EvidenceTimePolicy(hidden=64, mode=args.mode)
        opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.)
        start = time.monotonic()
        with (run / "train.jsonl").open("w") as log:
            for step in tqdm(range(1, args.steps + 1), desc=f"Revision synthetic seed{seed}"):
                model.train()
                x, y = paired_branches(16, seed, first_group=(step - 1) * 16, split="train")
                length = 1 + (step - 1) % 4
                x = prefix(x, length)
                opt.zero_grad()
                out = model(**x)
                loss, metrics = diagnostic_loss(out, x["observations"],
                                                 y["states"][:, :length], y["decisions"][:, :length])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite diagnostic loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite diagnostic gradient")
                opt.step()
                log.write(json.dumps({"step": step, "cutoff_ordinal": length - 1, "loss": float(loss.detach()),
                                      "grad_norm": float(norm), **{k: float(v) for k, v in metrics.items()}}) + "\n")
        result = diagnose(model)
        result.update(seed=seed, mode=args.mode, steps=args.steps, elapsed_seconds=time.monotonic() - start,
                      parameters=sum(p.numel() for p in model.parameters()))
        torch.save({"model": model.state_dict(), "recipe": recipe, "seed": seed}, run / "final.pt")
        (run / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        reports.append(result)
        print("SYNTHETIC_DIAGNOSTIC", json.dumps(result), flush=True)
    (args.output / "RESULT.json").write_text(json.dumps({
        "complete": True, "synthetic_only": True, "reports": reports,
        "limitations": ["Not robot data, not closed-loop success, not an OXE result.",
                        "No matched-capacity trained memory baseline in this first contract diagnostic.",
                        "All target event labels are known because the data generator defines them.",
                        "No method-superiority conclusion from intervention or learnability."],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
