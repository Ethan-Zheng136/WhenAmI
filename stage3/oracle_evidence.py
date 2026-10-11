"""Oracle-only evidence diagnostics, NOT a deployable R+K replacement.

Separates mixed student/oracle training from the information interface. The
1024D probes all use identical K architecture and full-H training statistics.
"""

import argparse
import json
from pathlib import Path
import sys
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from stage3.behavior import load_behavior
from stage3.cache import EndpointCache
from stage3.common import BASE, sha256, write_json
from stage3.evaluate import denormalize
from stage3.refinement import PastRefinement, RefinementConfig

sys.path.insert(0, str(BASE / "evaluation"))
from past_protocol import sample_seed, action_metrics


def lift_condition(condition, hidden_mean, weight, bias):
    """Minimum-norm lift around TRAIN H mean. Contains only C, not sample H.

For full-row-rank W, projecting this reconstruction reproduces C. Nullspace
coordinates stay at their training means, so this is a controlled H bottleneck.
"""
    w = np.asarray(weight, dtype=np.float64)
    mean = np.asarray(hidden_mean, dtype=np.float64)
    c = np.asarray(condition, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if np.linalg.matrix_rank(w) != w.shape[0]:
        raise ValueError("Future projection must have full row rank for an information-preserving lift")
    inverse = np.linalg.pinv(w, rcond=1e-12)
    reconstructed = mean + (c - (mean @ w.T + b)) @ inverse.T
    error = float(np.max(np.abs(reconstructed @ w.T + b - c)))
    if error > 1e-6:
        raise ValueError("condition information was not preserved by lift")
    return reconstructed.astype(np.float32), error


class EvidenceProbe(nn.Module):
    def __init__(self, config, target_statistics, memory_mean, memory_std):
        super().__init__()
        # Consume the same initialization sequence as the original R+K trials,
        # then discard R. K32-condition is therefore directly comparable to K.
        base = PastRefinement(config, target_statistics, mode="adapter")
        self.k = base.k
        self.register_buffer("past_scale", base.r.past_scale.detach().clone())
        self.register_buffer("memory_mean", torch.as_tensor(memory_mean).float().clone())
        self.register_buffer("memory_scale", torch.as_tensor(memory_std).float().clamp_min(.02))
        if self.memory_mean.shape != self.memory_scale.shape:
            raise ValueError("memory statistics shape mismatch")
        if self.memory_mean.shape[0] != config.tokens:
            raise ValueError("wrong memory token count")
        if self.memory_mean.shape[-1] != config.condition_dim:
            # Same K except evidence projection; all H-interface probes use
            # the same dimensions/initialization, including the C-lift control.
            bias = self.k.future_input.bias.detach().clone()
            self.k.future_input = nn.Linear(self.memory_mean.shape[-1], config.k_hidden)
            with torch.no_grad():
                self.k.future_input.bias.copy_(bias)

    def forward(self, past_h, past_z, past_c, evidence):
        standardized = (evidence.detach() - self.memory_mean) / self.memory_scale
        delta = self.k(past_h.detach(), past_z.detach(), standardized) * self.past_scale
        return past_c.detach() + delta


def evidence_array(cache, view, static_path):
    hidden = cache.arrays["future_h"]
    condition = cache.arrays["future_c"]
    if view == "condition":
        reference = condition
        evidence = condition
        details = {}
    else:
        reference = hidden
        details = {}
        if view == "hidden":
            evidence = hidden
        elif view == "mean":
            evidence = np.broadcast_to(np.asarray(hidden[cache.train_indices]).mean(0), hidden.shape)
        elif view == "projected":
            ck = torch.load(cache.manifest["endpoints"]["future"]["path"], map_location="cpu", mmap=True, weights_only=True)
            q = ck["model"]["qformer"]
            w, b = q["v_proj.weight"].float().numpy(), q["v_proj.bias"].float().numpy()
            evidence, error = lift_condition(condition, np.asarray(hidden[cache.train_indices]).mean(0), w, b)
            # Quantify extraction precision separately from mathematical lift.
            original_projection_error = float(np.max(np.abs(np.asarray(hidden) @ w.T + b - condition)))
            details = {
                "lift_projection_max_abs_float64": error,
                "lift_projection_max_abs_float32": float(np.max(np.abs(evidence @ w.T + b - condition))),
                "original_cached_projection_max_abs_float32": original_projection_error,
                "projection_rank": int(np.linalg.matrix_rank(w)),
            }
        elif view == "static":
            if static_path is None:
                raise ValueError("--static-cache required")
            meta = json.loads((static_path / "manifest.json").read_text())
            if not meta["complete"] or meta["parent_cache_manifest_sha256"] != cache.manifest_sha256:
                raise ValueError("static evidence cache mismatch")
            filename = static_path / "future_static_h.npy"
            if sha256(filename) != meta["array_sha256"]:
                raise ValueError("static evidence modified")
            evidence = np.load(filename, mmap_mode="r", allow_pickle=False)
            details = {"static_cache": str(static_path), "static_array_sha256": meta["array_sha256"]}
        else:
            raise ValueError("unsupported view")
    if evidence.shape != reference.shape or not np.isfinite(evidence).all():
        raise ValueError("invalid evidence array")
    train = np.asarray(reference[cache.train_indices])
    return evidence, train.mean(0), train.std(0), details


@torch.no_grad()
def evaluate(model, reader, cache, evidence, output, device, seed):
    model.eval()
    raw = np.load(cache.path / "raw_actions.npy", mmap_mode="r")
    records = []
    for name in sorted(set(cache.datasets.tolist())):
        ids = cache.valid_indices[cache.datasets[cache.valid_indices] == name]
        values = {v: [] for v in ("evidence", "mean_intervention", "true_past")}
        for i in ids:
            batch = cache.batch([i], device)
            e = torch.from_numpy(np.array(evidence[i:i + 1], copy=True)).to(device)
            args = [batch[f"past_{key}"] for key in ("h", "z", "c")]
            conditions = {
                "evidence": model(*args, e),
                "mean_intervention": model(*args, model.memory_mean[None]),
                "true_past": batch["past_c"],
            }
            noise = sample_seed(name, cache.episodes[i], cache.timesteps[i], seed)
            for variant, condition in conditions.items():
                values[variant].append(reader.sample(batch["past_z"], condition, noise).cpu().numpy()[0])
        stats = cache.manifest["endpoints"]["past"]["statistics"][name]["action"]
        for variant, predictions in values.items():
            predictions = denormalize(np.stack(predictions), stats)
            target = np.asarray(raw[ids])
            metrics = action_metrics(predictions, target, stats)
            np.savez_compressed(
                output / f"{name}_{variant}.npz", predictions=predictions, target=target, indices=ids,
                episode=cache.episodes[ids], timestep=cache.timesteps[ids],
            )
            row = {"dataset": name, "variant": variant, **metrics}
            records.append(row)
            print("RESULT", row, flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=("condition", "hidden", "projected", "static", "mean"), required=True)
    parser.add_argument("--static-cache", type=Path)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_steps < 1:
        raise ValueError("positive steps required")
    args.output.mkdir(parents=True, exist_ok=False)
    cache = EndpointCache(args.cache)
    evidence, mean, std, details = evidence_array(cache, args.view, args.static_cache)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RefinementConfig()
    model = EvidenceProbe(config, cache.target_statistics(), mean, std).to(device)
    reader = load_behavior(cache.manifest["endpoints"]["past"], device)
    versions = [p._version for p in reader.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)
    metadata = {
        "schema": "whenami_oracle_evidence_probe_v1", "view": args.view,
        "cache": str(cache.path), "cache_manifest_sha256": cache.manifest_sha256,
        "config": config.as_dict(), "steps": args.max_steps, "batch_size": 8, "lr": 1e-4,
        "seed": args.seed, "memory_dim": int(mean.shape[-1]),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "details": details, "source_sha256": sha256(Path(__file__)),
        "notes": [
            "Oracle/evidence diagnostic, not deployable R+K. No student branch, no alignment/distillation.",
            "Only K optimized; same Past H/z/C, fixed Past DiT and action normalization.",
            "H probes share full-H TRAIN statistics and identical architecture.",
            "Projected H is a lift of C only, with all missing coordinates fixed to train means.",
            "Static H uses the Future encoder but repeats current frame; contains no future observations.",
            "Same seed/batches/action draws as previous K adapter; single seed, previously inspected validation.",
        ],
    }
    write_json(args.output / "config.json", metadata)

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    for step in range(1, args.max_steps + 1):
        model.train()
        ids = np.random.default_rng(args.seed + step).choice(cache.train_indices, 8, replace=True)
        batch = cache.batch(ids, device)
        e = torch.from_numpy(np.array(evidence[ids], copy=True)).to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp():
            condition = model(batch["past_h"], batch["past_z"], batch["past_c"], e)
            if step == 1:
                torch.testing.assert_close(condition, batch["past_c"], rtol=0, atol=0)
            loss = reader.loss(batch["past_actions"], batch["past_z"], condition, args.seed + step)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if reader.training or any(p.grad is not None for p in reader.parameters()):
            raise RuntimeError("frozen reader accumulated gradients or entered train mode")
        if [p._version for p in reader.parameters()] != versions:
            raise RuntimeError("reader modified")
        row = {"step": step, "action_fm": float(loss.detach()), "grad_norm": float(norm)}
        with (args.output / "train.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if step == 1 or step % 100 == 0:
            print("TRAIN", row, flush=True)
    path = args.output / "checkpoint.pt"
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "metadata": metadata}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(loaded["model"], strict=True)
    records = evaluate(model, reader, cache, evidence, args.output, device, args.seed)
    write_json(args.output / "results.json", {
        "complete": True, "metadata": metadata, "checkpoint_sha256": sha256(path),
        "frozen_reader_unchanged": True, "results": records,
    })


if __name__ == "__main__":
    main()
