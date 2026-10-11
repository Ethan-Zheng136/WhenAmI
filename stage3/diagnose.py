"""Read-only diagnosis of the frozen-endpoint pilot, not architecture selection.

Feature swaps are off-manifold interventions, NOT isolated history ablations:
past H contains both past and current evidence. All donors/statistics come
from connector TRAIN episodes; destination-reader z and action labels stay put.
Gradient comparisons use paired TRAIN batches and do not update any weights.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stage3.behavior import load_behavior
from stage3.cache import EndpointCache
from stage3.common import sha256, write_json
from stage3.connector import ConnectorConfig, SharedConnector
from stage3.objectives import connector_inputs, directional_losses


def donor_indices(cache, instructions, indices, seed):
    """Prefer same instruction, always same dataset and a different episode."""
    rng = np.random.default_rng(seed)
    donors, matched = [], []
    text = np.asarray([x.strip().lower() for x in instructions])
    for i in indices:
        candidates = cache.train_indices[
            (cache.datasets[cache.train_indices] == cache.datasets[i])
            & (cache.episodes[cache.train_indices] != cache.episodes[i])
        ]
        if not len(candidates):
            raise ValueError("no different-episode training donor")
        same = candidates[text[candidates] == text[i]]
        matched.append(bool(len(same)))
        donors.append(int(rng.choice(same if len(same) else candidates)))
    return np.asarray(donors), np.asarray(matched)


def gradient_relation(first, second):
    dot = sum((x.double() * y.double()).sum() for x, y in zip(first, second))
    aa = sum(x.double().square().sum() for x in first)
    bb = sum(y.double().square().sum() for y in second)
    norm_a, norm_b = float(aa.sqrt()), float(bb.sqrt())
    return {
        "cosine": float(dot / (aa.sqrt() * bb.sqrt()).clamp_min(1e-30)),
        "first_norm": norm_a, "second_norm": norm_b,
        "second_over_first_norm": norm_b / max(norm_a, 1e-30),
    }


def summarize_relations(rows):
    return {
        key: {
            "mean": float(np.mean([r[key] for r in rows])),
            "median": float(np.median([r[key] for r in rows])),
            "min": float(min(r[key] for r in rows)),
            "max": float(max(r[key] for r in rows)),
        } for key in rows[0]
    } | {"negative_cosine_fraction": float(np.mean([r["cosine"] < 0 for r in rows]))}


def shared_parameters(connector):
    # Direction embeddings and directional adapters are not truly common
    # coordinates, despite having been packed into a single Parameter.
    return [p for name, p in connector.named_parameters() if
            name.startswith(("blocks.", "final_norm.")) or name == "query_position"]


@torch.no_grad()
def feature_interventions(connector, reader, cache, device, output, seed):
    instructions = json.loads((cache.path / "instructions.json").read_text())
    indices = cache.valid_indices
    donors, matched = donor_indices(cache, instructions, indices, seed)
    write_json(output / "intervention_donors.json", {
        "indices": indices.tolist(), "donors": donors.tolist(),
        "same_instruction": matched.tolist(),
    })
    result = []
    for dataset in sorted(set(cache.datasets.tolist())):
        selection = np.flatnonzero(cache.datasets[indices] == dataset)
        ids, ds = indices[selection], donors[selection]
        per_variant = {}
        for start in range(0, len(ids), 8):
            target = cache.batch(ids[start:start + 8], device)
            source = cache.batch(ds[start:start + 8], device)
            features = {
                "intact": (target["past_h"], target["past_z"]),
                "swap_h": (source["past_h"], target["past_z"]),
                "swap_z": (target["past_h"], source["past_z"]),
                "swap_hz": (source["past_h"], source["past_z"]),
            }
            for variant, (h, z) in features.items():
                pred = connector.predict(h, z, "forward")
                align = connector.alignment_loss(pred, target["future_c"], "future")
                fm = reader.loss(target["future_actions"], target["future_z"], pred, seed + start)
                per_variant.setdefault(variant, []).append((len(h), float(align), float(fm)))
        for variant, records in per_variant.items():
            result.append({
                "dataset": dataset, "variant": variant, "n": len(ids),
                "same_instruction_donor_fraction": float(matched[selection].mean()),
                "alignment": sum(n * a for n, a, _ in records) / len(ids),
                "action_fm": sum(n * f for n, _, f in records) / len(ids),
            })
    return result


def gradient_diagnosis(connector, readers, cache, device, batches, seed):
    params = shared_parameters(connector)
    rng = np.random.default_rng(seed)
    records = []
    for dataset in sorted(set(cache.datasets.tolist())):
        available = cache.train_indices[cache.datasets[cache.train_indices] == dataset]
        for j in range(batches):
            ids = rng.choice(available, 8, replace=True)
            batch = cache.batch(ids, device)
            pred = connector(*connector_inputs(batch, "bidirectional"))
            _, values = directional_losses(connector, pred, batch, readers, seed + j)
            names = ["forward_alignment", "forward_action_fm", "backward_alignment", "backward_action_fm"]
            grads = {}
            for k, name in enumerate(names):
                grads[name] = torch.autograd.grad(
                    values[name], params, retain_graph=k < len(names) - 1,
                )
            fg = tuple(x + y for x, y in zip(grads[names[0]], grads[names[1]]))
            bg = tuple(x + y for x, y in zip(grads[names[2]], grads[names[3]]))
            relationships = {
                "forward_total_vs_backward_total": gradient_relation(fg, bg),
                "forward_action_vs_alignment": gradient_relation(grads[names[1]], grads[names[0]]),
                "backward_action_vs_alignment": gradient_relation(grads[names[3]], grads[names[2]]),
                "forward_action_vs_backward_action": gradient_relation(grads[names[1]], grads[names[3]]),
            }
            records.append({
                "dataset": dataset, "batch": j, "indices": ids.tolist(),
                "losses": {name: float(values[name].detach()) for name in names},
                "relationships": relationships,
            })
    summary = {}
    for dataset in sorted(set(cache.datasets.tolist())):
        subset = [r for r in records if r["dataset"] == dataset]
        summary[dataset] = {
            name: summarize_relations([r["relationships"][name] for r in subset])
            for name in subset[0]["relationships"]
        }
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradient-batches", type=int, default=16, help="per dataset")
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.gradient_batches < 1:
        raise ValueError("positive gradient batch count required")
    args.output.mkdir(parents=True, exist_ok=False)
    cp = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    cache = EndpointCache(cp["metadata"]["cache"])
    if cache.manifest_sha256 != cp["metadata"]["cache_manifest_sha256"]:
        raise ValueError("checkpoint/cache mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    connector = SharedConnector(ConnectorConfig(**cp["metadata"]["connector"])).to(device).eval()
    connector.load_state_dict(cp["connector"], strict=True)
    before = {k: v.detach().clone() for k, v in connector.state_dict().items()}
    readers = {s: load_behavior(cache.manifest["endpoints"][s], device) for s in ("future", "past")}
    versions = {s: [p._version for p in r.parameters()] for s, r in readers.items()}
    interventions = feature_interventions(connector, readers["future"], cache, device, args.output, args.seed)
    print("INTERVENTIONS", json.dumps(interventions), flush=True)
    records, gradients = gradient_diagnosis(
        connector, readers, cache, device, args.gradient_batches, args.seed,
    )
    for k, v in connector.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    for s, r in readers.items():
        assert [p._version for p in r.parameters()] == versions[s]
        assert not r.training and all(p.grad is None for p in r.parameters())
    write_json(args.output / "gradient_batches.json", records)
    report = {
        "complete": True, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint), "cache_manifest_sha256": cache.manifest_sha256,
        "precision": "float32", "seed": args.seed,
        "feature_interventions": interventions, "shared_gradient_relations": gradients,
        "notes": [
            "H contains current as well as historical evidence; this is NOT an isolated history test.",
            "Feature swapping breaks cross-source coherence; FM changes are not closed-loop outcomes.",
            "Donors from same dataset, different TRAIN episode; instruction match preferred, not always possible.",
            "Reader z/actions unchanged; R never receives action labels or target-side features.",
            "Gradients measured on TRAIN batches, shared blocks/final norm/query positions only; no updates.",
            "Cosine of local gradients alone cannot establish the cause of poor generalization.",
        ],
    }
    write_json(args.output / "results.json", report)
    print("GRADIENTS", json.dumps(gradients), flush=True)


if __name__ == "__main__":
    main()
