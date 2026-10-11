"""Read-only real OXE multi-anchor audit, one eligible train episode per domain."""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from whenami_revision.raw_data import RevisionEpisodes, tensorize
from whenami_revision.contracts import revision_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, default=Path(
        "/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    data = RevisionEpisodes(args.index)
    # The registry/rule is exactly the current Stage3 train split; no sealed/dev reads.
    from stage3_scale.inventory import REGISTRY, load_roles, role_for_key
    assert hashlib.sha256(REGISTRY.read_bytes()).hexdigest() == data.index["prior_role_registry_sha256"]
    pinned = load_roles(REGISTRY)
    rows = []
    for domain, entries in sorted(data.episodes.items()):
        chosen = None
        rejected = []
        begin = len(entries) // 2
        for j in range(min(100, len(entries))):
            entry = entries[(begin + j) % len(entries)]
            if entry["length"] < 4:
                continue
            assert role_for_key(domain, entry["episode_key"], pinned) == "train"
            cutoff = min(entry["length"] - 2, max(2, entry["length"] // 2))
            anchors = tuple(sorted(set([max(0, cutoff - 6), max(0, cutoff - 3), cutoff - 1, cutoff])))
            try:
                earlier = data.prefix(entry, anchors, anchors[1])
                later = data.prefix(entry, anchors, cutoff)
            except ValueError as error:
                rejected.append(str(error))
                continue
            if any(not s.strip() for s in later.instructions):
                rejected.append("empty instruction")
                continue
            pair = revision_pair(earlier, later, anchors[0])
            view = tensorize(later, lambda image: torch.from_numpy(
                np.array(image.resize((8, 8)), copy=True)).float().mean((0, 1)) / 255.)
            labels = data.action_labels(entry, later)
            assert view["times"][0].tolist() == list(np.asarray(anchors) - anchors[0])
            assert int(view["executed_mask"].sum()) == cutoff - anchors[0]
            assert all(t < cutoff for times in later.interval_action_times for t in times)
            chosen = {
                "dataset": domain, "episode_key": entry["episode_key"], "role": "train",
                "source_frames": anchors, "actual_delta_frames": np.diff(anchors).tolist(),
                "event_id": pair["event_id"], "earlier_evidence_cutoff": earlier.cutoff,
                "later_evidence_cutoff": later.cutoff,
                "policy_tensor_keys": sorted(view),
                "executed_actions_count": int(view["executed_mask"].sum()),
                "current_action_label_valid_steps": int(labels.valid.sum()),
                "observation_decode_passed": True, "event_state_labels_available": False,
                "skipped_candidates": len(rejected),
            }
            break
        if chosen is None:
            raise RuntimeError(f"No eligible audit episode in {domain}: {rejected}")
        rows.append(chosen)
        print("REVISION_RAW_CONTRACT_OK", domain, chosen["source_frames"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "complete": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
        "domains_passed": len(rows), "cases": rows,
        "input_index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "training_or_eval_modified": False, "sealed_test_accessed": False,
        "note": "Actual train-image/action readback only. No event labels fabricated and no model performance claim.",
    }, indent=2))


if __name__ == "__main__":
    main()
