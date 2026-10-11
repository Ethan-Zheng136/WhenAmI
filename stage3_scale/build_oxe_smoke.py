"""Build an explicitly PARTIAL 27-domain raw fixture from audited TRAIN episodes.

Not a replacement for full preparation: only two real train episodes per domain.
Allows testing weighted batches, padding masks and32-card admission without
waiting for conversion of the5TB source corpus.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from stage3.common import DEFAULT_PAST, sha256, write_json
from stage3_scale.oxe_data import EpisodeDecoder, valid_centres, finalize, oxe_source_hashes
from stage3.reader_study_data import episode_key
from stage3_scale.inventory import load_roles, role_for_key, REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        os.environ.pop(k, None)
    audit = json.loads(args.audit.read_text())
    inventory = json.loads(args.inventory.read_text())
    if not audit["complete"] or audit["passed"] != 27 or audit["inventory_sha256"] != sha256(args.inventory):
        raise ValueError("complete matching27-domain audit required")
    if audit["decoder_sha256"] != sha256(Path(__file__).with_name("oxe_data.py")):
        raise ValueError("decoder changed since real-domain audit")
    args.output.mkdir(parents=True, exist_ok=False)
    stats = json.loads((DEFAULT_PAST.parents[1] / "dataset_statistics.json").read_text())
    pinned = load_roles(REGISTRY)
    write_json(args.output / "preparation_recipe.json", {
        "smoke_only": True, "audit_sha256": sha256(args.audit),
        "inventory_sha256": sha256(args.inventory), "oxe_source": oxe_source_hashes(),
        "decoder_sha256": audit["decoder_sha256"],
    })
    for domain in audit["results"]:
        selected_by_source = {}
        for sample in domain["samples"]:
            selected_by_source.setdefault(sample["source"], {})[sample["source_episode_ordinal"]] = sample["episode_key"]
        for source, selected in selected_by_source.items():
            row = next(r for r in inventory["source_shards"] if r["id"] == source)
            decoder = EpisodeDecoder(row, stats)
            records, chunks, pointer = [], [], 0
            for ordinal, serialized in enumerate(decoder.iterator(row["path"])):
                if ordinal in selected:
                    actions, images, language = decoder(serialized, ordinal)
                    key = episode_key(row["dataset"], actions, language, images)
                    if key != selected[ordinal] or role_for_key(row["dataset"], key, pinned) != "train":
                        raise ValueError("audited sample identity or partition changed")
                    spans = []
                    for frame in images:
                        spans.append((pointer, len(frame)))
                        pointer += len(frame)
                        chunks.append(frame)
                    records.append({
                        "dataset": row["dataset"], "episode_key": key,
                        "raw_actions": torch.from_numpy(actions.copy()),
                        "frame_spans": torch.tensor(spans, dtype=torch.int64),
                        "instructions": [s.decode() for s in language],
                        "centres": torch.tensor(valid_centres(actions, images, language), dtype=torch.int64),
                        "length": len(actions),
                    })
                if ordinal >= max(selected):
                    break
            if len(records) != len(selected):
                raise ValueError("audited fixture episode missing")
            path = args.output / "shards" / row["dataset"] / (source.split("/")[-1] + ".pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"schema": "whenami_raw_train_shard_v1", "source_id": source,
                        "episodes": records, "frames": torch.from_numpy(
                            np.frombuffer(b"".join(chunks), np.uint8).copy())}, path)
            write_json(path.with_suffix(".json"), {
                "complete": True, "smoke_subset": True, "source_id": source, "dataset": row["dataset"],
                "file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path),
                "windows": sum(len(r["centres"]) for r in records),
                "episodes": [{"episode_key": r["episode_key"], "length": r["length"],
                              "windows": len(r["centres"])} for r in records],
            })
        print("OXE_SMOKE_DOMAIN", domain["dataset"], len(domain["samples"]), flush=True)
    index = finalize(args.output, inventory, allow_partial=True)
    assert not index["complete"] and index["domain_count"] == 27


if __name__ == "__main__":
    main()
