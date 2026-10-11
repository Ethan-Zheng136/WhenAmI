"""Recover new multi-anchor clips from already registered production-DEV episodes.

Only selected dev TFRecord ordinals are deserialized; no new partition is invented.
All observed identities are checked against the same registry and training index.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
from PIL import Image

from stage3.common import STORE, DEFAULT_FUTURE
from stage3_oxe_eval.prepare import sha, write_json
from whenami_timeline_a.data import INDEX, choose_times

PARENT = STORE / "outputs/stage3_oxe_eval/dev27_64ep_01/MANIFEST.json"
INVENTORY = STORE / "outputs/stage3_joint/oxe27_inventory_01/inventory.json"


def worker(spec):
    import torch
    from stage3_scale.oxe_data import EpisodeDecoder, action_window
    from stage3.reader_study_data import episode_key
    from stage3_scale.inventory import REGISTRY, load_roles, role_for_key
    torch.set_num_threads(1)
    os.environ.update(TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1")
    out = Path(spec["output"]) / spec["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    stats = json.loads((DEFAULT_FUTURE.parents[1] / "dataset_statistics.json").read_text())
    decoder = EpisodeDecoder(spec["sources"][0], stats)
    roles = load_roles(REGISTRY)
    selected = {}
    for row in spec["episodes"]:
        selected.setdefault(row["source_id"], {})[row["source_ordinal"]] = row
    clips, rejected = [], []
    for source in spec["sources"]:
        path = Path(source["path"])
        if (path.stat().st_size, path.stat().st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
            raise ValueError("TFRecord changed")
        needed = selected[source["id"]]
        for ordinal, serialized in enumerate(decoder.iterator(path)):
            if ordinal > max(needed):
                break
            if ordinal not in needed:
                continue  # do not deserialize unrelated train/test trajectories
            row = needed[ordinal]
            actions, images, language = decoder(serialized, ordinal)
            key = episode_key(spec["dataset"], actions, language, images)
            if key != row["episode_key"] or role_for_key(spec["dataset"], key, roles) != "dev":
                raise ValueError("dev identity does not reproduce production decoder")
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
                ("timeline-a-dev01/" + key).encode()).digest()[:8], "little"))
            if len(actions) < spec["nodes"] + 1:
                rejected.append({"episode_key": key, "reason": "too_short"})
                continue
            times = None
            for _ in range(512):
                candidate = choose_times(len(actions), rng, spec["nodes"])
                texts = language[int(candidate[0]):int(candidate[-1]) + 1]
                if not texts[0].decode().strip() or any(x != texts[0] for x in texts):
                    continue
                if any(not images[t] for t in candidate):
                    continue
                times = candidate
                break
            if times is None:
                rejected.append({"episode_key": key, "reason": "no_constant_instruction_clip"})
                continue
            frames = []
            for t in times:
                with Image.open(io.BytesIO(bytes(images[t]))) as image:
                    frames.append(np.asarray(image.convert("RGB").resize((224, 224))))
            labels, masks = zip(*(action_window(actions, int(t)) for t in times))
            filename = key + ".npz"
            dest = out / filename
            tmp = dest.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(
                    f, images=np.stack(frames), raw_actions=np.stack(labels),
                    action_mask=np.stack(masks), times=times - times[0],
                    source_times=times, instructions=np.asarray([language[t].decode() for t in times]),
                )
            os.replace(tmp, dest)
            clips.append({
                "dataset": spec["dataset"], "episode_key": key, "role": "dev",
                "source_id": source["id"], "source_ordinal": ordinal,
                "source_times": times.tolist(), "file": f"{spec['dataset']}/{filename}",
                "sha256": sha(dest), "bytes": dest.stat().st_size,
            })
    result = {"dataset": spec["dataset"], "clips": clips, "rejected": rejected,
              "selected_episodes": len(spec["episodes"]), "complete": True}
    write_json(out / "COMPLETE.json", result)
    print("TIMELINE_DEV_DOMAIN", spec["dataset"], len(clips), flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--nodes", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if min(args.episodes, args.workers) < 1 or args.nodes < 2:
        p.error("positive sizes required")
    args.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads(PARENT.read_text())
    inv = json.loads(INVENTORY.read_text())
    if not parent["complete"] or parent["train_episode_overlap"]:
        raise ValueError("parent dev cohort not audited")
    grouped = {}
    for row in parent["episodes"]:
        grouped.setdefault(row["dataset"], []).append(row)
    specs = []
    for name, rows in sorted(grouped.items()):
        chosen = sorted(rows, key=lambda r: hashlib.sha256(
            ("timeline-a-dev-selection/" + r["episode_key"]).encode()).digest())[:args.episodes]
        sources = {r["source_id"] for r in chosen}
        specs.append({"dataset": name, "episodes": chosen, "nodes": args.nodes,
                      "sources": [r for r in inv["source_shards"] if r["id"] in sources],
                      "output": str(args.output)})
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(worker, spec) for spec in specs]):
            results.append(future.result())
    clips = sorted([r for result in results for r in result["clips"]],
                   key=lambda r: (r["dataset"], r["episode_key"]))
    if len({r["dataset"] for r in clips}) != 27:
        raise ValueError("missing development domain; inspect domain reports")
    actual = json.loads(INDEX.read_text())
    train_keys = {(r["dataset"], r["episode_key"]) for r in actual["episodes"]}
    dev_keys = {(r["dataset"], r["episode_key"]) for r in clips}
    if len(dev_keys) != len(clips) or train_keys & dev_keys:
        raise ValueError("duplicate or train-overlapping dev episode")
    write_json(args.output / "MANIFEST.json", {
        "complete": True, "schema": "timeline_a_dev_raw_v1", "nodes": args.nodes,
        "domains": 27, "episodes": len(clips), "train_overlap": 0,
        "parent_manifest_sha256": sha(PARENT), "train_index_sha256": sha(INDEX),
        "statistics_sha256": sha(DEFAULT_FUTURE.parents[1] / "dataset_statistics.json"),
        "preparer_sha256": sha(__file__), "clips": clips,
        "notes": ["Existing production dev episodes, new deterministic multi-anchor clips.",
                  "Base pretraining exposure not excluded; not sealed test.",
                  "Episode counts capped at16/domain; small domains retain available cases.",
                  "No data-selected checkpoint or outcome-based sample selection."],
    })
    print("TIMELINE_DEV_READY", len(clips), "episodes", flush=True)


if __name__ == "__main__":
    main()
