"""Prepare bounded existing-DEV-only OXE27 episodes, without training feature caches."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path("/mnt/pfs/zhengguantian/WoG_v2")
INVENTORY = ROOT / "outputs/stage3_joint/oxe27_inventory_01/inventory.json"
TRAIN_INDEX = ROOT / "outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json"
HISTORY = (-15, -11, -7, -3, 0)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def selected_centres(actions, images, language, key, maximum=2):
    # No future image is required for a causal action-window evaluation.
    # Match the main training requirement of at least one subsequent observation.
    if len(actions) < 2 or actions.shape[1:] != (7,) or not np.isfinite(actions).all():
        return []
    valid = [t for t in range(len(actions) - 1)
             if language[t].decode().strip()
             and all(images[max(0, t + off)] for off in HISTORY)]
    seed = int.from_bytes(hashlib.sha256(("oxe-eval35k/centres/" + key).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(valid, size=min(maximum, len(valid)), replace=False).tolist()) if valid else []


def worker(spec):
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
        os.environ.pop(k, None)
    os.environ.update(TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", WAI_BOUNDED_INPUT="1")
    import torch  # PPU image requires this import ordering.
    torch.set_num_threads(1)
    from PIL import Image
    from stage3_scale.oxe_data import EpisodeDecoder, action_window
    from stage3_scale.inventory import role_for_key, load_roles, REGISTRY
    from stage3.reader_study_data import episode_key

    name, out = spec["dataset"], Path(spec["output"]) / spec["dataset"]
    out.mkdir(parents=True, exist_ok=True)
    proof = out / "COMPLETE.json"
    if proof.exists():
        prior = json.loads(proof.read_text())
        if prior["spec_sha256"] != spec["spec_sha256"]:
            raise ValueError("preparation recipe changed")
        for row in prior["episodes"]:
            if sha(out / row["file"]) != row["sha256"]:
                raise ValueError("prepared data changed")
        return prior
    stats = json.loads(Path(spec["statistics"]).read_text())
    pinned = load_roles(REGISTRY)
    rows = sorted(spec["rows"], key=lambda r: hashlib.sha256(
        ("oxe-eval35k/shards/" + r["id"]).encode()).digest())
    decoder = EpisodeDecoder(rows[0], stats)
    seen, episodes, scanned = set(), [], 0
    counts = {"train": 0, "dev": 0, "test": 0, "invalid": 0, "duplicate": 0}
    scanned_shards, began = [], time.monotonic()
    for row in rows:
        source = Path(row["path"])
        st = source.stat()
        if (st.st_size, st.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
            raise ValueError("source changed since production inventory")
        scanned_shards.append(row["id"])
        for ordinal, serialized in enumerate(decoder.iterator(source)):
            scanned += 1
            actions, images, language = decoder(serialized, ordinal)
            if len(actions) < 2 or not np.isfinite(actions).all():
                counts["invalid"] += 1
                continue
            key = episode_key(name, actions, language, images)
            role = role_for_key(name, key, pinned)
            counts[role] += 1
            # Identity-only scan; do not decode test/train pixels or retain them.
            if role != "dev":
                continue
            if key in seen:
                counts["duplicate"] += 1
                continue
            seen.add(key)
            centres = selected_centres(actions, images, language, key, spec["windows"])
            if not centres:
                counts["invalid"] += 1
                continue
            needed = sorted({max(0, t + off) for t in centres for off in HISTORY})
            decoded = {}
            for t in needed:
                with Image.open(io.BytesIO(bytes(images[t]))) as image:
                    decoded[t] = np.asarray(image.convert("RGB").resize((224, 224)))
            frames = np.stack([np.stack([decoded[max(0, t + off)] for off in HISTORY]) for t in centres])
            targets, masks = zip(*(action_window(actions, t) for t in centres))
            payload = dict(
                frames=frames, raw_actions=np.stack(targets), action_mask=np.stack(masks),
                timestep=np.asarray(centres, np.int64),
                frame_indices=np.asarray([[max(0, t + off) for off in HISTORY] for t in centres]),
                instructions=np.asarray([language[t].decode() for t in centres]),
            )
            filename = key + ".npz"
            target = out / filename
            temp = target.with_suffix(f".tmp-{os.getpid()}")
            with temp.open("wb") as f:
                np.savez_compressed(f, **payload)
            os.replace(temp, target)
            episodes.append({
                "dataset": name, "episode_key": key, "role": role,
                "source_id": row["id"], "source_ordinal": ordinal,
                "length": len(actions), "centres": centres, "windows": len(centres),
                "history_padded": [t < 15 for t in centres],
                "action_padded": [t + 16 > len(actions) for t in centres],
                "file": filename, "sha256": sha(target), "bytes": target.stat().st_size,
            })
            if len(episodes) % 8 == 0:
                print("OXE_DEV_PREP", name, len(episodes), "episodes", scanned,
                      "scanned", round(time.monotonic() - began, 1), "seconds", flush=True)
            if len(episodes) >= spec["episodes"]:
                break
        if len(episodes) >= spec["episodes"]:
            break
    result = {
        "complete": True, "spec_sha256": spec["spec_sha256"], "dataset": name,
        "episode_count": len(episodes), "window_count": sum(e["windows"] for e in episodes),
        "quota_met": len(episodes) == spec["episodes"],
        "all_source_shards_reached": len(scanned_shards) == len(rows),
        "source_scan_counts": counts, "records_scanned": scanned, "scanned_shards": scanned_shards,
        "seconds": time.monotonic() - began, "episodes": episodes,
    }
    write_json(proof, result)
    print("OXE_DEV_DOMAIN_COMPLETE", name, len(episodes), "episodes", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.episodes, args.windows, args.workers) < 1:
        p.error("positive limits required")
    from stage3_scale.inventory import REGISTRY
    from stage3_scale.oxe_data import oxe_source_hashes
    inventory = json.loads(INVENTORY.read_text())
    if sha(REGISTRY) != inventory["prior_role_registry_sha256"]:
        raise ValueError("production role registry changed")
    prep_recipe = json.loads((TRAIN_INDEX.parent / "preparation_recipe.json").read_text())
    if oxe_source_hashes() != prep_recipe["oxe_sources"]:
        raise ValueError("production standardization changed")
    recipe = {
        "schema": "stage3_oxe27_existing_dev_v1",
        "created_utc_protocol_date": "2026-10-03",
        "target_episodes_per_domain": args.episodes, "windows_per_episode": args.windows,
        "inventory_sha256": sha(INVENTORY), "registry_sha256": sha(REGISTRY),
        "statistics_sha256": sha(inventory["statistics_path"]),
        "preparer_sha256": sha(__file__), "oxe_sources": oxe_source_hashes(),
        "history_offsets": HISTORY,
        "role": "existing_production_dev_only",
        "base_pretraining_exposure": "not guaranteed absent",
    }
    recipe = json.loads(json.dumps(recipe))
    args.output.mkdir(parents=True, exist_ok=args.resume)
    recipe_path = args.output / "recipe.json"
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError("cannot resume with a different preparation recipe")
    write_json(recipe_path, recipe)
    specs = []
    for d in inventory["datasets"]:
        spec = {
            "dataset": d["dataset"], "rows": [r for r in inventory["source_shards"] if r["dataset"] == d["dataset"]],
            "statistics": inventory["statistics_path"], "output": str(args.output.resolve()),
            "episodes": args.episodes, "windows": args.windows, "recipe": recipe,
        }
        spec["spec_sha256"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        specs.append(spec)
    done = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(worker, spec): spec["dataset"] for spec in specs}
        for future in as_completed(futures):
            done.append(future.result())
    # Audit against ACTUAL emitted training identities, not only the intended hash.
    actual = json.loads(TRAIN_INDEX.read_text())
    trained = {(e["dataset"], e["episode_key"]) for e in actual["episodes"]}
    selected = [e for d in sorted(done, key=lambda d: d["dataset"]) for e in d["episodes"]]
    keys = [(e["dataset"], e["episode_key"]) for e in selected]
    if len(keys) != len(set(keys)) or any(k in trained for k in keys):
        raise ValueError("duplicate or train-exposed evaluation episode")
    if any(d["episode_count"] == 0 for d in done):
        raise ValueError("at least one domain has no eligible dev episodes; report before evaluating")
    write_json(args.output / "MANIFEST.json", {
        "complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": recipe, "train_index_sha256": sha(TRAIN_INDEX),
        "train_episode_overlap": 0, "domain_count": len(done), "episode_count": len(selected),
        "window_count": sum(e["windows"] for e in selected), "episodes": selected,
        "domains": [{k: v for k, v in d.items() if k != "episodes"} for d in sorted(done, key=lambda d: d["dataset"])],
        "notes": ["Existing Stage3 development partition, not sealed test.",
                  "Source-record parsing for partition identity; only dev pixels decoded/retained.",
                  "Base pretraining exposure cannot be ruled out.", "Quota-bounded shard-cluster diagnostic sample."],
    })
    print("OXE27_DEV_READY", len(done), len(selected), "episodes", flush=True)


if __name__ == "__main__":
    main()
