"""Read real episodes in every original OXE domain; no performance claims.

Checks original standardization, primary-camera bytes, language,7D actions,
normalization, deterministic role identity, and actual image decoding.
"""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from stage3.common import DEFAULT_PAST, write_json, sha256
from stage3_scale.oxe_data import EpisodeDecoder, valid_centres, oxe_source_hashes, frame_indices, action_window
from stage3.reader_study_data import episode_key
from stage3_scale.inventory import role_for_key
from evaluation.past_protocol import normalize_actions


def audit_domain(rows, statistics, sample_target=2):
    decoder = EpisodeDecoder(rows[0], statistics)
    samples, scanned = [], 0
    for row in rows[:4]:
        for j, serialized in enumerate(decoder.iterator(row["path"])):
            scanned += 1
            actions, images, language = decoder(serialized, j)
            if not np.isfinite(actions).all():
                continue
            centres = valid_centres(actions, images, language)
            if not centres:
                continue
            key = episode_key(row["dataset"], actions, language, images)
            if role_for_key(row["dataset"], key, {}) != "train":
                continue
            t = centres[len(centres) // 2]
            image_ids = frame_indices(t, len(actions))
            for k in (image_ids[0], t, image_ids[-1]):
                with Image.open(io.BytesIO(images[k])) as im:
                    im.convert("RGB").resize((224, 224)).load()
            target, action_mask = action_window(actions, t)
            normalized = normalize_actions(target, statistics[row["dataset"]]["action"])
            assert normalized.shape == (16, 7) and np.isfinite(normalized).all()
            actions2, images2, language2 = decoder(serialized, j)
            assert np.array_equal(actions, actions2)
            assert np.array_equal(images, images2)
            assert key == episode_key(row["dataset"], actions2, language2, images2)
            samples.append({
                "source": row["id"], "source_episode_ordinal": j, "length": len(actions),
                "action_shape": list(actions.shape), "primary_camera": decoder.primary_key,
                "valid_window_count": len(centres), "selected_timestep": t,
                "valid_action_steps": int(action_mask.sum()),
                "history_boundary_repeat": t < 15, "future_boundary_repeat": t + 15 >= len(actions),
                "episode_key": key, "partition": "train", "instruction": language[t].decode(),
                "normalized_action_min": float(normalized.min()), "normalized_action_max": float(normalized.max()),
                "actual_images_decode": True, "repeat_identity_matches": True,
            })
            if len(samples) >= sample_target:
                break
            if scanned >= 400:
                break
        if len(samples) >= sample_target or scanned >= 400:
            break
    if len(samples) < sample_target:
        raise RuntimeError(f"{rows[0]['dataset']}: found {len(samples)} usable windows after{scanned} source episodes")
    return {"dataset": rows[0]["dataset"], "complete": True, "scanned_episodes": scanned, "samples": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        os.environ.pop(k, None)
    args.output.mkdir(exist_ok=False, parents=True)
    inventory = json.loads(args.inventory.read_text())
    stats = json.loads((DEFAULT_PAST.parents[1] / "dataset_statistics.json").read_text())
    results = []
    for domain in inventory["datasets"]:
        name = domain["dataset"]
        rows = [r for r in inventory["source_shards"] if r["dataset"] == name]
        print("AUDIT_START", name, flush=True)
        try:
            result = audit_domain(rows, stats, args.samples)
        except Exception as error:
            import traceback
            result = {"dataset": name, "complete": False, "error": repr(error), "traceback": traceback.format_exc()}
        write_json(args.output / f"{name}.json", result)
        results.append(result)
        print("AUDIT_DONE", name, result["complete"], result.get("error", ""), flush=True)
    report = {
        "complete": all(r["complete"] for r in results), "dataset_count": len(results),
        "passed": sum(r["complete"] for r in results), "results": results,
        "inventory_sha256": sha256(args.inventory), "oxe_source_sha256": oxe_source_hashes(),
        "decoder_sha256": sha256(Path(__file__).with_name("oxe_data.py")),
        "notes": ["Small real train-episode compatibility audit per dataset, not complete corpus conversion.",
                  "Per-sample states are raw dataset entries, not model metrics.",
                  "No dev/test episode samples retained or evaluated."],
    }
    write_json(args.output / "results.json", report)
    if not report["complete"]:
        raise SystemExit("Some domain audits failed; no full OXE readiness claimed")


if __name__ == "__main__":
    main()
