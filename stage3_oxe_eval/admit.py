"""One-model-at-a-time local loading/sampling admission and Stage3 golden replay."""

import argparse
from pathlib import Path
import json
import time

import numpy as np
import torch

from stage3.common import STORE
from stage3_oxe_eval.policy import Policy, KINDS
from stage3_oxe_eval.prepare import write_json
from evaluation.past_protocol import sample_seed
from stage3.evaluate import denormalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    args.output.mkdir(parents=True, exist_ok=False)
    root = STORE / "outputs/stage3_reader_study/data_01/train_dev"
    ids = np.load(root / "valid_indices.npy")
    datasets, episodes, times = [np.load(root / (x + ".npy")) for x in ("dataset", "episode", "timestep")]
    frames = np.load(root / "frames.npy", mmap_mode="r")
    instructions = json.loads((root / "instructions.json").read_text())
    manifest = json.loads((root / "samples_manifest.json").read_text())
    offsets = manifest["frame_offsets"]
    policy = Policy(args.kind)
    results = []
    for name in ("bridge_orig", "fractal20220817_data"):
        i = int(ids[datasets[ids] == name][0])
        now = frames[i, offsets.index(0)]
        history = frames[i, [offsets.index(t) for t in (-15, -11, -7, -3)]]
        torch.cuda.synchronize()
        started = time.monotonic()
        encoded = policy.encode(now, history, instructions[i])
        seed = sample_seed(name, episodes[i], times[i], 42)
        noise = torch.randn((1, 16, 7), device=policy.device,
                            generator=torch.Generator(device=policy.device).manual_seed(seed))
        prediction = policy.sample(encoded, noise)
        torch.cuda.synchronize()
        result = {"dataset": name, "index": i, "seconds": time.monotonic() - started,
                  "cognition_original_path": policy.check_cognition(now, history, instructions[i])}
        if args.kind == "past":
            result["past_condition_original_path"] = policy.check_past_condition(now, history, instructions[i])
        if args.kind == "stage3":
            with np.load(policy.checkpoint.parents[1] / "development/step-035000" / f"{name}_online.npz") as old:
                j = np.flatnonzero(old["indices"] == i).item()
                previous = old["predictions"][j]
            raw = denormalize(prediction, policy.statistics[name]["action"])[0]
            error = np.abs(raw - previous)
            # Report actual errors; fail on substantial action drift. This is
            # deployment admission, not a claim of cross-runtime bitwise equality.
            new_norm = prediction[0].copy()
            bounds = policy.statistics[name]["action"]
            from evaluation.past_protocol import normalize_actions
            old_norm = normalize_actions(previous, bounds)
            new_norm = normalize_actions(raw, bounds)
            norm_error = np.abs(new_norm - old_norm)
            result.update(raw_max_abs=float(error.max()), normalized_max_abs=float(norm_error.max()),
                          normalized_rmse=float(np.sqrt(np.square(norm_error).mean())))
            if norm_error.max() > .02 or np.sqrt(np.square(norm_error).mean()) > .005:
                raise ValueError("Stage3 golden replay drift: " + str(result))
        np.savez_compressed(args.output / f"{name}.npz", prediction=prediction,
                            initial_noise=noise.cpu().numpy(), current=now, history=history)
        results.append(result)
        print("ADMISSION_SAMPLE", result, flush=True)
    write_json(args.output / "RESULT.json", {
        "complete": True, "kind": args.kind, "policy": policy.record, "results": results,
        "note": "Two old-dev samples are loader replay fixtures, not the planned multi-domain evaluation.",
    })


if __name__ == "__main__":
    main()
