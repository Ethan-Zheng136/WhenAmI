"""Run portable causal fixtures; report numerical differences, not task success."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from whenami_timeline_b_handoff.runtime import build, load_release, prefix_inputs, sample, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--require-bitwise", action="store_true",
                   help="Only suitable for same-platform roundtrip, not a cross-device assumption")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = json.loads((args.release / "MODEL.json").read_text())
    for filename, key in (("config.json", "config_sha256"),
                          ("dataset_statistics.json", "statistics_sha256")):
        if sha(args.release / filename) != manifest[key]:
            raise ValueError(filename + " changed")
    config = json.loads((args.release / "config.json").read_text())
    student, visual, transform, tokenizer, prompt = build(config, args.device)
    load_release(student, args.release)
    rows = []
    for fixture in sorted((args.release / "fixtures").glob("*.npz")):
        info = json.loads(fixture.with_suffix(".json").read_text())
        with np.load(fixture) as data:
            prefix = prefix_inputs(data["images"], info["instructions"], data["times"],
                                   visual, transform, tokenizer, prompt, args.device)
            actual = sample(student, prefix, data["initial_noise"])
            metrics = {}
            for key, value in actual.items():
                expected = data[key]
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError("invalid replay output: " + key)
                delta = value.astype(np.float64) - expected.astype(np.float64)
                metrics[key] = {
                    "max_abs": float(np.abs(delta).max()),
                    "rmse": float(np.sqrt(np.mean(delta ** 2))),
                    "bitwise_equal": bool(np.array_equal(value, expected)),
                }
            if args.require_bitwise and not all(v["bitwise_equal"] for v in metrics.values()):
                raise ValueError("requested same-platform bitwise replay failed")
            rows.append({"fixture": fixture.name, "metrics": metrics})
            np.savez_compressed(args.output / fixture.name, **actual)
    if len(rows) != 2:
        raise ValueError("expected Bridge and Fractal numerical fixtures")
    result = {
        "complete": True, "model_variant": manifest["variant"], "step": manifest["step"],
        "torch": torch.__version__, "device": torch.cuda.get_device_name(args.device),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 2**30,
        "fixtures": rows,
        "scope": "Finite causal inference and numerical comparison, not SIMPLER success. "
                 "Cross-platform differences require review; exit 0 alone is not acceptance.",
    }
    (args.output / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
