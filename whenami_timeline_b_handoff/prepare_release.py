"""Export a matched B0/B1 pair, source snapshot and causal numerical fixtures.

Only reads committed training artifacts. Runs on one free accelerator; no
optimizer state, simulator, or teacher is loaded.
"""
import argparse
import gc
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from stage3.common import BASE, REFERENCE, STORE, DEFAULT_FUTURE
from whenami_timeline_b_handoff.runtime import (
    SCHEMA, build, load_dcp, load_release, prefix_inputs, sample, sha,
)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def export(student, root, source, config, source_meta):
    root.mkdir()
    shards, group, size, entries = [], {}, 0, 0

    def flush():
        nonlocal group, size
        if not group:
            return
        name = f"model-{len(shards) + 1:03d}.safetensors"
        save_file(group, str(root / name))
        shards.append({"file": name, "sha256": sha(root / name), "keys": list(group),
                       "bytes": (root / name).stat().st_size})
        group, size = {}, 0

    for key, tensor in student.state_dict().items():
        n = tensor.numel() * tensor.element_size()
        if size and size + n > 512 * 1024**2:
            flush()
        group[key] = tensor.detach().cpu().contiguous()
        size += n
        entries += 1
    flush()
    write(root / "config.json", config)
    shutil.copy2(DEFAULT_FUTURE.parents[1] / "dataset_statistics.json",
                 root / "dataset_statistics.json")
    write(root / "MODEL.json", {
        "schema": SCHEMA, "complete": True,
        "variant": source_meta["recipe"]["settings"]["variant"], "step": source_meta["step"],
        "source_checkpoint": str(source),
        "source_committed_sha256": sha(source / "COMMITTED.json"),
        "source_dcp_metadata_sha256": sha(source / "model_dcp/.metadata"),
        "teacher": source_meta["recipe"]["teacher"],
        "state_entries": entries, "shards": shards,
        "config_sha256": sha(root / "config.json"),
        "statistics_sha256": sha(root / "dataset_statistics.json"),
        "precision": "BF16 parameters, original FP32 buffers; no quantization",
        "scope": "Complete causal B student, including unused auxiliary head; no teacher/optimizer/RNG.",
    })


def fixtures():
    root = STORE / "outputs/whenami_timeline_a/dev27_16ep_01"
    manifest = json.loads((root / "MANIFEST.json").read_text())
    result = []
    for i, dataset in enumerate(("bridge_orig", "fractal20220817_data")):
        record = next(r for r in manifest["clips"] if r["dataset"] == dataset)
        path = root / record["file"]
        if sha(path) != record["sha256"]:
            raise ValueError("DEV file changed")
        with np.load(path) as archive:
            # Future image, future instruction and action labels never leave this reader.
            observed = {
                "images": archive["images"][:2].copy(),
                "times": archive["times"][:2].copy(),
                "instructions": archive["instructions"][:2].tolist(),
            }
        noise = np.random.default_rng(20261005 + i).standard_normal((1, 16, 7)).astype(np.float32)
        result.append((dataset, observed, noise, record))
    return result


def input_parity(observed, native, visual, transform, tokenizer, prompt, device):
    from whenami_timeline_b.data import student_inputs
    case = {
        "observations": {**observed, "times": torch.from_numpy(observed["times"].copy())},
        "labels": {"actions": torch.zeros(2, 16, 7), "action_mask": torch.ones(2, 16, dtype=torch.bool)},
        "provenance": {"dataset": "numerical_fixture", "episode_key": "observed_prefix_only"},
    }
    reference, _ = student_inputs([case], visual, transform, tokenizer, prompt, device)
    for name in ("input_ids", "attention_mask", "patches", "vae", "times"):
        if not torch.equal(getattr(native, name), getattr(reference, name)):
            raise ValueError("native/production prefix mismatch: " + name)
    if not all(torch.equal(native.pixels[k], reference.pixels[k]) for k in native.pixels):
        raise ValueError("native/production pixels mismatch")


def snapshot(root, recipes):
    code = root / "code"
    code.mkdir()
    # Check the exact running sources before making a private receiving-server snapshot.
    for recipe in recipes:
        for field, base in (("source", BASE), ("reference_source", REFERENCE)):
            for name, expected in recipe[field].items():
                if sha(base / name) != expected:
                    raise ValueError("source contract drift: " + name)
    directories = (
        "whenami_timeline_b_handoff", "whenami_timeline_b", "whenami_timeline_joint",
        "whenami_timeline_a", "whenami_timeline_transfer", "whenami_revision",
        "stage3", "stage3_scale", "stage3_perf", "stage3_oxe_eval", "stage3_oxe_eval50",
        "evaluation", "WoG_16rank_probe",
    )
    for directory in directories:
        for path in (BASE / directory).rglob("*.py"):
            destination = code / path.relative_to(BASE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    tokenizer = Path("WoG_16rank_probe/prismatic/models/backbones/llm/llama2_config")
    shutil.copytree(BASE / tokenizer, code / tokenizer, dirs_exist_ok=True)
    for name in ("LICENSE", "LICENSE.md"):
        for source_root, target_root in ((BASE, code), (REFERENCE, code / "WoG_16rank_probe")):
            if (source_root / name).is_file():
                shutil.copy2(source_root / name, target_root / name)
    vision = code / "WoG_16rank_probe/pretrained/vision"
    vision.mkdir(parents=True, exist_ok=True)
    for name in ("dinov2_weights.pth", "Wan2.1_VAE.pth"):
        shutil.copy2(STORE / "ckpts/vision" / name, vision / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = "cuda:0"
    variants = ("action_only", "condition_align")
    sources = [
        STORE / "outputs/whenami_timeline_b/training" /
        f"timeline_b_{variant}_A10k_320_20k_01/main/checkpoints/step-{args.step:06d}"
        for variant in variants
    ]
    metas = [json.loads((source / "COMMITTED.json").read_text()) for source in sources]
    for variant, meta in zip(variants, metas):
        if not meta.get("complete") or meta["step"] != args.step or meta["recipe"]["settings"]["variant"] != variant:
            raise ValueError("not the requested committed B pair")
    comparable = [json.loads(json.dumps(meta["recipe"])) for meta in metas]
    for recipe in comparable:
        recipe["settings"].pop("variant")
    if comparable[0] != comparable[1]:
        raise ValueError("unmatched B pair")
    config = json.loads((DEFAULT_FUTURE.parents[1] / "config.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "WORK_STATUS.json", {"complete": False, "phase": "building"})
    cases = fixtures()
    student, visual, transform, tokenizer, prompt = build(config, device)
    report = {}
    for variant, source, source_meta in zip(variants, sources, metas):
        print("LOADING_B_DCP", variant, source, flush=True)
        load_dcp(student, source)
        root = args.output / variant
        traces = []
        for name, observed, noise, record in cases:
            prefix = prefix_inputs(**observed, visual=visual, transform=transform,
                                   tokenizer=tokenizer, prompt=prompt, device=device)
            input_parity(observed, prefix, visual, transform, tokenizer, prompt, device)
            traces.append(sample(student, prefix, noise))
        export(student, root, source, config, source_meta)
        # Load every exported tensor back, then replay fixed noise through all nine steps.
        load_release(student, root)
        checks = []
        for (name, observed, noise, record), trace in zip(cases, traces):
            prefix = prefix_inputs(**observed, visual=visual, transform=transform,
                                   tokenizer=tokenizer, prompt=prompt, device=device)
            actual = sample(student, prefix, noise)
            errors = {k: float(np.max(np.abs(actual[k] - trace[k]))) for k in trace}
            if any(v != 0 for v in errors.values()):
                raise ValueError("same-device export/reload is not bitwise equivalent")
            fixture_dir = root / "fixtures"
            fixture_dir.mkdir(exist_ok=True)
            np.savez_compressed(fixture_dir / f"{name}.npz",
                                images=observed["images"], times=observed["times"],
                                initial_noise=noise, **trace)
            write(fixture_dir / f"{name}.json", {
                "instructions": observed["instructions"], "dataset": name,
                "episode_key": record["episode_key"],
                "source_dev_sha256": record["sha256"],
                "scope": "Two observed frames only. Numerical acceptance fixture, not a benchmark score.",
            })
            checks.append({"dataset": name, "export_reload_max_abs": errors,
                           "production_prefix_bitwise_equal": True})
        # Explicit cold-start extension: one genuine observation, not duplicated fake history.
        _, observed, noise, _ = cases[0]
        cold = prefix_inputs(observed["images"][-1:], observed["instructions"][-1:], [0],
                             visual, transform, tokenizer, prompt, device)
        sample(student, cold, noise)
        report[variant] = {
            "step": args.step, "state_entries": len(student.state_dict()), "fixtures": checks,
            "cold_start_single_observation_finite": True,
            "cold_start_scope": "Interface smoke only; B training used two observed nodes.",
        }
        print("B_EXPORT_REPLAY_PASS", variant, checks, flush=True)
        gc.collect()
    snapshot(args.output, [meta["recipe"] for meta in metas])
    write(args.output / "VERIFICATION.json", {
        "complete": True, "utc": datetime.now(timezone.utc).isoformat(),
        "platform": "local PPU; not NVIDIA and not SIMPLER",
        "torch": torch.__version__, "arms": report,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "teacher_loaded": False, "optimizer_loaded": False, "base_weight_checkpoint_loaded": False,
        "nvidia_verified": False, "simpler_verified": False,
    })
    write(args.output / "WORK_STATUS.json", {"complete": True, "phase": "source/export/replay verified"})
    checksums = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha(path)}  {path.relative_to(args.output)}")
    (args.output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print("RELEASE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
