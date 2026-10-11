"""Distributed raw-image development action evaluation for the joint student.

No cached VLM/Past features are used. All ranks run identical counts of root
FSDP forward calls (including padding for load balancing), avoiding collective
mismatches when dataset sizes are not divisible by the number of ranks.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from stage3.common import BASE, sha256, write_json
from stage3_scale.model import tokenize_instruction
from stage3.evaluate import denormalize

sys.path.insert(0, str(BASE / "evaluation"))
from past_protocol import sample_seed, action_metrics


@torch.no_grad()
def evaluate(model, visual, raw_root, output, transform, tokenizer, prompt_builder,
             statistics, device, limit_per_dataset=0, noise_seed=42):
    root = Path(raw_root)
    if (root / "SEALED_DO_NOT_EVALUATE").exists():
        raise ValueError("sealed test prohibited")
    manifest = json.loads((root / "samples_manifest.json").read_text())
    if manifest.get("reader_study", {}).get("role") != "train_dev":
        raise ValueError("requires explicitly permitted development raw data")
    dataset = np.load(root / "dataset.npy")
    episodes = np.load(root / "episode.npy")
    times = np.load(root / "timestep.npy")
    partition = np.load(root / "partition.npy")
    indices = np.load(root / "valid_indices.npy")
    if not all(partition[i] == "dev" for i in indices):
        raise ValueError("development index contains non-dev observations")
    frames = np.load(root / "frames.npy", mmap_mode="r")
    raw = np.load(root / "raw_actions.npy", mmap_mode="r")
    language = json.loads((root / "instructions.json").read_text())
    offsets = list(manifest["frame_offsets"])
    current_index = offsets.index(0)
    past_indices = [offsets.index(o) for o in (-15, -11, -7, -3)]
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if dist.is_initialized():
        dist.barrier()
    model.eval()
    visual.eval()
    selected = []
    for name in sorted(set(dataset.tolist())):
        ids = indices[dataset[indices] == name]
        selected.extend((ids[:limit_per_dataset] if limit_per_dataset else ids).tolist())
    if not selected:
        raise ValueError("empty development evaluation")
    rows = []
    # ceil(n/world) calls on every rank; padded cases computed then discarded.
    for local in range((len(selected) + world - 1) // world):
        ordinal = local * world + rank
        i = selected[min(ordinal, len(selected) - 1)]
        now = transform(Image.fromarray(frames[i, current_index]))
        history = [transform(Image.fromarray(frames[i, j])) for j in past_indices]
        now = {k: v[None].to(device) for k, v in now.items()}
        hist = {k: torch.stack([p[k] for p in history])[None].to(device) for k in now}
        tokens = tokenize_instruction(language[i], tokenizer, prompt_builder)[None].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = visual(now, hist)
            condition = model(
                input_ids=tokens, attention_mask=torch.ones_like(tokens, dtype=torch.bool),
                current=now, history_features=features["history_features"],
                history_vae=features["history_vae"], condition_only=True,
            )
            seed = sample_seed(dataset[i], episodes[i], times[i], noise_seed)
            # Match original action sampler's FP32 noise, use BF16 computation
            # because the joint policy/teacher FSDP config is BF16.
            generator = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn((1, 16, 7), generator=generator, device=device)
            ts = torch.linspace(0, 1, 10, device=device)
            for j in range(1, 10):
                v = model(noisy_actions=x, tau=ts[j - 1].expand(1), conditioning=condition)["velocity"]
                x = x + v.float() * (ts[j] - ts[j - 1])
        if ordinal < len(selected):
            rows.append((i, x[0].cpu().numpy()))
    np.savez_compressed(output / f"rank-{rank:05d}.npz",
                        indices=np.asarray([i for i, _ in rows], dtype=np.int64),
                        predictions=np.stack([p for _, p in rows]) if rows else np.empty((0, 16, 7), np.float32))
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        pieces = []
        for r in range(world):
            with np.load(output / f"rank-{r:05d}.npz") as a:
                pieces.extend(zip(a["indices"].tolist(), a["predictions"]))
        if sorted(i for i, _ in pieces) != sorted(selected):
            raise ValueError("missing/duplicate development samples")
        ordered = dict(pieces)
        results = []
        for name in sorted(set(dataset.tolist())):
            ids = np.asarray([i for i in selected if dataset[i] == name])
            p = denormalize(np.stack([ordered[i] for i in ids]), statistics[name]["action"])
            target = np.asarray(raw[ids])
            metrics = action_metrics(p, target, statistics[name]["action"])
            np.savez_compressed(output / f"{name}_online.npz",
                                predictions=p, target=target, indices=ids,
                                episode=episodes[ids], timestep=times[ids])
            results.append({"dataset": name, **metrics})
        write_json(output / "results.json", {
            "complete": True, "split": "development", "raw_inputs": True,
            "causal": True, "future_images_used": False, "noise_seed": noise_seed,
            "precision": "BF16 model compute, FP32 sampling state",
            "parent_samples_manifest_sha256": sha256(root / "samples_manifest.json"),
            "limit_per_dataset": limit_per_dataset, "results": results,
            "note": "Not numerically protocol-identical to FP32 cached pilot; compare matched joint runs.",
        })
    if dist.is_initialized():
        dist.barrier()
