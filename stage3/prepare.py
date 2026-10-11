"""Build NEW RLDS training samples, then cache frozen endpoint features.

Never train on evaluation/past_s2_20k_eval_01. Stage-3 validation is separated
by episode from Stage-3 training, though endpoint pretraining may have seen it.
Encode mode can run under torchrun; ranks write disjoint rows of shared .npy
files. manifest.json is published only after all ranks finish successfully.
"""

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from stage3.common import (
    BASE, STORE, DEFAULT_PAST, DEFAULT_FUTURE, reference_imports,
    endpoint_record, check_endpoint, source_fingerprint, check_sources, sha256, write_json,
)
from stage3.cache import FIELDS, episode_split, assert_episode_disjoint

sys.path.insert(0, str(BASE / "evaluation"))
from past_protocol import FUTURE_OFFSETS, interior_centres, condition_frames, normalize_actions


def build_samples(args):
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
        os.environ.pop(k, None)
    os.environ.update(WAI_BOUNDED_INPUT="1", TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="2")
    reference_imports()
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from vla.past import checkpoint_spec, PastSpec
    from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
    from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
    from prismatic.vla.datasets.rlds.obs_transforms import decode_and_resize
    from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

    if args.limit < 2 or args.per_episode < 1 or len(set(args.datasets)) != len(args.datasets):
        raise ValueError("need positive sampling settings and unique dataset names")
    args.output.mkdir(parents=True, exist_ok=False)
    endpoints = {side: endpoint_record(path) for side, path in [("past", args.past), ("future", args.future)]}
    cp = torch.load(args.past, map_location="cpu", mmap=True, weights_only=True)
    stage, offsets = checkpoint_spec(cp)
    if stage != 2:
        raise ValueError("Past checkpoint is not Stage 2")
    past_spec = PastSpec.from_offsets(offsets)
    cf = torch.load(args.future, map_location="cpu", mmap=True, weights_only=True)
    if "qformer" not in cf["model"] or "past_qformer" in cf["model"]:
        raise ValueError("Future checkpoint must be the trained WoG Stage-I QFormer")
    del cp, cf
    offsets_union = sorted(set(offsets) | set(FUTURE_OFFSETS))
    frames = np.lib.format.open_memmap(
        args.output / "frames.npy", mode="w+", dtype=np.uint8,
        shape=(args.limit * len(args.datasets), len(offsets_union), 224, 224, 3),
    )
    raw_actions, dataset_ids, episode_ids, times, language = [], [], [], [], []
    details = {}
    for name in args.datasets:
        statistics = endpoints["past"]["statistics"][name]
        identity = deepcopy(statistics)
        identity["action"]["mean"], identity["action"]["std"] = [0.] * 7, [1.] * 7
        kwargs, _ = get_oxe_dataset_kwargs_and_weights(
            STORE / "datasets/open-x-embodiment", [(name, 1.)],
            load_camera_views=("primary",), load_depth=False, load_proprio=False,
            load_language=True, action_proprio_normalization_type=NormalizationType.NORMAL,
        )
        if len(kwargs) != 1:
            raise ValueError(f"unsupported dataset {name}")
        kwargs = kwargs[0]
        kwargs.update(dataset_statistics=identity, num_parallel_reads=1, num_parallel_calls=1, shuffle=False)
        builder = tfds.builder(name, data_dir=str(STORE / "datasets/open-x-embodiment"))
        split = "train" if "val" in builder.info.splits else "train[:95%]"
        ds, _ = make_dataset_from_rlds(**kwargs, train=True, load_all_data_for_training=False)
        options = tf.data.Options()
        options.deterministic = True
        ds = ds.with_options(options)
        taken, scanned, skipped = 0, 0, 0
        for episode, traj in enumerate(ds.as_numpy_iterator()):
            scanned += 1
            actions = np.asarray(traj["action"], np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError("expected standardized raw 7D actions")
            decoded = {}
            for t in interior_centres(len(actions), offsets, args.per_episode):
                text = traj["task"]["language_instruction"][t].decode().strip()
                target = actions[t:t + 16]
                indices = [t + o for o in offsets_union]
                if not text or not np.isfinite(target).all() or any(
                    isinstance(traj["observation"]["image_primary"][i], bytes)
                    and not traj["observation"]["image_primary"][i] for i in indices
                ):
                    skipped += 1
                    continue
                for i in indices:
                    if i not in decoded:
                        obs = decode_and_resize(
                            {"image_primary": tf.convert_to_tensor(traj["observation"]["image_primary"][i])},
                            resize_size=(224, 224), depth_resize_size={},
                        )
                        decoded[i] = obs["image_primary"].numpy()
                frames[len(raw_actions)] = np.stack([decoded[i] for i in indices])
                raw_actions.append(target)
                dataset_ids.append(name)
                episode_ids.append(episode)
                times.append(t)
                language.append(text)
                taken += 1
                if taken >= args.limit:
                    break
            if taken >= args.limit or scanned >= args.max_episodes:
                break
        if taken == 0:
            raise ValueError(f"no valid windows for {name}")
        details[name] = {"split": split, "n": taken, "episodes_scanned": scanned, "skipped": skipped}
        del ds
        print(f"SAMPLES {name}: {taken} windows, split={split}", flush=True)
    frames.flush()
    del frames
    ids, episodes, steps = np.asarray(dataset_ids), np.asarray(episode_ids), np.asarray(times)
    train, valid = episode_split(ids, episodes, args.seed)
    assert_episode_disjoint(ids, episodes, train, valid)
    np.save(args.output / "raw_actions.npy", np.stack(raw_actions), allow_pickle=False)
    for name, arr in [
        ("dataset", ids), ("episode", episodes), ("timestep", steps),
        ("train_indices", train), ("valid_indices", valid),
    ]:
        np.save(args.output / f"{name}.npy", arr, allow_pickle=False)
    write_json(args.output / "instructions.json", language)
    for side in ("past", "future"):
        # Each fixed action reader must get its own checkpoint normalization.
        labels = np.stack([
            normalize_actions(action, endpoints[side]["statistics"][name]["action"])
            for action, name in zip(raw_actions, ids)
        ])
        np.save(args.output / f"{side}_actions.npy", labels, allow_pickle=False)
    manifest = {
        "protocol": "whenami_stage3_frozen_endpoints_v1", "complete": False,
        "n": len(ids), "split_seed": args.seed, "train_n": len(train), "valid_n": len(valid),
        "endpoints": endpoints, "past_metadata": past_spec.metadata(),
        "frame_offsets": offsets_union, "datasets": details,
        "reference_source_sha256": source_fingerprint(),
        "notes": [
            "Fresh training-split RLDS; never uses the previous Past evaluation cache.",
            "Connector train/validation split by whole episode, not adjacent windows.",
            "Frozen endpoints may have seen these episodes during their own pretraining.",
            "Small fixed subset and no augmentation: this is a Stage-3 pilot, not full OXE training.",
            "R receives source H and source current-image/language z; no actions or opposite-side H.",
            "Backward means Future evidence -> Past condition, NOT reverse physical actions.",
        ],
    }
    write_json(args.output / "samples_manifest.json", manifest)
    print(f"SAMPLES COMPLETE: train={len(train)}, valid={len(valid)}", flush=True)


def extract_one_endpoint(model, images, language, side, offsets, past_offsets):
    """Tap actual policy preprocessing/condition/cognition; skip only action sampling.

    Hooks live only on this in-memory instance, and are restored in finally.
    No label/action tensor is passed to the encoder or the VLM.
    """
    current, kwargs = condition_frames(images, offsets, "past" if side == "past" else "future_oracle", past_offsets)
    head = model.past_qformer if side == "past" else model.qformer
    captured = {}

    def before_projection(module, inputs):
        captured["h"] = inputs[0].detach().float().cpu().numpy()[0].copy()

    def capture_instead_of_sample(z, num_steps=10, per_token=None):
        if per_token is None:
            raise RuntimeError("condition route was not used")
        captured["z"] = z.detach().float().cpu().numpy()[0].copy()
        captured["c"] = per_token.detach().float().cpu().numpy()[0].copy()
        return torch.zeros(1, 16, 7, device=z.device, dtype=z.dtype)

    handle = head.v_proj.register_forward_pre_hook(before_projection)
    original_inference = model.action_model.inference
    model.action_model.inference = capture_instead_of_sample
    try:
        with torch.inference_mode():
            model.predict_action(image=current, instruction=language, unnorm_key=model._stage3_dataset, **kwargs)
        if set(captured) != {"h", "z", "c"}:
            raise RuntimeError("failed to capture all endpoint tensors")
        for key, value in captured.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{side}/{key} contains NaN or Inf")
        return captured
    finally:
        handle.remove()
        model.action_model.inference = original_inference


def encode(args):
    import torch.distributed as dist
    from PIL import Image
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if world > 1:
        dist.init_process_group("nccl")
    reference_imports()
    from vla import load_vla
    root = args.cache
    manifest = json.loads((root / "samples_manifest.json").read_text())
    if (root / "manifest.json").exists():
        raise FileExistsError("endpoint cache already finalized")
    check_sources(manifest["reference_source_sha256"])
    frames = np.load(root / "frames.npy", mmap_mode="r")
    language = json.loads((root / "instructions.json").read_text())
    dataset_ids = np.load(root / "dataset.npy", allow_pickle=False)
    n = manifest["n"]
    try:
        if rank == 0:
            for field, shape in FIELDS.items():
                if field.endswith("actions"):
                    continue
                path = root / f"{field}.npy"
                if path.exists():
                    raise FileExistsError(f"partial cache exists; use a fresh directory: {path}")
                arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, *shape))
                arr[:] = np.nan  # Detect unfilled rank rows before committing.
                arr.flush()
        if world > 1:
            dist.barrier()
        for side in ("past", "future"):
            record = manifest["endpoints"][side]
            check_endpoint(record)
            model = load_vla(
                record["path"], train_venc=True, action_model_type="DiT-L", load_for_training=False,
            ).float().cuda().eval().requires_grad_(False)
            if (model.whenami_stage == 2) != (side == "past"):
                raise ValueError("endpoint type mismatch")
            outputs = {
                key: np.load(root / f"{side}_{key}.npy", mmap_mode="r+") for key in ("h", "z", "c")
            }
            for j, i in enumerate(range(rank, n, world)):
                model._stage3_dataset = str(dataset_ids[i])
                values = extract_one_endpoint(
                    model, [Image.fromarray(im) for im in frames[i]], language[i], side,
                    manifest["frame_offsets"], manifest["past_metadata"]["obs_offsets"],
                )
                for key in outputs:
                    outputs[key][i] = values[key]
                if rank == 0 and (j + 1) % 10 == 0:
                    print(f"ENCODE {side}: rank0 {j + 1} samples", flush=True)
            for arr in outputs.values():
                arr.flush()
            del arr, outputs, model
            gc.collect()
            torch.cuda.empty_cache()
            if world > 1:
                dist.barrier()
        if rank == 0:
            for field in FIELDS:
                arr = np.load(root / f"{field}.npy", mmap_mode="r")
                if not np.isfinite(arr).all():
                    raise ValueError(f"unfilled/nonfinite cached tensor {field}")
            manifest["complete"] = True
            manifest["encoding_world_size"] = world
            manifest["finished_unix"] = time.time()
            manifest["array_sha256"] = {
                p.name: sha256(p) for p in root.glob("*.npy") if p.name != "frames.npy"
            }
            write_json(root / "manifest.json", manifest)
            print("ENDPOINT CACHE COMPLETE:", root / "manifest.json", flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("samples")
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--past", type=Path, default=DEFAULT_PAST)
    sample.add_argument("--future", type=Path, default=DEFAULT_FUTURE)
    sample.add_argument("--datasets", nargs="+", default=["fractal20220817_data", "bridge_orig"])
    sample.add_argument("--limit", type=int, default=512)
    sample.add_argument("--per-episode", type=int, default=4)
    sample.add_argument("--max-episodes", type=int, default=2000)
    sample.add_argument("--seed", type=int, default=42)
    enc = sub.add_parser("encode")
    enc.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    (build_samples if args.command == "samples" else encode)(args)


if __name__ == "__main__":
    main()
