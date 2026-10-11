"""Past Stage-2 evaluation, using a newly prepared raw-RLDS sample set.

Subcommands:
  prepare: one CPU process; deterministic standardized RLDS -> common .npy cache.
  evaluate: torchrun, one model per GPU; real past, static-history intervention,
            and optional WoG Future oracle, all on identical samples/noise seeds.

Training scripts, existing checkpoints and old comparison results are not edited.
"""

import argparse
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from past_protocol import (
    FUTURE_OFFSETS, VARIANTS, action_metrics, condition_frames,
    interior_centres, ordered_predictions, sample_seed,
)

REPO = Path(__file__).resolve().parents[1] / "WoG_16rank_probe"
sys.path.insert(0, str(REPO))
DATA = Path("/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment")


def write_json(path, payload):
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_record(path):
    path = Path(path).resolve()
    st = path.stat()
    config = path.parents[1] / "config.json"
    return {
        "path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
        "config": json.loads(config.read_text()), "config_sha256": sha(config),
        "statistics_sha256": sha(path.parents[1] / "dataset_statistics.json"),
    }


def prepare(args):
    # Platform WORLD_SIZE denotes node count BEFORE torchrun. No process group here.
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
        os.environ.pop(k, None)
    os.environ.update(
        WAI_BOUNDED_INPUT="1", TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="2", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    )
    import torch  # Correct import ordering for the PPU image; no model/GPU tensors.
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from vla.past import checkpoint_spec, PastSpec
    from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
    from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
    from prismatic.vla.datasets.rlds.obs_transforms import decode_and_resize
    from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

    if args.limit < 1 or args.per_episode < 1 or args.max_episodes < 1:
        raise ValueError("sampling limits must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    ck = torch.load(args.past, map_location="cpu", mmap=True, weights_only=True)
    stage, offsets = checkpoint_spec(ck)
    if stage != 2:
        raise ValueError("--past must be a trained Past Stage-2 checkpoint")
    spec = PastSpec.from_offsets(offsets)
    del ck
    oracle = torch.load(args.future, map_location="cpu", mmap=True, weights_only=True)
    if "qformer" not in oracle["model"] or "past_qformer" in oracle["model"]:
        raise ValueError("--future must be WoG Stage I, not a Past checkpoint")
    if not any("per_attn" in k for k in oracle["model"]["action_model"]):
        raise ValueError("Future checkpoint has no per-attention (possibly WoG Stage II)")
    del oracle
    stats_file = args.past.parents[1] / "dataset_statistics.json"
    statistics = json.loads(stats_file.read_text())
    union = sorted(set(offsets) | set(FUTURE_OFFSETS))
    manifest = {
        "protocol": "whenami_past_raw_rlds_v1", "created_unix": time.time(),
        "past": model_record(args.past), "future": model_record(args.future),
        "past_metadata": spec.metadata(), "frame_offsets": union,
        "future_dino_offsets": list(FUTURE_OFFSETS), "target_action_offsets": list(range(16)),
        "limit_per_dataset": args.limit, "max_samples_per_episode": args.per_episode,
        "datasets": {},
        "notes": [
            "Past never receives real future observations or target actions.",
            "Future is privileged oracle reference, not equal-information comparison or guaranteed upper bound.",
            "Static history is an input intervention, NOT a separately trained current-only baseline.",
            "Only complete interior windows; short episodes and episode boundaries are excluded.",
            "Fractal train[95%:] is NOT held out from models trained on the full train split.",
            "Past warm-started from Future and had additional training; training budgets differ.",
            "Metrics are offline action errors, not closed-loop success rates.",
            "This protocol differs from the old cached compare; old scores are not directly comparable.",
        ],
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p) for p in [
                REPO / "vla/vla.py", REPO / "vla/qformer.py",
                REPO / "action_model/action_model.py",
                REPO / "prismatic/vla/datasets/rlds/oxe/transforms.py",
                REPO / "prismatic/vla/datasets/rlds/obs_transforms.py",
            ]
        },
        "eval_source_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("past_protocol.py")]},
    }
    for name in args.datasets:
        dataset_dir = args.output / name
        dataset_dir.mkdir()
        kw, _ = get_oxe_dataset_kwargs_and_weights(
            DATA, [(name, 1.0)], load_camera_views=("primary",),
            load_depth=False, load_proprio=False, load_language=True,
            action_proprio_normalization_type=NormalizationType.NORMAL,
        )
        if len(kw) != 1:
            raise ValueError(f"dataset {name} not supported")
        kw = kw[0]
        # Preserve standardized RAW expert actions. With NORMAL, mean=0/std=1
        # is identity, unlike BOUNDS_Q99 which irreversibly clips outliers.
        identity_stats = deepcopy(statistics[name])
        identity_stats["action"]["mean"] = [0.] * 7
        identity_stats["action"]["std"] = [1.] * 7
        kw.update(dataset_statistics=identity_stats, num_parallel_reads=1, num_parallel_calls=1, shuffle=False)
        builder = tfds.builder(name, data_dir=str(DATA))
        split = "val" if "val" in builder.info.splits else "train[95%:]"
        ds, _ = make_dataset_from_rlds(**kw, train=False)
        deterministic = tf.data.Options()
        deterministic.deterministic = True
        ds = ds.with_options(deterministic)
        images = np.lib.format.open_memmap(
            dataset_dir / "frames.npy", mode="w+", dtype=np.uint8,
            shape=(args.limit, len(union), 224, 224, 3),
        )
        actions, episode_ids, timesteps, instructions = [], [], [], []
        scanned = skipped = 0
        for episode_idx, traj in enumerate(ds.as_numpy_iterator()):
            scanned += 1
            raw_actions = np.asarray(traj["action"], dtype=np.float32)
            if raw_actions.ndim != 2 or raw_actions.shape[1] != 7:
                raise ValueError("expected standardized 7D expert actions")
            centres = interior_centres(len(raw_actions), offsets, args.per_episode)
            decoded = {}
            for t in centres:
                text = traj["task"]["language_instruction"][t].decode().strip()
                target = raw_actions[t:t + 16]
                if not text or not np.isfinite(target).all():
                    skipped += 1
                    continue
                indices = [t + o for o in union]
                assert min(indices) >= 0 and max(indices) < len(raw_actions)
                if any(
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
                n = len(actions)
                images[n] = np.stack([decoded[i] for i in indices])
                actions.append(target)
                episode_ids.append(episode_idx)
                timesteps.append(t)
                instructions.append(text)
                if len(actions) == args.limit:
                    break
            if len(actions) == args.limit or scanned >= args.max_episodes:
                break
        images.flush()
        del images, ds
        if not actions:
            raise ValueError(f"No complete labelled windows found for {name}")
        n = len(actions)
        np.save(dataset_dir / "actions.npy", np.stack(actions), allow_pickle=False)
        np.save(dataset_dir / "episode.npy", np.asarray(episode_ids), allow_pickle=False)
        np.save(dataset_dir / "timestep.npy", np.asarray(timesteps), allow_pickle=False)
        write_json(dataset_dir / "instructions.json", instructions)
        manifest["datasets"][name] = {
            "n": n, "episodes": len(set(episode_ids)), "episodes_scanned": scanned,
            "skipped_unlabelled_or_invalid_windows": skipped, "split": split,
            "statistics": statistics[name],
            "standardization_function": kw["standardize_fn"].__name__,
            "standardization_sha256": hashlib.sha256(inspect.getsource(kw["standardize_fn"]).encode()).hexdigest(),
            "frames_shape": [args.limit, len(union), 224, 224, 3],
            "sample_files_sha256": {
                p.name: sha(p) for p in [
                    dataset_dir / "actions.npy", dataset_dir / "episode.npy",
                    dataset_dir / "timestep.npy", dataset_dir / "instructions.json",
                ]
            },
        }
        print(f"PREPARED {name}: {n} windows / {len(set(episode_ids))} episodes, split={split}", flush=True)
    write_json(args.output / "manifest.json", manifest)
    print("PREPARE PASS:", args.output / "manifest.json", flush=True)


def evaluate(args):
    import torch
    import torch.distributed as dist
    from PIL import Image

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    if world > 1:
        dist.init_process_group("nccl")
    from vla import load_vla

    manifest = json.loads((args.samples / "manifest.json").read_text())
    if manifest["protocol"] != "whenami_past_raw_rlds_v1":
        raise ValueError("unsupported sample protocol")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("noise seeds must be distinct")
    if args.output.exists():
        raise FileExistsError(args.output)
    for kind in ("past", "future"):
        record = manifest[kind]
        stat = Path(record["path"]).stat()
        if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise ValueError(f"{kind} checkpoint changed after sample preparation")
    for relative, expected in manifest["source_sha256"].items():
        if sha(REPO / relative) != expected:
            raise ValueError(f"Model/data source changed after preparation: {relative}")
    past_offsets = manifest["past_metadata"]["obs_offsets"]
    offsets = manifest["frame_offsets"]
    payload = {
        "protocol": manifest["protocol"], "sample_manifest": str(args.samples / "manifest.json"),
        "past_checkpoint": manifest["past"]["path"], "future_checkpoint": manifest["future"]["path"],
        "noise_seeds": args.seeds, "variants": args.variants, "results": [], "notes": manifest["notes"],
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # One load per checkpoint, not one load per dataset or intervention.
        for kind in ("past", "future"):
            variants = [v for v in args.variants if (v == "future_oracle") == (kind == "future")]
            if not variants:
                continue
            model = load_vla(
                manifest[kind]["path"], train_venc=True, action_model_type="DiT-L",
                load_for_training=False,
            ).float().cuda().eval()
            if kind == "past" and (model.whenami_stage != 2 or model.obs_offsets != past_offsets):
                raise ValueError("Past architecture/sampling does not match prepared samples")
            if kind == "future" and (model.whenami_stage == 2 or model.use_temporal):
                raise ValueError("Future oracle must use the WoG QFormer path")
            for name, meta in manifest["datasets"].items():
                data = args.samples / name
                frames = np.load(data / "frames.npy", mmap_mode="r", allow_pickle=False)
                gt = np.load(data / "actions.npy", allow_pickle=False)
                episode = np.load(data / "episode.npy", allow_pickle=False)
                timestep = np.load(data / "timestep.npy", allow_pickle=False)
                language = json.loads((data / "instructions.json").read_text())
                n = meta["n"]
                indices = np.arange(rank, n, world, dtype=np.int64)
                for variant in variants:
                    predictions = np.empty((len(indices), len(args.seeds), 16, 7), np.float64)
                    for local_i, i in enumerate(indices):
                        # Only the selected historical/future subset reaches the model.
                        pil = [Image.fromarray(im) for im in frames[i]]
                        current, kw = condition_frames(pil, offsets, variant, past_offsets)
                        for seed_i, seed in enumerate(args.seeds):
                            paired_seed = sample_seed(name, episode[i], timestep[i], seed)
                            torch.manual_seed(paired_seed)
                            torch.cuda.manual_seed_all(paired_seed)
                            with torch.inference_mode():
                                p, _ = model.predict_action(
                                    image=current, instruction=language[i], unnorm_key=name, **kw
                                )
                            if p.shape != (16, 7) or not np.isfinite(p).all():
                                raise ValueError(f"Invalid prediction: {name}/{i}/{variant}")
                            predictions[local_i, seed_i] = p
                        if rank == 0 and (local_i + 1) % 10 == 0:
                            print(f"{name} {variant}: rank0 {local_i + 1}/{len(indices)}", flush=True)
                    local = {"indices": indices, "predictions": predictions}
                    shards = [local]
                    if world > 1:
                        shards = [None] * world
                        dist.all_gather_object(shards, local)
                    if rank == 0:
                        ordered = ordered_predictions(shards, n)
                        per_seed = [
                            action_metrics(ordered[:, j], gt, meta["statistics"]["action"])
                            for j in range(len(args.seeds))
                        ]
                        # Average scores across stochastic draws, not predictions:
                        # averaging predictions would evaluate an ensemble instead.
                        summary = {
                            key: float(np.mean([s[key] for s in per_seed if s[key] is not None]))
                            if any(s[key] is not None for s in per_seed) else None
                            for key in ["r_6", "r2_6", "normalized_mse_6", "normalized_mae_6",
                                        "normalized_first_action_mse_6", "gripper_accuracy"]
                        }
                        pred_file = args.output.parent / f"{args.output.stem}_{name}_{variant}_predictions.npz"
                        with pred_file.open("xb") as f:
                            np.savez_compressed(
                                f, predictions=ordered, target=gt, episode=episode, timestep=timestep,
                                seeds=np.asarray(args.seeds),
                            )
                        payload["results"].append({
                            "dataset": name, "variant": variant, "n": n, "episodes": meta["episodes"],
                            "split": meta["split"], "checkpoint": manifest[kind]["path"],
                            **summary, "per_seed": per_seed, "predictions_file": str(pred_file),
                        })
                        print(f"RESULT {name} {variant}: {json.dumps(summary)}", flush=True)
                        write_json(args.output.with_suffix(".partial.json"), payload)
                    if world > 1:
                        dist.barrier()
                del frames
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if rank == 0:
            payload["complete"] = True
            write_json(args.output, payload)
            print("EVALUATION COMPLETE:", args.output, flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--past", type=Path, required=True)
    p.add_argument("--future", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=["fractal20220817_data", "bridge_orig"])
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--per-episode", type=int, default=4)
    p.add_argument("--max-episodes", type=int, default=1000)
    p = sub.add_parser("evaluate")
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)


if __name__ == "__main__":
    main()
