"""Pure evaluation contracts; no torch/TF imports or model/data side effects."""

import hashlib
import numpy as np


FUTURE_OFFSETS = (3, 7, 11, 15)  # Actual WoG training: four DINO frames, current enters VAE.
VARIANTS = ("past", "past_static", "future_oracle")


def interior_centres(length, past_offsets, per_episode=4, horizon=16):
    """Only complete, unpadded windows; sample a bounded number per episode."""
    if per_episode < 1:
        raise ValueError("per_episode must be positive")
    lo = -min(past_offsets)
    hi = length - 1 - max(horizon - 1, max(FUTURE_OFFSETS))
    if hi < lo:
        return []
    return np.linspace(lo, hi, min(per_episode, hi - lo + 1), dtype=int).tolist()


def condition_frames(frames, offsets, variant, past_offsets):
    """Return current and exactly the permitted conditioning frames.

    Frames are PIL images or arbitrary test objects. Future images can be in the
    common cache but are never passed to the Past policy.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    index = {int(o): i for i, o in enumerate(offsets)}
    if len(index) != len(offsets) or 0 not in index:
        raise ValueError("offsets must be unique and contain the current frame")
    current = frames[index[0]]
    if variant == "future_oracle":
        return current, {"cond_frames": [frames[index[o]] for o in FUTURE_OFFSETS]}
    history = [frames[index[o]] for o in past_offsets if o < 0]
    if variant == "past_static":
        history = [current] * len(history)
    return current, {"past_frames": history}


def sample_seed(dataset_name, episode, timestep, base_seed):
    """Paired diffusion noise, invariant to rank count and evaluation order."""
    text = f"{dataset_name}:{int(episode)}:{int(timestep)}:{int(base_seed)}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little") % (2**31)


def normalize_actions(raw, stats):
    """Same BOUNDS_Q99 transform as training, AFTER OXE standardization."""
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", [True] * 6 + [False]), dtype=bool)
    out = np.where(mask, np.clip(2 * (raw - low) / (high - low + 1e-8) - 1, -1, 1), raw)
    unused = np.asarray(stats["min"]) == np.asarray(stats["max"])
    return np.where(unused, 0, out).astype(np.float32)


def action_metrics(predicted, target, stats):
    """Raw standardized r/R² plus normalized errors; no NaN JSON values.

    Shapes [N,16,7]. These are offline expert-action errors, not task success.
    """
    p, g = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != g.shape or p.ndim != 3 or p.shape[1:] != (16, 7) or len(p) == 0:
        raise ValueError("expected matching nonempty [N,16,7] predictions and targets")
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError("predictions/targets contain NaN or Inf")
    rs = []
    for j in range(6):
        x, y = p[:, :, j].ravel(), g[:, :, j].ravel()
        rs.append(float(np.corrcoef(x, y)[0, 1]) if min(x.std(), y.std()) > 1e-10 else None)
    valid_r = [r for r in rs if r is not None]
    residual = float(np.square(p[:, :, :6] - g[:, :, :6]).sum())
    variance = float(np.square(g[:, :, :6] - g[:, :, :6].mean((0, 1))).sum())
    pn, gn = normalize_actions(p, stats), normalize_actions(g, stats)
    return {
        "n": len(p),
        "r_6": float(np.mean(valid_r)) if valid_r else None,
        "r_per_dim": rs,
        "r2_6": 1 - residual / variance if variance > 1e-12 else None,
        "rmse_per_dim": np.sqrt(np.square(p - g).mean((0, 1))).tolist(),
        "normalized_mse_6": float(np.square(pn[:, :, :6] - gn[:, :, :6]).mean()),
        "normalized_mae_6": float(np.abs(pn[:, :, :6] - gn[:, :, :6]).mean()),
        "normalized_first_action_mse_6": float(np.square(pn[:, 0, :6] - gn[:, 0, :6]).mean()),
        "gripper_accuracy": float(((p[:, :, 6] >= 0.5) == (g[:, :, 6] >= 0.5)).mean()),
    }


def ordered_predictions(shards, expected_n):
    """Reassemble variable-sized distributed shards, including empty ranks."""
    indices = np.concatenate([s["indices"] for s in shards])
    if sorted(indices.tolist()) != list(range(expected_n)):
        raise ValueError("distributed predictions have duplicate or missing samples")
    values = np.concatenate([s["predictions"] for s in shards], axis=0)
    return values[np.argsort(indices)]
