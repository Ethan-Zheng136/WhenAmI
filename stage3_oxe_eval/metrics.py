"""Masked, episode-equal metrics; no pooling of incompatible robot raw units."""

import numpy as np


def normalize(x, stats, clip=True):
    x = np.asarray(x, np.float64)
    mask = np.asarray(stats.get("mask", [True] * 6 + [False]), bool)
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    value = 2 * (x - lo) / (hi - lo + 1e-8) - 1
    if clip:
        value = np.clip(value, -1, 1)
    out = np.where(mask, value, x)
    unused = np.asarray(stats["min"]) == np.asarray(stats["max"])
    return np.where(unused, 0., out)


def denormalize(x, stats, clip=True):
    x = np.asarray(x, np.float64)
    if clip:
        x = np.clip(x, -1, 1)
    mask = np.asarray(stats.get("mask", [True] * 6 + [False]), bool)
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    return np.where(mask, .5 * (x + 1) * (hi - lo) + lo, x)


def episode_metrics(prediction, target, valid, model_stats, eval_stats):
    """prediction[windows,seeds,16,7]; target[windows,16,7]; equal windows."""
    p, y, m = np.asarray(prediction, np.float64), np.asarray(target, np.float64), np.asarray(valid, bool)
    if p.ndim != 4 or p.shape[2:] != (16, 7) or y.shape != p.shape[:1] + p.shape[2:]:
        raise ValueError("incorrect prediction/target shape")
    if m.shape != y.shape[:-1] or not m.any(-1).all() or not m[:, 0].all():
        raise ValueError("invalid action-window mask")
    if not np.isfinite(p).all() or not np.isfinite(y[m]).all():
        raise ValueError("nonfinite valid prediction/target")
    raw = denormalize(p, model_stats, clip=True)
    raw_unclipped = denormalize(p, model_stats, clip=False)
    gn = normalize(y, eval_stats)
    pn = normalize(raw, eval_stats)
    pu = normalize(raw_unclipped, eval_stats, clip=False)
    error = pn[..., :6] - gn[:, None, ..., :6]
    unclip_error = pu[..., :6] - gn[:, None, ..., :6]
    weight = m[:, None]
    counts = weight.sum(-1)
    def masked_average(value):
        return np.where(weight, value, 0.).sum(-1) / counts
    per_window_seed = {
        "mse6": masked_average(np.square(error).mean(-1)),
        "unclipped_mse6": masked_average(np.square(unclip_error).mean(-1)),
        "mae6": masked_average(np.abs(error).mean(-1)),
        "first_action_mse6": np.square(error[..., 0, :]).mean(-1),
        "gripper_accuracy": masked_average((raw[..., 6] >= .5) == (y[:, None, ..., 6] >= .5)),
    }
    active = np.asarray(eval_stats["min"])[:6] != np.asarray(eval_stats["max"])[:6]
    if active.any():
        per_window_seed["active_dim_mse"] = masked_average(np.square(error[..., active]).mean(-1))
    # Raw moments are only summed WITHIN a domain. Each episode has total weight1.
    # Binarize gripper for raw RMSE as in original deployed postprocessing.
    deployed = raw.copy()
    deployed[..., 6] = (deployed[..., 6] >= .5).astype(float)
    truth = np.broadcast_to(y[:, None], p.shape)
    safe_truth = np.where(weight[..., None], truth, 0.)
    w = weight[..., None] / counts[..., None, None] / (p.shape[0] * p.shape[1])
    moments = {
        "p": (deployed * w).sum((0, 1, 2)),
        "y": (safe_truth * w).sum((0, 1, 2)),
        "p2": (np.square(deployed) * w).sum((0, 1, 2)),
        "y2": (np.square(safe_truth) * w).sum((0, 1, 2)),
        "py": (deployed * safe_truth * w).sum((0, 1, 2)),
        "se": (np.square(deployed - safe_truth) * w).sum((0, 1, 2)),
    }
    return {
        "mean": {k: float(v.mean()) for k, v in per_window_seed.items()},
        "per_window_seed": per_window_seed, "moments": moments,
        "raw_prediction": raw, "raw_unclipped_prediction": raw_unclipped,
        "active_continuous_dimensions": int(active.sum()),
    }


def summarize_raw(moments):
    av = {k: np.mean([m[k] for m in moments], axis=0) for k in moments[0]}
    vx = np.maximum(av["p2"] - av["p"]**2, 0.)
    vy = np.maximum(av["y2"] - av["y"]**2, 0.)
    cov = av["py"] - av["p"] * av["y"]
    r = [float(np.clip(cov[j] / np.sqrt(vx[j] * vy[j]), -1, 1))
         if min(vx[j], vy[j]) > 1e-16 else None for j in range(6)]
    finite = [v for v in r if v is not None]
    return {
        "r6": float(np.mean(finite)) if finite else None,
        "r_per_dim": r,
        "r2_6": float(1 - av["se"][:6].sum() / vy[:6].sum()) if vy[:6].sum() > 1e-16 else None,
        "rmse_per_dim": np.sqrt(av["se"]).tolist(),
    }
