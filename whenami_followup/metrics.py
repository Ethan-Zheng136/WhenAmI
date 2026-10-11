"""Numpy-only scoring and fixed-domain paired episode uncertainty."""
import numpy as np


def score(prediction, target, mask, statistics):
    p, t, m = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if p.ndim != 3 or p.shape[1:] != (16, 7) or t.shape != (16, 7) or m.shape != (16,):
        raise ValueError("expected seeds x 16 x 7 predictions and one current target")
    if not np.isfinite(p).all() or not m[0] or not m.any():
        raise ValueError("nonfinite action or invalid target mask")
    unused = np.asarray(statistics["min"]) == np.asarray(statistics["max"])
    norm = np.where(unused, 0, np.clip(p, -1, 1))
    err = np.square(norm[..., :6] - t[None, ..., :6]).mean(-1)
    unc = np.where(unused, 0, p)
    ue = np.square(unc[..., :6] - t[None, ..., :6]).mean(-1)
    return {
        "mse6": (err * m).sum(-1) / m.sum(),
        "unclipped_mse6": (ue * m).sum(-1) / m.sum(),
        "first_action_mse6": err[:, 0],
        "gripper_accuracy": (((p[..., 6] >= .5) == (t[None, ..., 6] >= .5)) * m).sum(-1) / m.sum(),
    }


def aggregate(rows):
    domains = {}
    for row in rows:
        domains.setdefault(row["dataset"], []).append(row)
    names = ("mse6", "unclipped_mse6", "first_action_mse6", "gripper_accuracy")
    domains = {d: {"episodes": len(rs), **{
        k: float(np.mean([r[k] for r in rs])) for k in names
    }} for d, rs in domains.items()}
    return {"domains": domains, "macro": {
        k: float(np.mean([r[k] for r in domains.values()])) for k in names
    }}


def paired(before, after, names, larger=False):
    names = np.asarray(names)
    a, b = np.asarray(before), np.asarray(after)
    rng = np.random.default_rng(20261003)
    domains = sorted(set(names))
    ba, bb = np.zeros(2000), np.zeros(2000)
    da, db = [], []
    for d in domains:
        ids = np.flatnonzero(names == d)
        draws = rng.integers(0, len(ids), size=(2000, len(ids)))
        ba += a[ids][draws].mean(1) / len(domains)
        bb += b[ids][draws].mean(1) / len(domains)
        da.append(a[ids].mean())
        db.append(b[ids].mean())
    pre, post = float(np.mean(da)), float(np.mean(db))
    gain = 100 * (bb - ba) if larger else 100 * (1 - bb / ba)
    return {"before": pre, "after": post,
            "improvement": 100 * (post - pre) if larger else 100 * (1 - post / pre),
            "unit": "percentage_points" if larger else "percent_relative_reduction",
            "paired_episode_95pct_interval": np.quantile(gain, [.025, .975]).tolist(),
            "domains_favorable": int(sum(y > x if larger else y < x for x, y in zip(da, db))),
            "domains": len(domains), "episodes": len(a)}
