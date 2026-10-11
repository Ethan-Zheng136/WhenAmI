"""[zgt] 条件里装了多少动作信息？—— 绕开 DiT 的线性探针。

评测 r/R² 测的是「条件 + DiT」这个整体。条件不好、或 DiT 不会用，
两种情况在那个指标上长得一样。本脚本只看条件本身：

    条件向量 --岭回归--> 真实动作，留出集 R²

结论指向两种完全不同的修法：
  * WoG 显著更高  -> 差距在「条件编码了什么」，该改信息路径（加宽 T）
  * 两者相当      -> 差距在「DiT 怎么用它」，加宽 T 是白做

不先分辨就改架构，等于再赌一次 21 小时。

用法（单卡即可）：
    python3 probe_cond_info.py --ckpt WoG=... --ckpt WhenAmI=... \
        --temporal-for WhenAmI --limit 256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

OUT = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/oxe")
OFFSETS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def build_extra(ckpt: str, temporal: bool) -> dict:
    """与 eval_conditioned_dist.py 同一套推断：结构从权重读，不从 config 读。"""
    if not temporal:
        return {}
    sd = torch.load(ckpt, map_location="cpu", mmap=True,
                    weights_only=False)["model"].get("converge", {})
    if any(k.startswith("flow.") for k in sd):
        ct = "flow"
    elif any(k.startswith("q_past.") for k in sd):
        ct = "masked"
    elif any("obs_encoder" in k for k in sd):
        ct = "recurrent"
    else:
        raise ValueError(f"cannot infer converge_type: {sorted(sd)[:5]}")

    extra = dict(use_temporal=True, obs_offsets=OFFSETS, converge_type=ct)
    if ct == "flow":
        enc = {k: v for k, v in sd.items() if k.startswith("encoder.")}
        extra.update(
            flow_enc_layers=len({k.split(".")[2] for k in enc
                                 if k.startswith("encoder.blocks.")}),
            flow_cross_layers=len({k.split(".")[2] for k in enc
                                   if k.startswith("encoder.cross_frame.")}),
            flow_enc_hidden=int(enc["encoder.query"].shape[-1]),
            flow_enc_use_ffn=any("ffn" in k for k in enc),
        )
    del sd
    return extra


def collect(ckpt: str, temporal: bool, ds: str, limit: int):
    """跑一遍数据，只取条件，不进 DiT。"""
    from vla.load import load_vla

    extra = build_extra(ckpt, temporal)
    print(f"    extra={extra}", flush=True)
    vla = load_vla(ckpt, load_for_training=False, train_venc=True,
                   action_model_type="DiT-L", future_action_window_size=15,
                   **extra)
    vla = vla.to(torch.float32).to("cuda").eval()

    d = np.load(OUT / f"{ds}.npz", allow_pickle=True)
    n_total = len(d["action"])
    # 9 帧窗口需要前后各 4 帧，两端裁掉
    valid = [i for i in range(4, n_total - 4)][:limit]

    conds, acts = [], []
    # 直接调 _condition_from_frames 需要自行构造 pixel_values / model_dtype，
    # 那等于把 predict_action 的预处理重写一遍 —— 一旦写歪，测的就不是线上
    # 那条路径了。改成挂钩子：让 predict_action 正常跑，顺手把条件截出来。
    grabbed = {}
    orig = type(vla)._condition_from_frames

    def hooked(self, *a, **kw):
        out = orig(self, *a, **kw)
        grabbed["cond"] = out.detach().float().flatten().cpu().numpy()
        return out

    type(vla)._condition_from_frames = hooked
    try:
        for cnt, i in enumerate(valid):
            cur = Image.fromarray(d["frames_cur"][i])
            instr = str(d["lang"][i]) or "do the task"
            if temporal:
                rows = [i + o for o in OFFSETS]
                kw = dict(cond_frames=[Image.fromarray(d["frames_cur"][r]) for r in rows],
                          cond_actions=np.stack([d["action"][r][0] for r in rows[:-1]]))
            else:
                kw = dict(cond_frames=[Image.fromarray(f) for f in d["frames_fut"][i]])

            grabbed.clear()
            with torch.no_grad():
                vla.predict_action(image=cur, instruction=instr, unnorm_key=ds, **kw)
            if "cond" not in grabbed:
                raise RuntimeError("hook did not fire - condition path changed")
            conds.append(grabbed["cond"])
            acts.append(d["action"][i][:16].ravel())
            if cnt and cnt % 40 == 0:
                print(f"      {cnt}/{len(valid)}", flush=True)
    finally:
        type(vla)._condition_from_frames = orig

    del vla
    torch.cuda.empty_cache()
    return np.stack(conds), np.stack(acts)


def probe_r2(X: np.ndarray, Y: np.ndarray, alphas=(0.1, 1.0, 10.0, 100.0)) -> dict:
    """岭回归线性探针，留出集 R²。alpha 在留出集上选最好的那个 ——
    两个模型用同一套流程，比较才公平。"""
    n = len(X)
    n_tr = int(n * 0.7)
    Xtr, Xte = X[:n_tr], X[n_tr:]
    Ytr, Yte = Y[:n_tr], Y[n_tr:]

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    ymu = Ytr.mean(0)

    best = {"r2": -np.inf}
    G = Xtr.T @ Xtr
    rhs = Xtr.T @ (Ytr - ymu)
    for a in alphas:
        W = np.linalg.solve(G + a * np.eye(G.shape[0]), rhs)
        P = Xte @ W + ymu
        ss_res = ((Yte - P) ** 2).sum()
        ss_tot = ((Yte - Yte.mean(0)) ** 2).sum() + 1e-12
        r2 = float(1 - ss_res / ss_tot)
        # 逐维相关，与评测口径对齐（前 6 维）
        P3, Y3 = P.reshape(len(P), 16, -1), Yte.reshape(len(Yte), 16, -1)
        rs = [float(np.corrcoef(P3[:, :, j].ravel(), Y3[:, :, j].ravel())[0, 1])
              for j in range(6)]
        if r2 > best["r2"]:
            best = {"r2": r2, "alpha": a, "r6": float(np.nanmean(rs))}
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--temporal-for", action="append", default=[])
    ap.add_argument("--datasets", nargs="+", default=["fractal20220817_data"])
    ap.add_argument("--limit", type=int, default=256)
    ap.add_argument("--out", default=str(OUT / "probe_cond_info.json"))
    args = ap.parse_args()

    results = []
    for ds in args.datasets:
        print(f"\n=== {ds} ===", flush=True)
        for spec in args.ckpt:
            name, path = spec.split("=", 1)
            temporal = name in args.temporal_for
            print(f"  [{name}] temporal={temporal}", flush=True)
            X, Y = collect(path, temporal, ds, args.limit)
            res = probe_r2(X, Y)
            print(f"    cond_dim={X.shape[1]}  n={len(X)}  "
                  f"probe R2={res['r2']:.4f}  r6={res['r6']:.4f}  "
                  f"(alpha={res['alpha']})", flush=True)
            results.append(dict(tag=name, ds=ds, cond_dim=int(X.shape[1]),
                                n=int(len(X)), **res))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n  -> {args.out}", flush=True)

    print("\n  === 小结 ===", flush=True)
    for ds in args.datasets:
        rs = [r for r in results if r["ds"] == ds]
        if len(rs) == 2:
            a, b = rs
            print(f"  {ds}:  {a['tag']} R2={a['r2']:.4f}  |  "
                  f"{b['tag']} R2={b['r2']:.4f}  |  "
                  f"diff={b['r2'] - a['r2']:+.4f}", flush=True)
