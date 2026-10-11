"""[zgt] 线性 probe —— 条件表示里到底编码了多少「当下」。

为什么需要它：
    动作预测是间接指标，要穿过「条件 -> DiT -> 16 步扩散 -> 动作」整条链，
    任何一环都可能掩盖条件本身的质量。probe 直接问一个更根本的问题：
    这个 [16,32] 的向量里，关于「当下是什么状态」的信息有多少？

    用**线性** probe 是有意的：若需要复杂解码器才能读出，说明信息藏得太深，
    下游用不上。线性可读才算「表示里真的有」。

三个对照（这是本脚本的重点）：
    1. WoG 的 O^c        vs  WhenAmI 的 s*_t     —— 谁编码得更好
    2. WhenAmI 的 s*_t   vs  只用 alpha 的 s_α   —— beta 到底贡献了多少
    3. Lambda* 与「画面清晰度」是否相关           —— 精度是否学到了意义

探测目标（.npz 里现成的标签）：
    gt_togo  [N,3]  到目标的剩余位移 —— 未来证据最该帮上忙的地方
    gt_eef   [N,7]  末端位姿
    gt_grip  [N,1]  夹爪开合（二分类）
    progress [N,1]  tstep / ep_len，任务进度

结果写 --out 指定的 json。
"""
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
STEP = 4          # .npz 内相邻采样点的 tstep 间隔，实测恒为 4


def collect_samples(d, limit=None):
    """挑出同 episode 内能凑齐完整时间线的样本。"""
    idx = {(int(e), int(t)): i
           for i, (e, t) in enumerate(zip(d["ep_id"], d["tstep"]))}
    out = []
    for i, (e, t) in enumerate(zip(d["ep_id"], d["tstep"])):
        e, t = int(e), int(t)
        rows = [idx.get((e, t + o * STEP)) for o in OFFSETS]
        if all(r is not None for r in rows):
            out.append((i, rows))
            if limit and len(out) >= limit:
                break
    return out


@torch.no_grad()
def extract(ckpt_path, use_temporal, ds, samples, mode="full"):
    """跑一遍模型，取出条件表示。

    mode:
      full       正常路径（WhenAmI 走 alpha+beta 融合；WoG 走 QFormer）
      alpha_only 仅 WhenAmI：物理截断到当下为止，beta 缺席。
                 与 full 的差值就是 beta 的净贡献。
    """
    from vla import load_vla

    # converge_type 必须与训练一致：masked 与 recurrent 的模块结构不同，
    # 传错会让 state_dict 的 key 对不上（静默加载失败比报错更糟）。
    # 从 checkpoint 同目录的 config.json 读，而不是靠调用方记得传。
    extra = {}
    if use_temporal:
        # 从**权重本身**判断，而不是 config.json。config 记录的是提交时的
        # 配置，未必等于实际构造的模型 —— 参数透传曾经断链，config 写着
        # masked 而权重是 recurrent，两个 320 卡任务因此白跑。
        # 键名是结构的事实，配置只是意图。
        import torch as _t
        _sd = _t.load(str(ckpt_path), map_location="cpu", mmap=True,
                      weights_only=False)["model"].get("converge", {})
        # 三种实现的键名互斥，用它判定。新增 flow 时必须同步更新这里 ——
        # 漏掉就会被误判成 recurrent，加载时 key 全对不上。
        if any(k.startswith("flow.") for k in _sd):
            ct = "flow"
        elif any(k.startswith("q_past.") for k in _sd):
            ct = "masked"
        elif any("obs_encoder" in k for k in _sd):
            ct = "recurrent"
        else:
            raise ValueError(
                f"cannot infer converge_type from checkpoint keys: "
                f"{sorted(_sd)[:5]}... (expected flow./q_past./obs_encoder.)"
            )
        extra = dict(use_temporal=True, obs_offsets=OFFSETS, converge_type=ct)
        if ct == "flow":
            # flow 的编码器容量是可配的，不同 run 结构不同（v2 无 FFN 4.5M、
            # v3 有 FFN 44.3M）。从权重本身推断，不靠调用方记得传环境变量 ——
            # 配置漏传导致 key 对不上，正是此前白跑 20 小时的那类事故。
            enc = {k: v for k, v in _sd.items() if k.startswith("encoder.")}
            use_ffn = any("ffn" in k for k in enc)
            n_enc = len({k.split(".")[2] for k in enc if k.startswith("encoder.blocks.")})
            n_cross = len({k.split(".")[2] for k in enc
                           if k.startswith("encoder.cross_frame.")})
            hidden = enc["encoder.query"].shape[-1]
            extra.update(flow_enc_layers=n_enc, flow_cross_layers=n_cross,
                         flow_enc_hidden=int(hidden), flow_enc_use_ffn=use_ffn)
            print(f"    flow enc: {n_enc}+{n_cross} 层 h={hidden} ffn={use_ffn}"
                  f"（据权重推断）", flush=True)
        del _sd
        print(f"    converge_type={ct}（据权重键名判定）", flush=True)
    # train_venc=True 必须与训练一致：它决定 ActionModel 是否构建 per_attn，
    # 也决定 conditioning module 是否被冻结。
    vla = load_vla(str(ckpt_path), load_for_training=False, train_venc=True,
                   action_model_type="DiT-L", future_action_window_size=15, **extra)
    vla = vla.to(torch.float32).to("cuda").eval()

    d = np.load(OUT / f"{ds}.npz", allow_pickle=True)
    tf = vla.vlm.vision_backbone.image_transform
    dev = vla.vlm.device

    feats, lams = [], []
    for cnt, (i, rows) in enumerate(samples):
        cur = Image.fromarray(d["frames_cur"][i])
        px = tf(cur)
        pixel_values = ({k: v[None].to(dev) for k, v in px.items()}
                        if isinstance(px, dict) else px[None].to(dev))

        if use_temporal:
            keep = rows if mode == "full" else rows[:OFFSETS.index(0) + 1]
            imgs = [Image.fromarray(d["frames_cur"][r]) for r in keep]
            stacked = {}
            for im in imgs:
                p = tf(im)
                for k, v in (p.items() if isinstance(p, dict) else [(None, p)]):
                    stacked.setdefault(k, []).append(v)
            frames = {k: torch.stack(v, 0)[None].to(dev) for k, v in stacked.items()}

            now_idx = OFFSETS.index(0)
            past = {k: v[:, :now_idx] for k, v in frames.items()} if now_idx else None
            future = ({k: v[:, now_idx + 1:] for k, v in frames.items()}
                      if mode == "full" else
                      {k: v[:, :0] for k, v in frames.items()})   # 空 future
            acts = torch.as_tensor(
                np.stack([d["action"][r][0] for r in keep[:-1]]), device=dev
            )[None].float()

            if mode == "full":
                cond, _, _ = vla._compute_condition(pixel_values, future, past, acts)
            else:
                # alpha_only：物理截断观测，而不只是把 beta 置零 ——
                # 否则 alpha 仍会从未来帧吸到信息，那是泄漏不是消融。
                args, kwargs = _student_inputs(vla, pixel_values, past, acts, now_idx)
                cond = vla.converge(*args, **kwargs)
                cond = cond if torch.is_tensor(cond) else cond.mu
                cond = cond[:, now_idx] if cond.dim() == 4 else cond
        else:
            future = {}
            fut_imgs = [Image.fromarray(f) for f in d["frames_fut"][i]]
            stacked = {}
            for im in fut_imgs:
                p = tf(im)
                for k, v in (p.items() if isinstance(p, dict) else [(None, p)]):
                    stacked.setdefault(k, []).append(v)
            future = {k: torch.stack(v, 0)[None].to(dev) for k, v in stacked.items()}
            cond, _, _ = vla._compute_condition(pixel_values, future, None, None)

        feats.append(cond.flatten().float().cpu().numpy())
        if cnt and cnt % 25 == 0:
            print(f"    {cnt}/{len(samples)}", flush=True)

    del vla
    torch.cuda.empty_cache()
    return np.stack(feats)


def _student_inputs(vla, pixel_values, past, acts, now_idx):
    """复用 _compute_condition 的前半段，拼出 student 路径的输入。"""
    now_feats, _ = vla._encode_frames(
        {k: v.unsqueeze(1) for k, v in pixel_values.items()}, pixel_values
    )
    if past:
        past_feats, _ = vla._encode_frames(past, pixel_values)
        seq = torch.cat([past_feats, now_feats], dim=1)
        n_past = next(iter(past.values())).shape[1]
    else:
        seq, n_past = now_feats, 0
    return (seq,), dict(n_frames=n_past + 1, actions=acts[:, :n_past],
                        n_past=n_past, use_future=False)


def ridge_probe(X, Y, n_folds=5, alpha=None):
    """闭式岭回归 + K 折交叉验证，返回 held-out R²。

    用岭而非普通最小二乘：特征 512 维而样本仅几百，OLS 会把训练集拟合到
    R²≈1、held-out 变成大负数，什么也测不出来。正则强度按 (特征数/样本数)
    缩放 —— 样本越少压得越狠。两个模型用同一个公式，保证公平。
    """
    if alpha is None:
        alpha = max(1.0, 10.0 * X.shape[1] / max(len(X), 1))
    N = len(X)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xs = np.hstack([Xs, np.ones((N, 1))])
    idx = np.arange(N)
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)

    preds = np.zeros_like(Y, dtype=float)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        A = Xs[tr].T @ Xs[tr] + alpha * np.eye(Xs.shape[1])
        W = np.linalg.solve(A, Xs[tr].T @ Y[tr])
        preds[f] = Xs[f] @ W
    ss_res = ((Y - preds) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-12))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wog", required=True)
    ap.add_argument("--whenami", required=True)
    ap.add_argument("--ds", default="fractal20220817_data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/tmp/probe_result.json")
    a = ap.parse_args()

    d = np.load(OUT / f"{a.ds}.npz", allow_pickle=True)
    samples = collect_samples(d, a.limit or None)
    rows = [i for i, _ in samples]
    print(f"\n{'='*64}\n{a.ds}: {len(samples)} 个样本\n{'='*64}", flush=True)

    targets = {
        "gt_togo (剩余位移)": d["gt_togo"][rows],
        "gt_eef (末端位姿)": d["gt_eef"][rows],
        "progress (任务进度)": (d["tstep"][rows] / d["ep_len"][rows])[:, None],
        "gt_grip (夹爪)": d["gt_grip"][rows],
    }

    reps = {}
    print("\n--- 提取 WoG 的 O^c ---", flush=True)
    reps["WoG O^c"] = extract(a.wog, False, a.ds, samples)
    print("\n--- 提取 WhenAmI 的 s*_t (alpha+beta) ---", flush=True)
    reps["WhenAmI s*_t"] = extract(a.whenami, True, a.ds, samples, "full")
    print("\n--- 提取 WhenAmI 的 s_alpha (无 beta) ---", flush=True)
    try:
        reps["WhenAmI alpha only"] = extract(a.whenami, True, a.ds, samples, "alpha_only")
    except Exception as e:
        print(f"    alpha_only 提取失败（beta 贡献无法量化）: {type(e).__name__}: {e}")

    out = {}
    print(f"\n{'='*64}\n线性 probe: held-out R²（5 折）\n{'='*64}")
    print(f"{'探测目标':<22}" + "".join(f"{k:>22}" for k in reps))
    print("-" * (22 + 22 * len(reps)))
    for tname, Y in targets.items():
        row, scores = f"{tname:<22}", {}
        for rname, X in reps.items():
            s = ridge_probe(X, Y.astype(float))
            scores[rname] = s
            row += f"{s:>22.4f}"
        print(row)
        out[tname] = scores

    if "WhenAmI alpha only" in reps:
        print(f"\n{'='*64}\nbeta 的净贡献（s*_t − alpha only）\n{'='*64}")
        for tname, sc in out.items():
            delta = sc["WhenAmI s*_t"] - sc["WhenAmI alpha only"]
            flag = "未来有贡献" if delta > 0.01 else ("持平" if delta > -0.01 else "反而更差")
            print(f"  {tname:<22} {delta:+.4f}   {flag}")

    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n  结果写入 {a.out}")
