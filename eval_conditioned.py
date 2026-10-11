"""[zgt] Stage I 条件质量评测 —— 让条件真正走进 DiT。

为什么需要这个脚本：
    repro_oxe.py 走 predict_action 的默认路径，而那条路径调用
    `action_model.inference(z)` 时不传 per_token，DiT 于是回退到
    `uncondition` 占位符 —— 训练时条件是必传的，评测却整段绕过。
    结果是两个模型都在被当作"无条件 DiT"来测，比出来的数字与
    各自的条件机制无关。

本脚本改为：
    两边都喂真实未来帧，各自算出条件后注入 DiT。
      WoG        : 5 帧 -> QFormer -> O^c
      WhenAmI    : 9 帧 -> alpha/beta 汇聚 -> 当下节点 s*_t
    这是 Stage I 应有的衡量口径 —— 它评的是**条件质量**，
    而非可部署的策略能力（策略能力要等 Stage II 撤掉未来之后才谈）。

数据：
    .npz 里同一 episode 的相邻采样点间隔恰为 4，与 obs_offsets
    [-4..4] 对齐，故可就地拼出 9 帧时间线，无需重新导出数据。

用法：
    python3 eval_conditioned.py --ckpt NAME=PATH [--temporal] [--n 150]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

OUT = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/oxe")
DIMS = ["dx", "dy", "dz", "roll", "pitch", "yaw", "grip"]


def build_index(d):
    """(ep_id, tstep) -> 行号，用于就地拼出时间线。"""
    return {(int(e), int(t)): i
            for i, (e, t) in enumerate(zip(d["ep_id"], d["tstep"]))}


def collect_samples(d, idx, offsets, n):
    """挑出能凑齐完整 offsets 的样本。

    过去帧不是随便找一帧就行 —— 必须来自同一 episode 且时间间隔正确，
    否则喂给 alpha 的是别的轨迹，测出来的东西没有意义。
    """
    step = 4          # .npz 内相邻采样点的 tstep 间隔，实测恒为 4
    out = []
    for i, (e, t) in enumerate(zip(d["ep_id"], d["tstep"])):
        e, t = int(e), int(t)
        rows = [idx.get((e, t + o * step)) for o in offsets]
        if all(r is not None for r in rows):
            out.append((i, rows))
            if len(out) >= n:
                break
    return out


@torch.no_grad()
def evaluate(ckpt_path, tag, ds, n, use_temporal, conditioned):
    from vla import load_vla

    offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4] if use_temporal else None
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
        extra = dict(use_temporal=True, obs_offsets=offsets, converge_type=ct)
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

    # train_venc 必须与训练时一致（训练用的是 True）。
    # 它决定 ActionModel 是否构建 per_attn：
    #     use_per_attn = True if train_venc else False
    # load_vla 默认 False，于是评测加载出来的 DiT 根本没有 per_attn，
    # 条件无论怎么传都不会影响输出 —— 实测 |out(cond) - out(none)| = 0。
    # 日志里那行 "Freeze conditioning module" 就是走进了这个分支。
    t0 = time.time()
    vla = load_vla(str(ckpt_path), load_for_training=False, train_venc=True,
                   action_model_type="DiT-L", future_action_window_size=15, **extra)
    vla = vla.to(torch.float32).to("cuda").eval()      # bf16 下 .numpy() 会炸
    print(f"    加载 {time.time()-t0:.0f}s", flush=True)

    d = np.load(OUT / f"{ds}.npz", allow_pickle=True)
    idx = build_index(d)

    # 两条路径评同一批样本：都要求同 episode 内能凑齐 9 帧。
    # WoG 本身只需 5 帧，但若各自用不同子集，r/R² 的差异就分不清
    # 是方法差异还是样本差异。align 保证唯一变量是条件机制。
    if conditioned:
        align = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        samples = collect_samples(d, idx, align, n)
        print(f"    可用样本 {len(samples)}（两条路径共用，需凑齐 9 帧）", flush=True)
    else:
        samples = [(i, None) for i in range(min(n, len(d["frames_cur"])))]

    preds, gts = [], []
    for cnt, (i, rows) in enumerate(samples):
        instr = str(d["lang"][i]) or "do the task"
        cur = Image.fromarray(d["frames_cur"][i])

        kw = {}
        if conditioned:
            if use_temporal:
                frames = [Image.fromarray(d["frames_cur"][r]) for r in rows]
                # 相邻节点之间的动作：取每段起点那一帧的首个动作
                acts = np.stack([d["action"][r][0] for r in rows[:-1]])
                kw = dict(cond_frames=frames, cond_actions=acts)
            else:
                # WoG：当前帧 + 4 帧未来，与训练时的 5 帧一致
                kw = dict(cond_frames=[Image.fromarray(f) for f in d["frames_fut"][i]])

        act, _ = vla.predict_action(image=cur, instruction=instr,
                                    unnorm_key=ds, **kw)
        preds.append(act)
        gts.append(d["action"][i][:16])
        if cnt and cnt % 50 == 0:
            print(f"    {cnt}/{len(samples)}", flush=True)

    P, G = np.stack(preds), np.stack(gts)

    rs = []
    print(f"  {'维度':<8}{'预测std':>10}{'真实std':>10}{'相关r':>9}")
    print("  " + "-" * 40)
    for j in range(6):
        p, g = P[:, :, j].ravel(), G[:, :, j].ravel()
        r = float(np.corrcoef(p, g)[0, 1]) if p.std() > 1e-9 and g.std() > 1e-9 else float("nan")
        rs.append(r)
        print(f"  {DIMS[j]:<8}{p.std():>10.4f}{g.std():>10.4f}{r:>9.3f}")

    P6, G6 = P[:, :, :6], G[:, :, :6]
    r2 = float(1 - ((P6 - G6) ** 2).sum() / (((G6 - G6.mean((0, 1))) ** 2).sum() + 1e-12))
    r6 = float(np.nanmean(rs))
    print("  " + "-" * 40)
    print(f"  前6维 r = {r6:.3f} | R² = {r2:.3f} | n = {len(P)}")
    return dict(tag=tag, ds=ds, r_6=r6, r2_6=r2, n=len(P))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--temporal", action="store_true")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--datasets", nargs="+", default=["fractal20220817_data"])
    ap.add_argument("--no-cond", action="store_true",
                    help="run the old unconditioned path, for comparison")
    a = ap.parse_args()

    results = []
    for ds in a.datasets:
        for spec in a.ckpt:
            name, path = spec.split("=", 1)
            p = Path(path)
            if not p.exists():
                raise SystemExit(f"checkpoint not found: {p}")
            mode = "无条件" if a.no_cond else "带条件"
            print(f"\n{'='*66}\n{name}  ×  {ds}  [{mode}]\n{'='*66}", flush=True)
            results.append(evaluate(p, name, ds, a.n, a.temporal, not a.no_cond))

    print(f"\n{'='*66}\n汇总\n{'='*66}")
    print(f"{'ckpt':<20}{'数据集':<26}{'r':>8}{'R²':>9}{'n':>6}")
    print("-" * 66)
    for x in results:
        print(f"{x['tag']:<20}{x['ds']:<26}{x['r_6']:>8.3f}{x['r2_6']:>9.3f}{x['n']:>6}")
