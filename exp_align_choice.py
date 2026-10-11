"""[zgt] 对齐空间的选型实验 —— 双向推演的地基。

为什么必须先做这个：
    文献（Cycle Consistency in Video Object-Centric Learning）明确警告，
    在 latent 空间强行做循环一致性是病态的：它惩罚合法变化，把表示推向
    塌缩的平均值。建议改为在"重建流形"上对齐。

    但 patch feature 是 64x1024 维，直接 L2 很可能被高频噪声主导 ——
    模型会学"预测平均 patch"而不是真正的动力学。那样双向推演的地基就没了。

    故先用真实 OXE 数据做一个小实验：同样的转移算子 T，只换对齐损失，
    看哪种能学出真动力学。

判据（这是关键，不能只看 loss 下降）：
    1. 预测 vs 真实的相关性        —— 真的在预测，而非输出常数
    2. 预测的方差 / 真实的方差      —— 塌缩的话方差会远小于 1
    3. 区分力：预测 t+1 与 t+4 的差异 —— 塌缩的话两者会趋同
    4. 对比"永远输出上一帧"的基线   —— 至少要赢过这个平凡解

结果写 /tmp/align_choice.json
"""
import json
import sys
import time
from pathlib import Path

_T0 = time.time()
_PROG = Path("/tmp/align_progress.txt")
_PROG.write_text("start\n")


def log(m):
    """进度写文件 —— 本环境下后台进程的 stdout 重定向不可靠。

    用追加模式：早先版本每次都 read_text()+write_text()，在训练循环里
    反复调用会让 I/O 随文件增长而变慢，看起来像训练卡住。
    """
    with _PROG.open("a") as f:
        f.write(f"[{time.time()-_T0:6.1f}s] {m}\n")

import numpy as np
log("importing torch (PPU init 较慢)")
import torch
import torch.nn as nn
import torch.nn.functional as F
log("torch ready")

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

OUT = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/oxe")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OFFSETS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
STEP = 4


def collect(d, limit):
    """同 episode 内凑齐 9 帧的样本。"""
    idx = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(d["ep_id"], d["tstep"]))}
    out = []
    for i, (e, t) in enumerate(zip(d["ep_id"], d["tstep"])):
        e, t = int(e), int(t)
        rows = [idx.get((e, t + o * STEP)) for o in OFFSETS]
        if all(r is not None for r in rows):
            out.append(rows)
            if len(out) >= limit:
                break
    return out


class TinyT(nn.Module):
    """最小的转移算子：s' = s + net([s, a, dt, dir])。

    残差形式 + 零初始化末层 = 恒等起步（安全：在残差分支上）。
    动作、跨度、方向都作为条件 —— 这正是要验证的设计。
    """

    def __init__(self, dim, action_dim=7, hidden=512):
        super().__init__()
        self.act = nn.Linear(action_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(dim + 64 + 3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, a, dt, direction):
        B = s.shape[0]
        # dt 同时给归一化值与周期编码：前者利于外推，后者对未见跨度插值更稳
        w = dt / 16.0
        cond = torch.stack([w, torch.sin(w * np.pi), torch.cos(w * np.pi)], -1)
        cond = cond.expand(B, 3) if cond.dim() == 1 else cond
        a_e = self.act(a)
        x = torch.cat([s, a_e, cond * direction], -1)
        return s + self.net(x)


def run(mode, feats, acts, proj, steps=400, seed=0):
    """训一个 T，用指定的对齐损失，返回诊断指标。

    feats: [N, 9, P, C] 真实 patch feature
    acts:  [N, 8, 7]    相邻帧之间的等效动作
    """
    torch.manual_seed(seed)
    N, K, P, C = feats.shape
    D = 256                                   # 状态维度
    enc = nn.Linear(C, D).to(DEV)             # patch -> 状态（逐 patch）
    T = TinyT(D).to(DEV)
    dec = nn.Linear(D, proj.shape[1]).to(DEV)  # 状态 -> 对齐空间
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(T.parameters()) + list(dec.parameters()), lr=1e-3
    )

    def align(pred, target):
        if mode == "l2_raw":
            return (pred - target).pow(2).mean()
        if mode == "cos_raw":
            return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
        if mode == "l2_proj":
            return (pred - target).pow(2).mean()
        if mode == "cos_proj":
            return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
        raise ValueError(mode)

    use_proj = mode.endswith("_proj")
    for it in range(steps):
        b = torch.randint(0, N, (16,), device=DEV)
        f, a = feats[b], acts[b]
        tgt_all = (f @ proj) if use_proj else f          # [16, 9, P, ?]
        s = enc(f[:, 0]).mean(1)                          # 从 t-4 起，池化成单状态
        loss = 0.0
        for k in range(K - 1):
            s = T(s, a[:, k], torch.full((s.shape[0],), 4.0, device=DEV), 1.0)
            loss = loss + align(dec(s), tgt_all[:, k + 1].mean(1))
        opt.zero_grad(set_to_none=True)
        (loss / (K - 1)).backward()
        opt.step()

    # ---- 诊断：在留出样本上看它到底学到了什么 ----
    with torch.no_grad():
        f, a = feats[-32:], acts[-32:]
        tgt_all = (f @ proj) if use_proj else f
        s = enc(f[:, 0]).mean(1)
        preds, tgts = [], []
        for k in range(K - 1):
            s = T(s, a[:, k], torch.full((s.shape[0],), 4.0, device=DEV), 1.0)
            preds.append(dec(s))
            tgts.append(tgt_all[:, k + 1].mean(1))
        P_ = torch.stack(preds, 1)            # [32, 8, d]
        G_ = torch.stack(tgts, 1)

        # 1. 相关性：真的在跟踪目标吗
        p, g = P_.flatten().cpu().numpy(), G_.flatten().cpu().numpy()
        r = float(np.corrcoef(p, g)[0, 1]) if p.std() > 1e-9 else float("nan")

        # 2. 方差比：塌缩的话预测方差远小于目标
        var_ratio = float(P_.std() / (G_.std() + 1e-9))

        # 3. 区分力：t+1 与 t+4 的预测该有明显差异
        sep_pred = float((P_[:, 0] - P_[:, -1]).abs().mean())
        sep_tgt = float((G_[:, 0] - G_[:, -1]).abs().mean())

        # 4. 平凡基线：永远输出第一帧
        base = tgt_all[:, 0].mean(1, keepdim=True).expand_as(G_)
        err_model = float((P_ - G_).pow(2).mean())
        err_base = float((base - G_).pow(2).mean())

    return dict(
        mode=mode, r=round(r, 4), var_ratio=round(var_ratio, 4),
        separation=round(sep_pred / (sep_tgt + 1e-9), 4),
        beats_trivial=bool(err_model < err_base),
        mse_ratio=round(err_model / (err_base + 1e-12), 4),
    )


if __name__ == "__main__":
    log("加载数据…")
    d = np.load(OUT / "fractal20220817_data.npz", allow_pickle=True)
    rows = collect(d, 160)
    log(f"  样本 {len(rows)}")

    # 用 DINOv2 的 patch feature 太重，这里用图像本身做低维代理：
    # 224x224x3 -> 8x8 网格的均值，得到 [64, 3]，再升维到伪 patch 特征。
    # 目的只是比较【对齐损失的形式】，不是复现真实编码器。
    # .npz 的逐元素索引会反复解压整个数组 —— 1440 次随机访问要跑二十分钟。
    # 先整体读入再按下标取。
    all_frames = d["frames_cur"]
    imgs = np.stack([[all_frames[r] for r in row] for row in rows])   # [N,9,224,224,3]
    N, K = imgs.shape[:2]
    x = torch.as_tensor(imgs, dtype=torch.float32, device=DEV) / 255.0
    x = x.view(N * K, 224, 224, 3).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x, (8, 8)).permute(0, 2, 3, 1).reshape(N, K, 64, 3)
    # 升到 256 维伪特征（固定随机矩阵，模拟 ViT 输出的冗余性）
    torch.manual_seed(0)
    up = torch.randn(3, 256, device=DEV) / 3 ** 0.5
    feats = x @ up                                            # [N, 9, 64, 256]

    # 相邻节点之间的等效动作：区间内动作求和（delta EEF 相加即合成位移）
    acts = torch.as_tensor(
        np.stack([[d["action"][r][0] for r in row[:-1]] for row in rows]),
        dtype=torch.float32, device=DEV,
    )
    log(f"  feats {tuple(feats.shape)}  acts {tuple(acts.shape)}")

    proj = torch.randn(256, 64, device=DEV) / 256 ** 0.5      # 固定随机投影

    res = []
    for mode in ["l2_raw", "cos_raw", "l2_proj", "cos_proj"]:
        r = run(mode, feats, acts, proj)
        res.append(r)
        log(f"  {mode:10} r={r['r']:+.4f}  方差比={r['var_ratio']:.3f}  "
            f"区分力={r['separation']:.3f}  胜过平凡={r['beats_trivial']}  "
            f"MSE比={r['mse_ratio']:.3f}")

    Path("/tmp/align_choice.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("\n  判据：r 高、方差比接近 1、区分力接近 1、胜过平凡解")
