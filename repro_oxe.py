"""repro_oxe.py — 用官方 ckpt 在 OXE 真实数据上做定量复现（全新 clone 环境）

目的
----
SIMPLER 主表在本机不可复现（PPU 无硬件 Vulkan，lavapipe 缺
VK_KHR_external_semaphore_fd）。本脚本做**能做的那部分**：
在 WoG 训练分布内的真实数据上，用官方两个 ckpt 评测动作预测精度。

可验证的论文定性结论
--------------------
C1  Stage-II (WoG-A) 是最终部署模型，其动作预测应**优于** Stage-I (WoG-V)。
    论文 §3.3：Stage-II 把 future condition 蒸馏进 VLM，使其能仅凭 z 推理。
C2  模型应显著优于「常数均值」平凡基线（否则说明加载/预处理有错）。
C3  上述应在 fractal 与 bridge **两个**数据集上都成立（论文 Table 1/2 分别评测）。

gripper 维的处理（上一版栽过的坑）
---------------------------------
WoG 的 predict_action（vla.py:377-380）对第 7 维做 where(<0.5, 0, 1) → 输出 {0,1}。
而 fractal 的 gripper_closedness_action 是 [-1,1] 连续值。两种编码不同，
直接算 R² 必然极差且会拖垮整体 MSE。故本脚本：
  · 主指标只用前 6 维（平移 + 旋转），这是可直接比较的
  · gripper 维单独用**二分类准确率**评（把真值按 >0 二值化后比对）

运行
----
  cd /home/zhengguantian/WoG_fresh && source env_fresh.sh
  $WOG_PY repro_oxe.py --n 150
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np
import torch

FRESH = Path("/home/zhengguantian/WoG_fresh")
WOG = FRESH / "WoG"
sys.path.insert(0, str(FRESH))          # stub_tf
import stub_tf; stub_tf.patch(verbose=False)   # 必须在 import vla 之前
sys.path.insert(0, str(WOG))
os.chdir(WOG)                            # 官方代码用硬编码相对路径找权重

from PIL import Image

OUT = Path("/mnt/pfs/zhengguantian/WoG_v2/outputs/oxe")
CKPT_ROOT = Path("/mnt/pfs/zhengguantian/WoG_v2/ckpts")
CKPTS = {
    "WoG-V (Stage-I)":  CKPT_ROOT / "WoG-V/checkpoints/step-100000-epoch-07-loss=0.1041.pt",
    "WoG-A (Stage-II)": CKPT_ROOT / "WoG-A/checkpoints/50k-1229.pt",
}
DIMS = ["dx", "dy", "dz", "roll", "pitch", "yaw", "grip"]


@torch.no_grad()
def evaluate(ckpt_path, tag, ds, n, use_temporal=False):
    from vla import load_vla
    t0 = time.time()
    # ===zgt===
    # use_temporal 经 load_vla 的 **kwargs 透传到 WoG.__init__，决定装配
    # converge（时间线）还是 qformer。装错的话 state_dict 的 key 对不上，
    # 会静默加载失败或报缺失键 —— 所以评测 WhenAmI 必须显式打开。
    # obs_offsets 需与训练时一致，否则取到的帧不是同一批。
    extra = {}
    if use_temporal:
        extra = dict(use_temporal=True, obs_offsets=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    vla = load_vla(str(ckpt_path), load_for_training=False,
                   action_model_type="DiT-L", future_action_window_size=15, **extra)
    # ===zgt===
    vla = vla.to(torch.float32).to("cuda").eval()   # bf16 下 .numpy() 会炸
    print(f"    加载 {time.time()-t0:.0f}s", flush=True)

    d = np.load(OUT / f"{ds}.npz", allow_pickle=True)
    n = min(n, len(d["frames_cur"]))
    preds, gts = [], []
    for i in range(n):
        instr = str(d["lang"][i]) or "do the task"
        act, _ = vla.predict_action(image=Image.fromarray(d["frames_cur"][i]),
                                    instruction=instr, unnorm_key=ds)
        preds.append(act)
        gts.append(d["action"][i][:16])
        if i and i % 50 == 0:
            print(f"    {i}/{n}", flush=True)
    P, G = np.stack(preds), np.stack(gts)

    # --- 前 6 维：连续量，可直接比 ---
    rows, rs = [], []
    for j in range(6):
        p, g = P[:, :, j].ravel(), G[:, :, j].ravel()
        r = float(np.corrcoef(p, g)[0, 1]) if p.std() > 1e-9 and g.std() > 1e-9 else float("nan")
        rs.append(r)
        rows.append((DIMS[j], p.mean(), g.mean(), p.std(), g.std(), r))

    P6, G6 = P[:, :, :6], G[:, :, :6]
    mse_model = float(((P6 - G6) ** 2).mean())
    mse_base = float(((G6 - G6.mean((0, 1))) ** 2).mean())    # 常数均值基线
    r2 = 1 - ((P6 - G6) ** 2).sum() / (((G6 - G6.mean((0, 1))) ** 2).sum() + 1e-12)

    # --- gripper：两个数据集编码不同，各自用合适的阈值 ---
    # fractal: gripper_closedness_action ∈ [-1,1]，是**增量**，68% 精确为 0
    #          -> 只在有明确开/合意图的样本上评（|g|>0.5），其余是"不动"无法与 {0,1} 比
    # bridge : ∈ [0,1]，是**绝对开合度**，双峰(43% 接近0 / 42% 接近1) -> 阈值 0.5
    gp = (P[:, :, 6] > 0.5).astype(int)
    graw = G[:, :, 6]
    if ds.startswith("fractal"):
        mask = np.abs(graw) > 0.5                 # 只看有明确意图的帧
        gg = (graw > 0).astype(int)
    else:
        mask = np.ones_like(graw, dtype=bool)
        gg = (graw > 0.5).astype(int)
    if mask.sum() > 0:
        grip_acc = float((gp[mask] == gg[mask]).mean())
        m = gg[mask].mean()
        grip_major = float(max(m, 1 - m))          # 多数类基线
    else:
        grip_acc = grip_major = float("nan")
    grip_n = int(mask.sum())

    print(f"\n  {'维度':<7} {'预测均值':>9} {'真实均值':>9} {'预测std':>8} {'真实std':>8} {'相关r':>7}")
    print("  " + "-" * 56)
    for nm, pm, gm, ps, gs, r in rows:
        print(f"  {nm:<7} {pm:9.4f} {gm:9.4f} {ps:8.4f} {gs:8.4f} {r:7.3f}")
    print("  " + "-" * 56)
    print(f"  平移(xyz) r = {np.nanmean(rs[:3]):.3f} | 旋转 r = {np.nanmean(rs[3:6]):.3f} "
          f"| 前6维 r = {np.nanmean(rs):.3f}")
    print(f"  前6维 R² = {r2:.3f} | MSE 模型 {mse_model:.5f} vs 基线 {mse_base:.5f} "
          f"({'降低 %.1f%% ✓' % ((1-mse_model/mse_base)*100) if mse_model < mse_base else '劣于基线 ✗'})")
    print(f"  gripper 二分类 acc = {grip_acc:.3f} (多数类基线 {grip_major:.3f}, n={grip_n})")

    del vla; torch.cuda.empty_cache()
    return dict(tag=tag, ds=ds, n=n,
                r_trans=float(np.nanmean(rs[:3])), r_rot=float(np.nanmean(rs[3:6])),
                r_6=float(np.nanmean(rs)), r2_6=float(r2),
                mse_model=mse_model, mse_base=mse_base,
                grip_acc=grip_acc, grip_major=grip_major, grip_n=grip_n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--datasets", nargs="+", default=["fractal20220817_data", "bridge_orig"])
    # ===zgt===
    # --ckpt NAME=PATH：评测我们自己训出来的 checkpoint，可重复传。
    # 不传则退回官方 WoG-V / WoG-A（原行为，用于复现论文对照）。
    # WhenAmI 的 checkpoint 含 converge 模块，需要 --temporal 让 load_vla
    # 用同样的结构装载，否则 key 对不上。
    ap.add_argument("--ckpt", action="append", metavar="NAME=PATH",
                    help="evaluate a custom checkpoint; repeatable")
    ap.add_argument("--temporal", action="store_true",
                    help="load checkpoints with the temporal (converge) head")
    a = ap.parse_args()

    if a.ckpt:
        ckpts = {}
        for spec in a.ckpt:
            if "=" not in spec:
                raise SystemExit(f"--ckpt expects NAME=PATH, got: {spec}")
            name, path = spec.split("=", 1)
            p = Path(path)
            if not p.exists():
                raise SystemExit(f"checkpoint not found: {p}")
            ckpts[name] = p
    else:
        ckpts = CKPTS
    # ===zgt===

    results = []
    for ds in a.datasets:
        for tag, ck in ckpts.items():
            print(f"\n{'='*70}\n{tag}  ×  {ds}\n{'='*70}", flush=True)
            results.append(evaluate(ck, tag, ds, a.n, use_temporal=a.temporal))

    # ---- 汇总 + 判定论文定性结论 ----
    print(f"\n\n{'='*70}\n汇总\n{'='*70}")
    print(f"{'ckpt':<18} {'数据集':<24} {'r(前6维)':>9} {'R²':>7} {'MSE↓%':>7} {'grip acc':>9}")
    print("-" * 70)
    for x in results:
        print(f"{x['tag']:<18} {x['ds']:<24} {x['r_6']:9.3f} {x['r2_6']:7.3f} "
              f"{(1-x['mse_model']/x['mse_base'])*100:6.1f}% {x['grip_acc']:9.3f}")

    # ===zgt===
    # 自定义 ckpt 时跳过论文结论判定 —— 那段假定结果里恰好有
    # Stage-I / Stage-II 两个官方 tag，对自训 checkpoint 不适用。
    if a.ckpt:
        raise SystemExit(0)
    # ===zgt===

    print(f"\n{'='*70}\n论文定性结论判定\n{'='*70}")
    for ds in a.datasets:
        v = next(x for x in results if x["ds"] == ds and "Stage-I)" in x["tag"])
        aa = next(x for x in results if x["ds"] == ds and "Stage-II" in x["tag"])
        ok = aa["r_6"] > v["r_6"]
        print(f"  [C1] {ds}: Stage-II r={aa['r_6']:.3f} vs Stage-I r={v['r_6']:.3f} "
              f"-> {'✓ 符合论文' if ok else '✗ 与论文相反'}")
    for x in results:
        ok = x["mse_model"] < x["mse_base"]
        print(f"  [C2] {x['tag']:<18} {x['ds']:<24} 优于常数基线 -> {'✓' if ok else '✗'}")

    Path("/mnt/pfs/zhengguantian/WoG_v2/outputs").mkdir(parents=True, exist_ok=True)
    with open("/mnt/pfs/zhengguantian/WoG_v2/outputs/repro_oxe.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n结果已存 /mnt/pfs/zhengguantian/WoG_v2/outputs/repro_oxe.json")
