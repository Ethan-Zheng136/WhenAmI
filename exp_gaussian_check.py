"""[zgt] 验证高斯假设：ObsEncoder 的精度 Lambda 能否反映观测质量。

为什么这个实验必须在写块 3 之前做
--------------------------------
整个 combine 设计（Lambda* = Lambda_alpha + Lambda_ell + Lambda_beta）建立在
一个前提上：观测越差，Lambda_ell 越小，融合时自动让位给其他时刻的证据。
如果 latent 空间里 Lambda 对观测质量不敏感，那么精度加权退化成一个奇怪的
门控，"自适应消歧"这个卖点就不成立，块 3 必须换路。

实验设计
--------
取 OXE 真实帧，人工制造四种退化，比较 Lambda 的均值：
    clean     原图
    occluded  中心 40% 区域涂黑（模拟遮挡）
    blurred   下采样再上采样（模拟运动模糊）
    dark      亮度压到 20%（模拟欠曝）

预测（写在跑之前，不许改）
------------------------
  P1  未训练的 encoder：四组 Lambda 应当**几乎相同**
      —— 因为 to_log_lam 是零初始化的，输出恒为 exp(init_log_lam)=1。
      这条是自检项：如果未训练就出现差异，说明零初始化没生效，代码有 bug。

  P2  mu 应当有差异 —— 编码器确实"看到"了退化，只是还没学会用 Lambda 表达。
      若 mu 也无差异，说明 patch 特征本身对退化不敏感，那整条路都有问题。

结论的解读
---------
  本实验**不能**证明高斯假设成立（那需要训练后再测）。
  它能证明的是：(a) 零初始化生效，(b) 视觉特征确实携带了退化信息。
  两者都成立，才值得投入训练去验证 Lambda 的自适应性。

用法:  python3 exp_gaussian_check.py
输出:  /tmp/whenami_gauss.json
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import numpy as np
import torch

from vla.temporal import StateSpec, ObsEncoder

OXE_NPZ = "/mnt/pfs/zhengguantian/WoG_v2/outputs/oxe/bridge_orig.npz"
N_SAMPLES = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def degrade(imgs: np.ndarray, kind: str) -> np.ndarray:
    """Apply a synthetic degradation to a batch of HWC uint8 images."""
    out = imgs.copy()
    if kind == "clean":
        return out
    if kind == "occluded":
        h, w = out.shape[1:3]
        y0, y1 = int(h * 0.3), int(h * 0.7)
        x0, x1 = int(w * 0.3), int(w * 0.7)
        out[:, y0:y1, x0:x1] = 0
        return out
    if kind == "blurred":
        t = torch.from_numpy(out).permute(0, 3, 1, 2).float()
        small = torch.nn.functional.interpolate(t, scale_factor=0.125, mode="bilinear", align_corners=False)
        back = torch.nn.functional.interpolate(small, size=t.shape[-2:], mode="bilinear", align_corners=False)
        return back.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    if kind == "dark":
        return (out.astype(np.float32) * 0.2).astype(np.uint8)
    raise ValueError(f"unknown degradation: {kind}")


def to_patch_features(imgs: np.ndarray) -> torch.Tensor:
    """Stand-in for the frozen DINOv2 path.

    We do not load the 7B checkpoint here -- the question under test is whether
    *some* visual signal survives into Lambda, not what DINOv2 specifically encodes.
    A fixed random projection of raw patches preserves the relative differences
    between degradations while keeping the experiment cheap.
    """
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0     # [B,3,H,W]
    t = torch.nn.functional.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)
    # 8x8 patches of 8x8 pixels -> 64 patches, each 3*8*8=192 dims
    p = t.unfold(2, 8, 8).unfold(3, 8, 8)                              # [B,3,8,8,8,8]
    p = p.permute(0, 2, 3, 1, 4, 5).reshape(t.shape[0], 64, -1)        # [B,64,192]
    g = torch.Generator().manual_seed(0)
    proj = torch.randn(p.shape[-1], 1024, generator=g) * (p.shape[-1] ** -0.5)
    return p @ proj                                                     # [B,64,1024]


def main():
    res = {"device": DEVICE}

    d = np.load(OXE_NPZ, allow_pickle=True)
    imgs = d["frames_cur"][:N_SAMPLES]                                  # [N,224,224,3] uint8
    res["n_samples"] = int(imgs.shape[0])

    spec = StateSpec(n_tokens=16, dim=32)
    enc = ObsEncoder(spec, patch_dim=1024).to(DEVICE).eval()

    stats = {}
    with torch.no_grad():
        for kind in ["clean", "occluded", "blurred", "dark"]:
            feats = to_patch_features(degrade(imgs, kind)).to(DEVICE)   # [N,64,1024]
            st = enc(feats, n_frames=1)                                 # treat each image as a 1-frame clip
            stats[kind] = {
                "lambda_mean": float(st.precision.mean()),
                "lambda_std": float(st.precision.std()),
                "mu_mean": float(st.mu.mean()),
                "mu_std": float(st.mu.std()),
            }
    res["stats"] = stats

    lams = [stats[k]["lambda_mean"] for k in stats]
    mus = [stats[k]["mu_std"] for k in stats]

    # P1: untrained encoder -> Lambda identical across degradations (zero-init works)
    res["P1_lambda_uniform_when_untrained"] = {
        "PASS": (max(lams) - min(lams)) < 1e-5,
        "lambda_spread": float(max(lams) - min(lams)),
        "note": "zero-init on to_log_lam means Lambda must be constant before training",
    }

    # P2: mu does differ -> visual features carry the degradation signal
    res["P2_mu_reflects_degradation"] = {
        "PASS": (max(mus) - min(mus)) > 1e-4,
        "mu_std_spread": float(max(mus) - min(mus)),
        "per_kind_mu_std": {k: round(v["mu_std"], 5) for k, v in stats.items()},
        "note": "if mu is flat too, patch features are blind to degradation -- design is in trouble",
    }

    with open("/tmp/whenami_gauss.json", "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
