"""[zgt] 验证块 1：状态表示 (vla/temporal.py)

关键检查（每条都事先知道答案）：
  1. 形状：输入 [B, T*P, C] -> 输出 [B, T, N, D]
  2. 时间维保留：不同帧的输出必须不同（若相同说明时间被碾平了 —— 那就退回 WoG）
  3. 初始化：精度头零初始化 -> 所有帧初始 Λ 相同，融合退化为等权
  4. β 缺席：BackwardMessage.absent 的精度必须严格为 0（不是近似 0）
  5. 融合公式：Λ* = Λ_α + Λ_ℓ + Λ_β 在 β 缺席时等价于只用 α + ℓ

结果写 /tmp/whenami_block1.json
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import torch
from vla.temporal import StateSpec, GaussianState, BackwardMessage, ObsEncoder

res = {}
torch.manual_seed(0)

B, T, P, C = 2, 4, 64, 1024
spec = StateSpec(n_tokens=16, dim=32)
enc = ObsEncoder(spec, patch_dim=C)

# 每帧用不同的常数填充，便于检测"时间是否被碾平"
feats = torch.cat([torch.full((B, P, C), float(i)) for i in range(T)], dim=1)  # [B, T*P, C]

# --- 1. 形状 ---
try:
    st = enc(feats, n_frames=T)
    res["1_shape"] = {
        "PASS": list(st.mu.shape) == [B, T, spec.n_tokens, spec.dim]
                and list(st.log_lam.shape) == [B, T, spec.n_tokens, spec.dim],
        "mu": list(st.mu.shape),
        "log_lam": list(st.log_lam.shape),
        "expected": [B, T, spec.n_tokens, spec.dim],
    }
except Exception as e:
    res["1_shape"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 2. 时间维保留：不同帧的 mu 必须不同 ---
try:
    per_frame = [st.mu[0, k].mean().item() for k in range(T)]
    all_same = max(per_frame) - min(per_frame) < 1e-6
    res["2_time_preserved"] = {
        "PASS": not all_same,
        "per_frame_mu_mean": [round(v, 5) for v in per_frame],
        "note": "各帧不同 => 时间维保留；若全相同说明被碾平（退化成 WoG）",
    }
except Exception as e:
    res["2_time_preserved"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 3. 精度头零初始化：初始 Λ 对所有帧/token 相同 ---
try:
    lam = st.precision
    res["3_lambda_init_uniform"] = {
        "PASS": bool((lam.max() - lam.min()).abs().item() < 1e-6),
        "lambda_min": round(lam.min().item(), 6),
        "lambda_max": round(lam.max().item(), 6),
        "note": "初始全相同 => 融合等权，让梯度决定谁更可信",
    }
except Exception as e:
    res["3_lambda_init_uniform"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 4. β 缺席：精度严格为 0 ---
try:
    beta = BackwardMessage.absent(st)
    p = beta.precision
    res["4_beta_absent_zero"] = {
        "PASS": bool((p == 0).all().item()),
        "max_precision": float(p.max()),
        "note": "严格为 0（非近似）=> 融合公式无需分支即可退化",
    }
except Exception as e:
    res["4_beta_absent_zero"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 5. 融合公式：β 缺席时等价于只用 α + ℓ ---
try:
    alpha = GaussianState(mu=torch.randn(B, T, 16, 32), log_lam=torch.zeros(B, T, 16, 32))
    ell = GaussianState(mu=torch.randn(B, T, 16, 32), log_lam=torch.zeros(B, T, 16, 32))
    beta = BackwardMessage.absent(alpha)

    def fuse(*parts):
        lam = sum(p.precision for p in parts)
        mu = sum(p.precision * p.mu for p in parts) / lam.clamp_min(1e-8)
        return mu, lam

    mu_with_beta, lam_with_beta = fuse(alpha, ell, beta)
    mu_without, lam_without = fuse(alpha, ell)

    res["5_fusion_degrades"] = {
        "PASS": bool(torch.allclose(mu_with_beta, mu_without, atol=1e-6)
                     and torch.allclose(lam_with_beta, lam_without, atol=1e-6)),
        "mu_diff": float((mu_with_beta - mu_without).abs().max()),
        "lam_diff": float((lam_with_beta - lam_without).abs().max()),
        "note": "含 β(精度0) 与不含 β 结果一致 => 训练/部署共用同一段代码",
    }
except Exception as e:
    res["5_fusion_degrades"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 参数量（供评估开销）---
try:
    res["_param_count"] = {
        "ObsEncoder": sum(p.numel() for p in enc.parameters()),
        "WoG_QFormer_ref": "约 100M（6 层 BERT + 1024 hidden）",
    }
except Exception:
    pass

with open("/tmp/whenami_block1.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
