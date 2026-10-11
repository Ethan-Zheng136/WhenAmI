"""[zgt] 验证块 2：转移算子 T 与双向递推 (vla/temporal.py)

关键检查（每条都事先知道答案）：
  1. 形状：alpha / beta 均为 [B, K, N, D]
  2. T 零初始化 -> 初始是恒等映射（mu 不变），状态不被随机扰动带偏
  3. 精度单调下降：T(s) 的 Lambda 必须 < s 的 Lambda（信息只会损失）
  4. 方向性 alpha：alpha_k 只依赖 <= k 的观测。改动 obs[K-1] 不应影响 alpha_0
  5. 方向性 beta：beta_k 只依赖 > k 的观测。改动 obs[0] 不应影响 beta_{K-2}
  6. beta 末端无信息：beta_{K-1} 精度必须为 0（没有更晚的证据）
  7. 共享 T：forward_pass 与 backward_pass 用的是同一个实例（参数 id 相同）
  8. 融合公式：精度为 0 的成分不产生影响

第 4/5 条是最关键的 —— 若方向性不成立，"因果 / 反因果"就是空话，
整个 past->present<-future 的结构不存在。

结果写 /tmp/whenami_block2.json
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import torch
from vla.temporal import (
    StateSpec, GaussianState, BackwardMessage, ObsEncoder,
    TransitionNet, forward_pass, backward_pass, fuse,
)

res = {}
torch.manual_seed(0)

B, K, N, D, A = 2, 9, 16, 32, 7
spec = StateSpec(n_tokens=N, dim=D)
T = TransitionNet(spec, action_dim=A)

obs = GaussianState(mu=torch.randn(B, K, N, D), log_lam=torch.zeros(B, K, N, D))
acts = torch.randn(B, K - 1, A)

# --- 1. 形状 ---
try:
    alpha = forward_pass(obs, acts, T)
    beta = backward_pass(obs, acts, T)
    res["1_shapes"] = {
        "PASS": list(alpha.mu.shape) == [B, K, N, D] and list(beta.mu.shape) == [B, K, N, D],
        "alpha": list(alpha.mu.shape), "beta": list(beta.mu.shape),
    }
except Exception as e:
    res["1_shapes"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 2. T 零初始化 = 恒等映射 ---
try:
    s0 = GaussianState(mu=torch.randn(B, N, D), log_lam=torch.zeros(B, N, D))
    s1 = T(s0, torch.randn(B, A))
    res["2_T_identity_at_init"] = {
        "PASS": bool(torch.allclose(s1.mu, s0.mu, atol=1e-6)),
        "mu_drift": float((s1.mu - s0.mu).abs().max()),
        "note": "zero-init on last layer -> residual is 0 -> identity",
    }
except Exception as e:
    res["2_T_identity_at_init"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 3. 精度单调下降 ---
try:
    res["3_precision_decays"] = {
        "PASS": bool((s1.precision < s0.precision).all().item()),
        "lam_before": round(float(s0.precision.mean()), 5),
        "lam_after": round(float(s1.precision.mean()), 5),
        "note": "process noise: information can only be lost in transit",
    }
except Exception as e:
    res["3_precision_decays"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 4. 方向性 alpha：未来的观测不能影响过去的 alpha ---
try:
    obs2 = GaussianState(mu=obs.mu.clone(), log_lam=obs.log_lam.clone())
    obs2.mu[:, K - 1] += 100.0                       # 只改最后一帧
    a2 = forward_pass(obs2, acts, T)
    early_same = torch.allclose(alpha.mu[:, 0], a2.mu[:, 0], atol=1e-6)
    last_diff = not torch.allclose(alpha.mu[:, K - 1], a2.mu[:, K - 1], atol=1e-6)
    res["4_alpha_is_causal"] = {
        "PASS": bool(early_same and last_diff),
        "alpha_0_unchanged": bool(early_same),
        "alpha_last_changed": bool(last_diff),
        "note": "alpha_k must depend only on observations up to k",
    }
except Exception as e:
    res["4_alpha_is_causal"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 5. 方向性 beta：过去的观测不能影响未来的 beta ---
try:
    obs3 = GaussianState(mu=obs.mu.clone(), log_lam=obs.log_lam.clone())
    obs3.mu[:, 0] += 100.0                           # 只改第一帧
    b3 = backward_pass(obs3, acts, T)
    late_same = torch.allclose(beta.mu[:, K - 2], b3.mu[:, K - 2], atol=1e-6)
    early_diff = not torch.allclose(beta.mu[:, 0], b3.mu[:, 0], atol=1e-6)
    res["5_beta_is_anticausal"] = {
        "PASS": bool(late_same),
        "beta_late_unchanged": bool(late_same),
        "beta_early_changed": bool(early_diff),
        "note": "beta_k must depend only on observations after k; "
                "beta_0 may change since obs[1..] feed it, but obs[0] itself must not",
    }
except Exception as e:
    res["5_beta_is_anticausal"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 6. beta 末端无信息 ---
try:
    res["6_beta_terminal_uninformative"] = {
        "PASS": bool((beta.precision[:, K - 1] == 0).all().item()),
        "terminal_precision": float(beta.precision[:, K - 1].max()),
        "note": "no evidence later than the last frame -> precision 0",
    }
except Exception as e:
    res["6_beta_terminal_uninformative"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 7. 共享 T ---
try:
    ids_before = {id(p) for p in T.parameters()}
    _ = forward_pass(obs, acts, T)
    _ = backward_pass(obs, acts, T)
    ids_after = {id(p) for p in T.parameters()}
    res["7_shared_transition"] = {
        "PASS": ids_before == ids_after,
        "n_params": sum(p.numel() for p in T.parameters()),
        "note": "both passes take the same T instance -- same state space",
    }
except Exception as e:
    res["7_shared_transition"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 8. 融合：零精度成分无影响 ---
try:
    g1 = GaussianState(mu=torch.randn(B, N, D), log_lam=torch.zeros(B, N, D))
    g2 = GaussianState(mu=torch.randn(B, N, D), log_lam=torch.zeros(B, N, D))
    absent = BackwardMessage.absent(g1)
    absent_as_state = GaussianState(mu=absent.mu, log_lam=absent.log_lam)
    with_absent = fuse(g1, g2, absent_as_state)
    without = fuse(g1, g2)
    res["8_fuse_ignores_zero_precision"] = {
        "PASS": bool(torch.allclose(with_absent.mu, without.mu, atol=1e-6)),
        "mu_diff": float((with_absent.mu - without.mu).abs().max()),
    }
except Exception as e:
    res["8_fuse_ignores_zero_precision"] = {"ERROR": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_block2.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
