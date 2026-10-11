"""[zgt] 验证块 3：TemporalConverge 汇聚 + 数值稳定性

优化风险是这一块的主要关注点 —— 从此处起模块进入梯度回路，
写错不会立刻报错，而是表现为"训练不收敛"，极难定位。
故本测试的重点不是功能，而是**四个会导致训练困难的点**：

  R1 尺度失配  s* 的数值范围若与 WoG 的 O^c 差太远，接进 per_attn 后
               DiT 收不住。实测 WoG v_proj 权重 std=0.0167，输出尺度很小。
  R2 梯度爆炸  融合含除法 mu = sum(Lambda*mu)/sum(Lambda)，分母趋零则爆炸
  R3 数值溢出  Lambda = exp(log_lam)，log_lam 无界则 exp 溢出
  R4 破坏预训练 新模块随机初始化会一上来扰动 OpenVLA 的行为

功能检查：
  1. 形状 [B, K, N, D]
  2. 零初始化 -> 初始输出恒为 0（等价于 per_attn 不起作用）
  3. use_future=False 时物理截断到当下，不泄漏未来观测
  4. 梯度可回传且有限

结果写 /tmp/whenami_block3.json
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import torch
from vla.temporal import StateSpec, TemporalConverge

res = {}
torch.manual_seed(0)

B, K, P, C, A = 2, 9, 64, 1024, 7
N_PAST = 4
spec = StateSpec()
conv = TemporalConverge(spec, patch_dim=C, action_dim=A)

feats = torch.randn(B, K * P, C)
acts = torch.randn(B, K - 1, A)

# --- 1. 形状 ---
try:
    s = conv(feats, acts, n_frames=K)
    res["1_shape"] = {
        "PASS": list(s.mu.shape) == [B, K, spec.n_tokens, spec.dim],
        "got": list(s.mu.shape),
        "expected": [B, K, spec.n_tokens, spec.dim],
    }
except Exception as e:
    res["1_shape"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 2. R4: 起步扰动小，但必须非零 ---
# 本项原先断言输出恒为 0（out_proj 零初始化）。那个断言把一个 bug 固化成了
# 规格：out_proj 是 converge 的唯一出口，输出恒零意味着梯度恒零，整个模块
# 永远学不动 —— 线上实测 32 个参数跑 10000 步逐字节未变。
# 真正该保证的是"扰动足够小以不冲击 OpenVLA 预训练行为"，而非"恰好为零"。
# 可训练性由 test_trainability.py 单独把关。
try:
    m = float(s.mu.abs().max())
    res["2_init_small_but_nonzero"] = {
        "PASS": 1e-8 < m < 0.1,
        "abs_max": m,
        "note": "small perturbation keeps OpenVLA behaviour intact; "
                "exactly zero would also zero every upstream gradient",
    }
except Exception as e:
    res["2_init_small_but_nonzero"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 3. R3: log_lam 钳位生效，Lambda 不溢出 ---
try:
    # 人为制造极端 patch 特征，逼 log_lam 走极端
    wild = torch.randn(B, K * P, C) * 1e3
    s_wild = conv(wild, acts, n_frames=K)
    lam = s_wild.precision
    res["3_precision_bounded"] = {
        "PASS": bool(torch.isfinite(lam).all().item() and lam.max() < 1e5),
        "lam_min": float(lam.min()),
        "lam_max": float(lam.max()),
        "all_finite": bool(torch.isfinite(lam).all().item()),
        "note": f"clamped to [{spec.log_lam_min}, {spec.log_lam_max}] -> "
                f"Lambda in [{torch.tensor(spec.log_lam_min).exp():.2e}, "
                f"{torch.tensor(spec.log_lam_max).exp():.2e}]",
    }
except Exception as e:
    res["3_precision_bounded"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 4. R2: 梯度有限，不爆炸 ---
try:
    conv2 = TemporalConverge(spec, patch_dim=C, action_dim=A)
    # 打破零初始化，否则梯度恒为 0 测不出问题
    torch.nn.init.normal_(conv2.out_proj.weight, std=0.02)
    f2 = torch.randn(B, K * P, C, requires_grad=True)
    out = conv2(f2, acts, n_frames=K)
    out.mu.square().mean().backward()

    grads = [p.grad for p in conv2.parameters() if p.grad is not None]
    gmax = max(float(g.abs().max()) for g in grads)
    finite = all(bool(torch.isfinite(g).all().item()) for g in grads)
    res["4_gradient_sane"] = {
        "PASS": finite and gmax < 1e4,
        "grad_absmax": gmax,
        "all_finite": finite,
        "input_grad_absmax": float(f2.grad.abs().max()),
    }
except Exception as e:
    res["4_gradient_sane"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 5. R1: 尺度与 WoG O^c 可比 ---
try:
    # out_norm 保证输入 out_proj 前是单位尺度；用 WoG 同量级权重时输出应同量级
    conv3 = TemporalConverge(spec, patch_dim=C, action_dim=A)
    torch.nn.init.normal_(conv3.out_proj.weight, std=0.0167)   # WoG v_proj 实测 std
    s3 = conv3(feats, acts, n_frames=K)
    res["5_scale_matches_wog"] = {
        "PASS": 1e-3 < float(s3.mu.std()) < 1.0,
        "s_star_std": float(s3.mu.std()),
        "wog_vproj_weight_std": 0.0167,
        "note": "LayerNorm before projection keeps s* in a range DiT can absorb",
    }
except Exception as e:
    res["5_scale_matches_wog"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 6. student 路径：物理截断，不泄漏未来 ---
try:
    s_student = conv(feats, acts, n_frames=K, use_future=False, n_past=N_PAST)
    expect_k = N_PAST + 1
    # 篡改未来帧的特征，student 输出必须完全不变
    tampered = feats.clone()
    tampered[:, (N_PAST + 1) * P:] += 1e3
    s_tampered = conv(tampered, acts, n_frames=K, use_future=False, n_past=N_PAST)
    res["6_student_no_future_leak"] = {
        "PASS": bool(s_student.mu.shape[1] == expect_k
                     and torch.allclose(s_student.mu, s_tampered.mu, atol=1e-6)),
        "student_K": int(s_student.mu.shape[1]),
        "expected_K": expect_k,
        "unchanged_after_tampering_future": bool(
            torch.allclose(s_student.mu, s_tampered.mu, atol=1e-6)
        ),
        "note": "truncating obs physically (not just zeroing beta) prevents alpha "
                "from absorbing future observations",
    }
except Exception as e:
    res["6_student_no_future_leak"] = {"ERROR": f"{type(e).__name__}: {e}"}

res["_params"] = {
    "TemporalConverge": sum(p.numel() for p in conv.parameters()),
    "WoG_QFormer_ref": "~100M (6-layer BERT, hidden 1024)",
}

with open("/tmp/whenami_block3.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)

# --- 7. VAE 上下文接入：形状正确、确实影响输出 ---
try:
    vae = torch.randn(B, 48, 784)          # 9 帧 -> VAE 时间维 3 -> 16*3=48 token
    s_no_vae = conv(feats, acts, n_frames=K)
    s_vae = conv(feats, acts, n_frames=K, vae_latent=vae)
    # 零初始化下两者输出都是 0，故比较 out_proj 之前的融合结果不可行；
    # 改为检查带 vae 时前向不报错、形状一致，并用非零投影验证其影响
    conv_v = TemporalConverge(spec, patch_dim=C, action_dim=A)
    torch.nn.init.normal_(conv_v.out_proj.weight, std=0.02)
    a1 = conv_v(feats, acts, n_frames=K).mu
    a2 = conv_v(feats, acts, n_frames=K, vae_latent=vae).mu
    res["7_vae_context"] = {
        "PASS": bool(list(s_vae.mu.shape) == [B, K, spec.n_tokens, spec.dim]
                     and not torch.allclose(a1, a2, atol=1e-6)),
        "shape_with_vae": list(s_vae.mu.shape),
        "changes_output": bool(not torch.allclose(a1, a2, atol=1e-6)),
        "note": "VAE motion context reaches the encoder and affects s*",
    }
except Exception as e:
    res["7_vae_context"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 8. student 路径必须丢弃 VAE（它由含未来的整段视频编码而来）---
try:
    s_student_novae = conv_v(feats, acts, n_frames=K, use_future=False, n_past=N_PAST).mu
    s_student_vae = conv_v(feats, acts, n_frames=K, use_future=False, n_past=N_PAST,
                           vae_latent=vae).mu
    res["8_student_drops_vae"] = {
        "PASS": bool(torch.allclose(s_student_novae, s_student_vae, atol=1e-6)),
        "identical": bool(torch.allclose(s_student_novae, s_student_vae, atol=1e-6)),
        "note": "passing vae_latent must be a no-op when use_future=False, "
                "otherwise future information leaks into the student",
    }
except Exception as e:
    res["8_student_drops_vae"] = {"ERROR": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_block3.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
