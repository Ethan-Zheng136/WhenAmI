"""[zgt] MaskedConverge 测试。

重点验证四类性质，每一条都对应递推版被 probe 打出来的一个缺陷：

  互斥性    改动未来帧不能影响 alpha / ell —— two-filter 的
            p ∝ alpha * ell * beta 要求三者基于不重叠观测，
            递推版让 o_t 同时进入 alpha 和 beta，证据被重复计数。
  时序敏感  打乱帧序必须改变输出 —— WoG 的 QFormer 对帧序不敏感
            （cross-attention 不带位置编码），我们靠时序嵌入补上。
  可训练性  全部参数都要收到梯度并真的移动 —— 递推版的 out_proj
            零初始化让 32 个参数在线上跑 10000 步逐字节未变。
  Lambda 校准  log_lam 必须响应重建误差 —— 递推版 log_decay 的 std
            只有 2.4e-4，融合退化成固定权重，于是 beta 在 gt_grip 上
            把 alpha 已编好的信息稀释了 0.147。

结果写 /tmp/whenami_masked.json
"""
import json
import sys

import torch

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")
from vla.temporal_masked import MaskedConverge  # noqa: E402
from vla.temporal import StateSpec             # noqa: E402

res = {}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, K, P, C = 2, 9, 8, 256          # 小尺寸：只验逻辑，不测性能
N_PAST = 4
SPEC = StateSpec()


def build(eval_mode=True):
    """eval_mode：BERT 自带 dropout，做确定性对照时必须关掉，
    否则两次前向本身就不同，测不出互斥性。"""
    torch.manual_seed(0)
    m = MaskedConverge(
        patch_dim=C, spec=SPEC, n_layers_side=2, n_layers_now=1,
        hidden=64, n_heads=4, ff=128, recon_dim=8, n_patch=P, n_action=16,
    ).to(DEV)
    return m.eval() if eval_mode else m.train()


def inputs():
    torch.manual_seed(1)
    return (torch.randn(B, K * P, C, device=DEV),
            torch.randn(B, 24, 7, device=DEV))


def run(m, x, a, **kw):
    return m(x, a, n_frames=K, n_past=N_PAST, **kw)


# --- 1. 形状 ---
try:
    m, (x, a) = build(), inputs()
    cond, aux = run(m, x, a)
    res["1_shape"] = {
        "PASS": list(cond.shape) == [B, SPEC.n_tokens, SPEC.dim],
        "got": list(cond.shape),
        "expected": [B, SPEC.n_tokens, SPEC.dim],
        "note": "only the present state; DiT runs once, unlike the 9x recurrent version",
    }
except Exception as e:
    res["1_shape"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 2. 互斥性：改未来帧，alpha / ell 必须不变 ---
try:
    m, (x, a) = build(), inputs()
    xv = x.view(B, K, P, C).clone()

    with torch.no_grad():
        past0, now0, fut0, _ = m._split(x, K, N_PAST)
        a0, _ = m.q_past(past0)
        e0, _ = m.q_now(now0)

        xv[:, N_PAST + 1:] += 10.0                      # 只动未来帧
        past1, now1, fut1, _ = m._split(xv.reshape(B, K * P, C), K, N_PAST)
        a1, _ = m.q_past(past1)
        e1, _ = m.q_now(now1)
        b0, _ = m.q_future(fut0)
        b1, _ = m.q_future(fut1)

    d_alpha = (a1.mu - a0.mu).abs().max().item()
    d_ell = (e1.mu - e0.mu).abs().max().item()
    d_beta = (b1.mu - b0.mu).abs().max().item()
    res["2_future_isolated"] = {
        "PASS": d_alpha < 1e-6 and d_ell < 1e-6 and d_beta > 1e-4,
        "d_alpha": f"{d_alpha:.2e}",
        "d_ell": f"{d_ell:.2e}",
        "d_beta": f"{d_beta:.2e}",
        "note": "future must move only beta; overlap would double-count evidence",
    }
except Exception as e:
    res["2_future_isolated"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 3. 互斥性：改过去帧，ell / beta 必须不变 ---
try:
    m, (x, a) = build(), inputs()
    xv = x.view(B, K, P, C).clone()
    with torch.no_grad():
        past0, now0, fut0, _ = m._split(x, K, N_PAST)
        e0, _ = m.q_now(now0)
        b0, _ = m.q_future(fut0)
        xv[:, :N_PAST] += 10.0                          # 只动过去帧
        past1, now1, fut1, _ = m._split(xv.reshape(B, K * P, C), K, N_PAST)
        e1, _ = m.q_now(now1)
        b1, _ = m.q_future(fut1)
        a0, _ = m.q_past(past0)
        a1, _ = m.q_past(past1)
    res["3_past_isolated"] = {
        "PASS": (e1.mu - e0.mu).abs().max().item() < 1e-6
                and (b1.mu - b0.mu).abs().max().item() < 1e-6
                and (a1.mu - a0.mu).abs().max().item() > 1e-4,
        "d_ell": f"{(e1.mu - e0.mu).abs().max().item():.2e}",
        "d_beta": f"{(b1.mu - b0.mu).abs().max().item():.2e}",
        "d_alpha": f"{(a1.mu - a0.mu).abs().max().item():.2e}",
    }
except Exception as e:
    res["3_past_isolated"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 4. use_future=False 真的丢掉 beta ---
try:
    m, (x, a) = build(), inputs()
    xv = x.view(B, K, P, C).clone()
    with torch.no_grad():
        c0, _ = run(m, x, a, use_future=False)
        xv[:, N_PAST + 1:] += 10.0                      # 篡改未来
        c1, _ = run(m, xv.reshape(B, K * P, C), a, use_future=False)
        cf0, _ = run(m, x, a)
        cf1, _ = run(m, xv.reshape(B, K * P, C), a)
    d_stu = (c1 - c0).abs().max().item()
    d_tea = (cf1 - cf0).abs().max().item()
    res["4_student_no_leak"] = {
        "PASS": d_stu < 1e-6 and d_tea > 1e-8,
        "student_delta": f"{d_stu:.2e}",
        "teacher_delta": f"{d_tea:.2e}",
        "note": "student must be blind to the future; teacher must not be",
    }
except Exception as e:
    res["4_student_no_leak"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 5. 时序敏感：打乱帧序必须改变输出 ---
try:
    m, (x, a) = build(), inputs()
    xv = x.view(B, K, P, C)
    perm = torch.tensor([3, 1, 2, 0, 4, 7, 6, 5, 8], device=DEV)   # 保持当下在原位
    with torch.no_grad():
        c0, _ = run(m, x, a)
        c1, _ = run(m, xv[:, perm].reshape(B, K * P, C), a)
    d = (c1 - c0).abs().max().item()
    res["5_order_sensitive"] = {
        "PASS": d > 1e-8,
        "delta": f"{d:.2e}",
        "note": "WoG's QFormer sees frames as an unordered set; the frame "
                "embedding is what makes order matter",
    }
except Exception as e:
    res["5_order_sensitive"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 6. 可训练性：全部参数收到梯度 ---
try:
    m, (x, a) = build(eval_mode=False), inputs()
    cond, aux = run(m, x, a)
    (cond.square().mean() + aux.recon + aux.act).backward()
    named = dict(m.named_parameters())
    dead = [n for n, p in named.items()
            if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)]
    res["6_all_params_get_grad"] = {
        "PASS": len(dead) == 0,
        "n_params": len(named),
        "dead": dead[:8],
        "note": "the recurrent version had 32/32 params frozen for 10000 steps",
    }
except Exception as e:
    res["6_all_params_get_grad"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 7. 优化器步进后参数真的移动 ---
try:
    m, (x, a) = build(eval_mode=False), inputs()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        cond, aux = run(m, x, a)
        (cond.square().mean() + aux.recon + aux.act).backward()
        opt.step()
    # 只检查可训练参数：word_embeddings 被显式冻结（我们传 inputs_embeds，
    # 它永不参与前向），不动是正确行为。
    frozen = [n for n, p in m.named_parameters()
              if p.requires_grad and torch.equal(before[n], p.detach())]
    res["7_params_move"] = {"PASS": len(frozen) == 0, "frozen": frozen[:8]}
except Exception as e:
    res["7_params_move"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 8. Lambda 校准：训练后 log_lam 必须脱离初始值并分化 ---
try:
    m, (x, a) = build(eval_mode=False), inputs()
    with torch.no_grad():
        _, now0, _, _ = m._split(x, K, N_PAST)
        st0, _ = m.q_now(now0)
        init_std = st0.log_lam.std().item()

    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        cond, aux = run(m, x, a)
        (cond.square().mean() + aux.recon + aux.act).backward()
        opt.step()

    with torch.no_grad():
        _, now1, fut1, npat = m._split(x, K, N_PAST)
        past1, _, _, _ = m._split(x, K, N_PAST)
        sa, _ = m.q_past(past1)
        sb, _ = m.q_future(fut1)
    spread = (sa.log_lam.mean() - sb.log_lam.mean()).abs().item()
    grown = max(sa.log_lam.std().item(), sb.log_lam.std().item())
    res["8_lambda_calibrated"] = {
        "PASS": grown > 1e-3 and spread > 1e-3,
        "init_std": f"{init_std:.2e}",
        "trained_std": f"{grown:.2e}",
        "alpha_vs_beta_gap": f"{spread:.2e}",
        "note": "recurrent version's log_decay std stayed at 2.4e-4 -> fusion "
                "degenerated into a fixed mixing ratio",
    }
except Exception as e:
    res["8_lambda_calibrated"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 9. 融合确实是精度加权（手工核对一个数）---
try:
    m, (x, a) = build(), inputs()
    with torch.no_grad():
        past, now, fut, _ = m._split(x, K, N_PAST)
        al, _ = m.q_past(past)
        el, _ = m.q_now(now)
        be, _ = m.q_future(fut)
        lam = al.precision + el.precision + be.precision
        manual = (al.precision * al.mu + el.precision * el.mu
                  + be.precision * be.mu) / lam
        cond, _ = run(m, x, a)
        expect = m.out_proj(m.out_norm(manual))
    res["9_precision_weighted"] = {
        "PASS": torch.allclose(cond, expect, atol=1e-5),
        "max_diff": f"{(cond - expect).abs().max().item():.2e}",
        "note": "mu* = sum(Lambda_i mu_i) / sum(Lambda_i)",
    }
except Exception as e:
    res["9_precision_weighted"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 10. 起步扰动小但非零 ---
try:
    m, (x, a) = build(), inputs()
    with torch.no_grad():
        cond, _ = run(m, x, a)
    v = cond.abs().mean().item()
    res["10_init_small_nonzero"] = {
        "PASS": 1e-8 < v < 0.1,
        "abs_mean": f"{v:.2e}",
        "note": "zero would kill every upstream gradient (see temporal.py)",
    }
except Exception as e:
    res["10_init_small_nonzero"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_masked.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
