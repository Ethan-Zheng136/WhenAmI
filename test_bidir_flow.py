"""[zgt] BidirFlow 测试。

每一项都对应一个已经踩过的坑或一个必须成立的设计性质：

  可训练性    v1 的 out_proj 零初始化让 32 个参数线上跑 10000 步逐字节未变。
              这里查每个参数都收到梯度、且优化器步进后真的移动。
  双向流动    同一个 T 正反都用 —— 查方向真的改变了行为（否则 dir 条件是摆设）。
  Lambda 自适应
              v1/v2 让 to_log_lam 猜自己有多确定，实测跨样本 std 仅 0.14%。
              这里 Lambda 来自推演误差，查它是否随输入真的变化。
  不泄漏      use_future=False 时篡改未来帧不能影响输出。
  往返一致    方向 token 不保证严格可逆，cycle loss 是显式检验。

结果写 /tmp/whenami_flow.json
"""
import json
import sys

import torch

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")
from vla.temporal_flow import BidirFlow, FlowOperator   # noqa: E402
from vla.temporal import StateSpec                      # noqa: E402

res = {}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, K, P, C = 4, 9, 16, 128
N_PAST = 4
SPEC = StateSpec()


def build(train=True):
    torch.manual_seed(0)
    m = BidirFlow(patch_dim=C, spec=SPEC, n_enc_layers=1, n_cross_layers=1,
                  enc_hidden=64, flow_hidden=128, readout_dim=16).to(DEV)
    return m.train() if train else m.eval()


def inputs():
    torch.manual_seed(1)
    return (torch.randn(B, K * P, C, device=DEV),
            torch.randn(B, K - 1, 7, device=DEV))


def run(m, x, a, **kw):
    return m(x, a, n_frames=K, n_past=N_PAST, **kw)


# --- 1. 形状 ---
try:
    m, (x, a) = build(False), inputs()
    with torch.no_grad():
        cond, aux = run(m, x, a)
    res["1_shape"] = {
        "PASS": list(cond.shape) == [B, SPEC.n_tokens, SPEC.dim],
        "cond": list(cond.shape),
        "fwd": f"{aux.fwd.item():.4f}",
        "bwd": f"{aux.bwd.item():.4f}",
        "cycle": f"{aux.cycle.item():.3e}",
    }
except Exception as e:
    import traceback
    res["1_shape"] = {"PASS": False, "err": f"{type(e).__name__}: {e}",
                      "tb": traceback.format_exc()[-300:]}

# --- 2. 方向真的改变行为（dir 条件不是摆设）---
try:
    torch.manual_seed(0)
    T = FlowOperator(SPEC.dim, 7, 128).to(DEV).eval()
    # 末层零初始化 -> 恒等起步，先扰动一下才能看出方向差异
    with torch.no_grad():
        T.net[-1].weight.normal_(0, 0.05)
    s = torch.randn(B, SPEC.n_tokens, SPEC.dim, device=DEV)
    act = torch.randn(B, 7, device=DEV)
    dt = torch.full((B,), 4.0, device=DEV)
    with torch.no_grad():
        f = T(s, act, dt, True)
        b = T(s, act, dt, False)
    d = (f - b).abs().max().item()
    res["2_direction_matters"] = {
        "PASS": d > 1e-4,
        "delta": f"{d:.3e}",
        "note": "same params, direction as condition - must behave differently",
    }
except Exception as e:
    res["2_direction_matters"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 3. dt 真的改变行为 ---
try:
    torch.manual_seed(0)
    T = FlowOperator(SPEC.dim, 7, 128).to(DEV).eval()
    with torch.no_grad():
        T.net[-1].weight.normal_(0, 0.05)
    s = torch.randn(B, SPEC.n_tokens, SPEC.dim, device=DEV)
    act = torch.randn(B, 7, device=DEV)
    with torch.no_grad():
        s1 = T(s, act, torch.full((B,), 1.0, device=DEV), True)
        s8 = T(s, act, torch.full((B,), 8.0, device=DEV), True)
    d = (s1 - s8).abs().max().item()
    res["3_dt_matters"] = {
        "PASS": d > 1e-4,
        "delta": f"{d:.3e}",
        "note": "OXE node spacing is 3-4 steps, not uniform - dt must condition T",
    }
except Exception as e:
    res["3_dt_matters"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 4. 不泄漏：use_future=False 时篡改未来不影响输出 ---
try:
    m, (x, a) = build(False), inputs()
    xv = x.view(B, K, P, C).clone()
    with torch.no_grad():
        c0, _ = run(m, x, a, use_future=False)
        xv[:, N_PAST + 1:] += 10.0
        c1, _ = run(m, xv.reshape(B, K * P, C), a, use_future=False)
        t0, _ = run(m, x, a)
        t1, _ = run(m, xv.reshape(B, K * P, C), a)
    d_stu = (c1 - c0).abs().max().item()
    d_tea = (t1 - t0).abs().max().item()
    res["4_no_future_leak"] = {
        "PASS": d_stu < 1e-5 and d_tea > 1e-6,
        "student_delta": f"{d_stu:.2e}",
        "teacher_delta": f"{d_tea:.2e}",
        "note": "deployment has no future; teacher must still see it",
    }
except Exception as e:
    res["4_no_future_leak"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 5. 可训练性：全部参数收到梯度 ---
try:
    m, (x, a) = build(), inputs()
    v = torch.randn(B, 32, 784, device=DEV)   # 不传 VAE 时 vae_proj 不参与前向
    # flow.net[-1] 零初始化 -> T 恒等起步，首步 flow 上游拿不到梯度。
    # 末层自身有梯度，第 2 步脱离零点后全线打通 —— 这是设计内的"延迟
    # 启动"，与 v1 的 out_proj 那种上下游一起死锁本质不同。故先跑几步。
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        cond, aux = run(m, x, a, vae_latent=v)
        (cond.square().mean() + aux.fwd + aux.bwd + aux.cycle).backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    cond, aux = run(m, x, a, vae_latent=v)
    (cond.square().mean() + aux.fwd + aux.bwd + aux.cycle).backward()
    named = dict(m.named_parameters())
    dead = [n for n, p in named.items()
            if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0.0)]
    res["5_all_params_grad"] = {
        "PASS": len(dead) == 0,
        "n_params": len(named),
        "dead": dead[:8],
        "note": "v1 had 32/32 params frozen for 10000 steps (zero-init out_proj)",
    }
except Exception as e:
    res["5_all_params_grad"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 6. 优化器步进后参数真的移动 ---
try:
    m, (x, a) = build(), inputs()
    v = torch.randn(B, 32, 784, device=DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    for _ in range(6):
        opt.zero_grad(set_to_none=True)
        cond, aux = run(m, x, a, vae_latent=v)
        (cond.square().mean() + aux.fwd + aux.bwd + aux.cycle).backward()
        opt.step()
    frozen = [n for n, p in m.named_parameters()
              if p.requires_grad and torch.equal(before[n], p.detach())]
    res["6_params_move"] = {"PASS": len(frozen) == 0, "frozen": frozen[:8]}
except Exception as e:
    res["6_params_move"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 7. Lambda 随输入自适应（v1/v2 的致命缺陷）---
try:
    m = build(False)
    torch.manual_seed(2)
    # 构造两组输入：一组各帧高度一致（易推演），一组各帧互不相关（难推演）
    easy = torch.randn(B, 1, P, C, device=DEV).expand(B, K, P, C) \
        + 0.01 * torch.randn(B, K, P, C, device=DEV)
    hard = torch.randn(B, K, P, C, device=DEV) * 3.0
    a = torch.randn(B, K - 1, 7, device=DEV)
    with torch.no_grad():
        _, aux_e = run(m, easy.reshape(B, K * P, C), a)
        _, aux_h = run(m, hard.reshape(B, K * P, C), a)
    # 推演误差应当在 hard 上显著更大 —— Lambda = 1/err^2 于是更小
    ratio = aux_h.fwd.item() / (aux_e.fwd.item() + 1e-12)
    res["7_lambda_adapts"] = {
        "PASS": ratio > 2.0,
        "err_easy": f"{aux_e.fwd.item():.4f}",
        "err_hard": f"{aux_h.fwd.item():.4f}",
        "ratio": f"{ratio:.2f}x",
        "note": "Lambda comes from rollout error, not from a head guessing - "
                "v1/v2 had cross-sample std of only 0.14%",
    }
except Exception as e:
    res["7_lambda_adapts"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 8. 往返一致可被优化（方向 token 学出的可逆性是真的）---
try:
    m, (x, a) = build(), inputs()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    with torch.no_grad():
        _, aux0 = run(m, x, a)
    c0 = aux0.cycle.item()
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        cond, aux = run(m, x, a)
        (aux.fwd + aux.bwd + 10.0 * aux.cycle).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        _, aux1 = run(m, x, a)
    # T 零初始化时是恒等映射，cycle 起点恒为 0 —— 拿它当基线没有意义。
    # 真正该保证的是：T 脱离恒等之后，往返误差仍被压在可忽略的量级。
    # 判据取相对量：cycle 相对于状态本身的尺度。
    # 分母必须是【状态本身】的尺度，不能用 cond —— out_proj 是 std=1e-3
    # 的小初始化，cond 的量级天然比状态小三个数量级，拿它当分母会让任何
    # cycle 都显得巨大（实测同一次训练：用 cond 得 10.5，用状态得 0.007）。
    with torch.no_grad():
        xs = m._split_frames(x, K, N_PAST)
        scale = m.encoder(xs).mu[:, N_PAST].pow(2).mean().item()
    rel = aux1.cycle.item() / (scale + 1e-12)
    res["8_cycle_trainable"] = {
        "PASS": rel < 0.05,
        "cycle_before": f"{c0:.3e}",
        "cycle_after": f"{aux1.cycle.item():.3e}",
        "relative_to_state": f"{rel:.3f}",
        "note": "direction-as-condition does not guarantee invertibility; "
                "cycle loss must be able to enforce it",
    }
except Exception as e:
    res["8_cycle_trainable"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 9. 起步扰动小但非零 ---
try:
    m, (x, a) = build(False), inputs()
    with torch.no_grad():
        cond, _ = run(m, x, a)
    v = cond.abs().mean().item()
    res["9_init_small_nonzero"] = {
        "PASS": 1e-8 < v < 0.1,
        "abs_mean": f"{v:.2e}",
        "note": "zero would kill every upstream gradient (v1's mistake)",
    }
except Exception as e:
    res["9_init_small_nonzero"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 10. 零初始化只在残差分支上 ---
try:
    m = build(False)
    zero_w = [n for n, p in m.named_parameters()
              if n.endswith("weight") and p.abs().max().item() == 0.0]
    # flow.net[-1] 在 s + net(...) 的残差分支；to_log_lam 与 to_mu 并联
    allowed = {"flow.net.4.weight", "encoder.to_log_lam.weight"}
    illegal = [n for n in zero_w if n not in allowed]
    res["10_zero_init_safe"] = {
        "PASS": len(illegal) == 0,
        "zero_weights": zero_w,
        "illegal": illegal,
    }
except Exception as e:
    res["10_zero_init_safe"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}



# --- 11. VAE 输入真的被用上（WoG 有、我们此前三版都丢了）---
try:
    m, (x, a) = build(False), inputs()
    v = torch.randn(B, 32, 784, device=DEV)
    with torch.no_grad():
        c0, _ = run(m, x, a)                    # 不给 VAE
        c1, _ = run(m, x, a, vae_latent=v)      # 给 VAE
        c2, _ = run(m, x, a, vae_latent=v * 5)  # 换一个 VAE
    d_use = (c1 - c0).abs().max().item()
    d_var = (c2 - c1).abs().max().item()
    res["11_vae_used"] = {
        "PASS": d_use > 1e-6 and d_var > 1e-6,
        "with_vs_without": f"{d_use:.2e}",
        "vae_sensitivity": f"{d_var:.2e}",
        "note": "Wan2.1 video VAE encodes how things move - WoG feeds it to "
                "the QFormer, our first three versions silently dropped it",
    }
except Exception as e:
    res["11_vae_used"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 12. 帧间交互真的发生（早先逐帧独立编码）---
try:
    m, (x, a) = build(False), inputs()
    xv = x.view(B, K, P, C).clone()
    with torch.no_grad():
        st0 = m.encoder(m._split_frames(x, K, N_PAST))
        xv[:, 0] += 10.0                        # 只改第 0 帧
        st1 = m.encoder(m._split_frames(xv.reshape(B, K * P, C), K, N_PAST))
    # 改第 0 帧后，最后一帧的编码也该变 —— 那才叫有交互
    d_far = (st1.mu[:, -1] - st0.mu[:, -1]).abs().max().item()
    d_own = (st1.mu[:, 0] - st0.mu[:, 0]).abs().max().item()
    res["12_cross_frame"] = {
        "PASS": d_far > 1e-6 and d_own > 1e-6,
        "own_frame_delta": f"{d_own:.2e}",
        "far_frame_delta": f"{d_far:.2e}",
        "note": "WoG flattens 5 frames into one cross-attention; encoding "
                "frames in isolation compresses 64x1024 to 16x32 too early",
    }
except Exception as e:
    res["12_cross_frame"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 13. 帧间交互不构成未来泄漏 ---
try:
    m, (x, a) = build(False), inputs()
    xv = x.view(B, K, P, C).clone()
    with torch.no_grad():
        c0, _ = run(m, x, a, use_future=False)
        xv[:, N_PAST + 1:] += 10.0
        c1, _ = run(m, xv.reshape(B, K * P, C), a, use_future=False)
    d = (c1 - c0).abs().max().item()
    res["13_no_leak_with_cross"] = {
        "PASS": d < 1e-5,
        "student_delta": f"{d:.2e}",
        "note": "deployment path truncates frames upstream, so cross-frame "
                "attention never sees the future there",
    }
except Exception as e:
    res["13_no_leak_with_cross"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_flow.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
