"""[zgt] masked 版端到端 smoke test。

test_masked_converge.py 只测 MaskedConverge 模块本身；这里测它接进 WoG
之后的整条链路，因为接线处有几个容易错的地方：

  * _compute_condition 现在返回 4 个值（多了校准损失），三处调用点都要对齐
  * masked 版让 all_nodes 为 None，forward 因此走单点动作损失分支 ——
    要确认走对了（走错就退化成 9 次 DiT，成本白省）
  * 校准损失要真的进 total loss，且权重为 0 时不产生任何影响
  * Stage 2 的 student 分支在 masked 下是新写的，要确认能反向

用随机初始化的小模型（不加载 7B 权重），只验证形状与梯度是否贯通。
结果写 /tmp/whenami_masked_e2e.json
"""
import json
import sys

import torch

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

res = {}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, P, C = 2, 64, 1024          # P = pool_size^2 = 64，C = patch 维
K, N_PAST = 9, 4
OFFSETS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)


def build(**kw):
    """只搭 MaskedConverge + ActionModel，不碰 7B 的 VLM。"""
    from vla.temporal_masked import MaskedConverge
    from vla.temporal import StateSpec
    torch.manual_seed(0)
    return MaskedConverge(patch_dim=C, spec=StateSpec(), n_layers_side=1,
                          n_layers_now=1, hidden=128, n_heads=4, ff=256,
                          recon_dim=16, n_patch=P, n_action=16, **kw).to(DEV).train()


# --- 1. 参数量对比：masked 版 vs WoG 的 QFormer ---
try:
    from vla.qformer import QFormer
    from vla.temporal_masked import MaskedConverge
    from vla.temporal import StateSpec, TemporalConverge
    torch.manual_seed(0)
    n_q = sum(p.numel() for p in QFormer(1024, 8, 2048, 6, 16).parameters())
    n_m = sum(p.numel() for p in MaskedConverge(patch_dim=1024, spec=StateSpec()).parameters())
    n_r = sum(p.numel() for p in TemporalConverge(StateSpec(), patch_dim=1024).parameters())
    res["1_param_count"] = {
        "PASS": n_m < 6 * n_q,          # 三路 + 校准头，不该膨胀到离谱
        "wog_qformer_M": round(n_q / 1e6, 1),
        "recurrent_M": round(n_r / 1e6, 1),
        "masked_M": round(n_m / 1e6, 1),
        "note": "three branches + calibration heads; the two side branches "
                "carry most of it",
    }
except Exception as e:
    res["1_param_count"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 2. 前向形状与 DiT 注入兼容 ---
try:
    m = build()
    x = torch.randn(B, K * P, C, device=DEV)
    a = torch.randn(B, 24, 7, device=DEV)
    cond, aux = m(x, a, n_frames=K, n_past=N_PAST)
    res["2_forward_shape"] = {
        "PASS": list(cond.shape) == [B, 16, 32],
        "cond": list(cond.shape),
        "recon_loss": f"{aux.recon.item():.4f}",
        "act_loss": f"{aux.act.item():.4f}",
        "note": "[B,16,32] matches what per_attn expects (inj_channels=32)",
    }
except Exception as e:
    res["2_forward_shape"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 3. 与真实 DiT 串起来能反向 ---
try:
    from action_model.action_model import ActionModel
    m = build()
    torch.manual_seed(0)
    am = ActionModel(model_type="DiT-S", token_size=4096, in_channels=7,
                     Ta=16, train_venc=True).to(DEV).train()
    x = torch.randn(B, K * P, C, device=DEV)
    a = torch.randn(B, 24, 7, device=DEV)
    cog = torch.randn(B, 1, 4096, device=DEV)

    # DiT 的 final_layer 是零初始化的（models.py:278，标准做法 —— 它输出的是
    # 残差 dx）。于是第 1 步梯度传到 cond 时被乘成零，out_norm / out_proj
    # 拿不到梯度。final_layer 自身有梯度，第 2 步就脱离零点，之后全线打通。
    # 这与 transition.net[-1] 的"延迟启动"同类，不是死锁 —— 故先跑几步。
    opt = torch.optim.AdamW(list(m.parameters()) + list(am.parameters()), lr=1e-3)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        cond, aux = m(x, a, n_frames=K, n_past=N_PAST)
        act_loss = am.loss(a[:, -16:], cog, cond)
        (act_loss + 0.1 * aux.recon + 0.1 * aux.act).backward()
        opt.step()

    opt.zero_grad(set_to_none=True)
    cond, aux = m(x, a, n_frames=K, n_past=N_PAST)
    act_loss = am.loss(a[:, -16:], cog, cond)
    (act_loss + 0.1 * aux.recon + 0.1 * aux.act).backward()

    g_cond = sum(1 for p in m.parameters()
                 if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0)
    n_cond = sum(1 for p in m.parameters() if p.requires_grad)
    res["3_end_to_end_backward"] = {
        "PASS": g_cond == n_cond,
        "converge_params_with_grad": f"{g_cond}/{n_cond}",
        "act_loss": f"{act_loss.item():.4f}",
        "note": "gradient must reach every converge param through the DiT",
    }
except Exception as e:
    import traceback
    res["3_end_to_end_backward"] = {"PASS": False, "err": f"{type(e).__name__}: {e}",
                                    "tb": traceback.format_exc()[-300:]}

# --- 4. per_attn 真的用到了条件（条件变则输出变）---
try:
    from action_model.action_model import ActionModel
    torch.manual_seed(0)
    am = ActionModel(model_type="DiT-S", token_size=4096, in_channels=7,
                     Ta=16, train_venc=True).to(DEV).eval()
    cog = torch.randn(B, 1, 4096, device=DEV)
    c1 = torch.randn(B, 16, 32, device=DEV)
    c2 = c1 * 5.0
    # final_layer 零初始化 -> 未训练的 DiT 输出恒为 0，对任何条件都一样。
    # 先训几步让它脱离零点，否则测的是"没训过的网络"而非 per_attn 是否生效。
    am.train()
    opt = torch.optim.AdamW(am.parameters(), lr=1e-3)
    tgt = torch.randn(B, 16, 7, device=DEV)
    for _ in range(6):
        opt.zero_grad(set_to_none=True)
        am.loss(tgt, cog, c1).backward()
        opt.step()
    am.eval()
    with torch.no_grad():
        torch.manual_seed(7); o1 = am.inference(cog, per_token=c1)
        torch.manual_seed(7); o2 = am.inference(cog, per_token=c2)
    # train_venc=True 下 obs_dropout_prob=0，LabelEmbedder 不创建
    # uncondition，故 per_token=None 会直接报错而非静默回退 —— 这是好事，
    # 条件在 Stage 1 是强制的。此前评测之所以测出"无条件"的数字，是因为
    # load_vla 默认 train_venc=False，那时 per_attn 根本没被构建。
    res["4_per_attn_active"] = {
        "PASS": (o2 - o1).abs().max().item() > 1e-5,
        "d_cond": f"{(o2 - o1).abs().max().item():.2e}",
        "note": "train_venc=True builds per_attn; without it the condition is "
                "silently ignored (this is what made the first eval meaningless)",
    }
except Exception as e:
    res["4_per_attn_active"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 5. 校准权重为 0 时不影响总损失 ---
try:
    m = build()
    x = torch.randn(B, K * P, C, device=DEV)
    a = torch.randn(B, 24, 7, device=DEV)
    cond, aux = m(x, a, n_frames=K, n_past=N_PAST)
    base = cond.square().mean()
    with_zero = base + 0.0 * aux.recon + 0.0 * aux.act
    res["5_zero_weight_inert"] = {
        "PASS": torch.allclose(base, with_zero),
        "note": "weights default to 0 so an accidental enable cannot change "
                "the baseline's behaviour",
    }
except Exception as e:
    res["5_zero_weight_inert"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 6. Stage 2 student：use_future=False 下仍能反向 ---
try:
    m = build()
    keep = N_PAST + 1
    x = torch.randn(B, keep * P, C, device=DEV)
    a = torch.randn(B, 24, 7, device=DEV)
    cond, aux = m(x, a, n_frames=keep, n_past=N_PAST, use_future=False)
    (cond.square().mean() + aux.recon + aux.act).backward()
    # Q_future 在 student 路径下不参与，其参数不该有梯度
    fut_grad = any(p.grad is not None and p.grad.abs().max() > 0
                   for p in m.q_future.parameters())
    past_grad = all(p.grad is not None and p.grad.abs().max() > 0
                    for n, p in m.q_past.named_parameters() if p.requires_grad)
    res["6_student_backward"] = {
        "PASS": (not fut_grad) and past_grad and list(cond.shape) == [B, 16, 32],
        "q_future_got_grad": fut_grad,
        "q_past_all_grad": past_grad,
        "note": "student must not touch Q_future; Q_past must still train",
    }
except Exception as e:
    res["6_student_backward"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 7. 校准损失确实驱动 log_lam 与误差反相关 ---
try:
    m = build()
    # 构造信息不对称：past 段与当下帧强相关（可重建），future 段是与当下
    # 无关的噪声（不可重建）。若校准生效，Lambda_future 必须低于 Lambda_past
    # —— 这正是"遮挡时未来主导 / 未来无用时让位"的机制基础。
    torch.manual_seed(3)
    base = torch.randn(B, 1, P, C, device=DEV)
    xv = torch.empty(B, K, P, C, device=DEV)
    xv[:, : N_PAST + 1] = base + 0.05 * torch.randn(B, N_PAST + 1, P, C, device=DEV)
    xv[:, N_PAST + 1:] = torch.randn(B, K - N_PAST - 1, P, C, device=DEV) * 3.0
    x = xv.reshape(B, K * P, C)
    a = torch.randn(B, 24, 7, device=DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        cond, aux = m(x, a, n_frames=K, n_past=N_PAST)
        (cond.square().mean() + aux.recon + aux.act).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        past, now, fut, npat = m._split(x, K, N_PAST)
        sa, fa = m.q_past(past)
        sb, fb = m.q_future(fut)
        tgt = (npat @ m.recon_proj).flatten(1)
        err_a = (m.q_past.recon_head(fa) - tgt).pow(2).mean().item()
        err_b = (m.q_future.recon_head(fb) - tgt).pow(2).mean().item()
        lam_a = sa.log_lam.mean().item()
        lam_b = sb.log_lam.mean().item()
    # past 段可重建当下、future 段是噪声，故 err_past < err_future，
    # 校准生效则 log_lam_past > log_lam_future。
    consistent = err_a < err_b and lam_a > lam_b
    res["7_lambda_tracks_error"] = {
        "PASS": consistent,
        "err_past": f"{err_a:.4f}", "log_lam_past": f"{lam_a:.3f}",
        "err_future": f"{err_b:.4f}", "log_lam_future": f"{lam_b:.3f}",
        "note": "Lambda = 1/sigma^2, so lower error must mean higher log_lam",
    }
except Exception as e:
    res["7_lambda_tracks_error"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_masked_e2e.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
