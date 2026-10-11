"""[zgt] 可训练性测试 —— 每个参数都必须真的在学。

为什么需要这一层测试：
    已有的 53 个测试全部验证【数值正确性】（因果性、融合公式、不泄漏未来），
    一个都没问过【参数能不能收到梯度】。结果是 out_proj 的零初始化让整个
    converge 的输出恒为零，梯度随之恒为零，32 个参数在线上跑了 10000 步
    逐字节未变 —— 数值全对，模型一动不动，测试全绿。

    零初始化只在【残差分支】上安全（x = x + branch(x)，主干直通）。
    用在【串联出口】上会切断整条路。WoG 的 per_attn 是前者，
    本模块的 out_proj 曾经是后者。

本测试因此检查：
  1. 前向输出非零（恒零输出 = 下游收不到任何信息）
  2. 全部可训练参数都收到非零梯度
  3. 模拟优化器步进后参数真的改变
  4. 零初始化只出现在残差/并联位置（回归防护）
  5. student 路径（Stage 2）同样可训练
  6. 起步扰动足够小，不冲击 OpenVLA 预训练行为

结果写 /tmp/whenami_trainability.json
"""
import json, sys
import torch

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")
from vla.temporal import TemporalConverge, StateSpec  # noqa: E402

res = {}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, K, P, C = 2, 9, 128, 2176


def build():
    torch.manual_seed(0)
    return TemporalConverge(patch_dim=C, spec=StateSpec()).to(DEV)


def inputs():
    torch.manual_seed(1)
    return (torch.randn(B, K * P, C, device=DEV),
            torch.randn(B, 24, 7, device=DEV))


def vae_latent():
    """FutureEnc 的全局运动上下文；不传则 vae_proj 不参与前向。"""
    torch.manual_seed(2)
    return torch.randn(B, 4, 784, device=DEV)   # vae_dim=784，见 temporal.py:180


def as_tensor(out):
    return out if torch.is_tensor(out) else out.mu


# --- 1. 前向输出非零 ---
try:
    conv, (x, a) = build(), inputs()
    out = as_tensor(conv(x, n_frames=K, actions=a, n_past=4))
    m = out.abs().mean().item()
    res["1_output_nonzero"] = {
        "PASS": m > 1e-8,
        "abs_mean": f"{m:.3e}",
        "note": "an all-zero condition carries no information to the DiT",
    }
except Exception as e:
    res["1_output_nonzero"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 2. 全部参数都收到非零梯度 ---
try:
    conv, (x, a) = build(), inputs()
    # 传 vae_latent：vae_proj 只在有全局运动上下文时参与前向，
    # 不传的话它本就不该有梯度，那是输入缺失而非参数死掉。
    as_tensor(conv(x, n_frames=K, actions=a, n_past=4,
                   vae_latent=vae_latent())).square().mean().backward()
    named = dict(conv.named_parameters())
    # 零初始化末层（transition.net[-1]）会让它之前的层首步梯度为零，但末层
    # 自身有梯度，第 2 步就脱离零点并带动上游 —— 实测 net[0] 的 |g| 从 0
    # 变为 6.2e-4 并持续增长。这属于"延迟启动"，与 out_proj 那种上下游一起
    # 死锁的情形本质不同，故这里只看首步就该有梯度的参数。
    deferred = {"transition.net.0.weight", "transition.net.0.bias",
                "transition.net.2.weight", "transition.net.2.bias",
                "transition.act_embed.weight", "transition.act_embed.bias"}
    dead = [n for n, p in named.items()
            if p.requires_grad and n not in deferred
            and (p.grad is None or p.grad.abs().max().item() == 0.0)]
    res["2_all_params_get_grad"] = {
        "PASS": len(dead) == 0,
        "n_params": len(named),
        "dead": dead,
        "deferred_by_design": sorted(deferred),
        "note": "this is exactly what the 53 numerical tests could not catch",
    }
except Exception as e:
    res["2_all_params_get_grad"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 3. 优化器步进后参数真的改变 ---
try:
    conv, (x, a) = build(), inputs()
    v = vae_latent()          # vae_proj 只在提供 vae_latent 时参与
    opt = torch.optim.AdamW(conv.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in conv.named_parameters()}
    # 跑够步数，让零初始化末层先脱离零点、再带动上游（实测第 2 步启动）
    for _ in range(6):
        opt.zero_grad(set_to_none=True)
        as_tensor(conv(x, n_frames=K, actions=a, n_past=4, vae_latent=v)).square().mean().backward()
        opt.step()
    frozen = [n for n, p in conv.named_parameters()
              if torch.equal(before[n], p.detach())]
    res["3_params_actually_move"] = {
        "PASS": len(frozen) == 0,
        "frozen": frozen,
        "note": "checkpoints showed all 32 params byte-identical across 5000 steps",
    }
except Exception as e:
    res["3_params_actually_move"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 4. 零初始化只允许在残差/并联位置（回归防护）---
try:
    conv = build()
    zero_w = [n for n, p in conv.named_parameters()
              if n.endswith("weight") and p.abs().max().item() == 0.0]
    # transition.net.-1 用在 mu = mu + delta（残差）；to_log_lam 与 to_mu 并联，
    # 不在 mu 的通路上。out_proj 是唯一出口，绝不允许为零。
    allowed = {"transition.net.4.weight", "obs_encoder.to_log_lam.weight"}
    illegal = [n for n in zero_w if n not in allowed]
    res["4_zero_init_only_on_safe_paths"] = {
        "PASS": len(illegal) == 0,
        "zero_weights": zero_w,
        "illegal": illegal,
        "note": "out_proj is the module's only exit; zero there kills every upstream grad",
    }
except Exception as e:
    res["4_zero_init_only_on_safe_paths"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 5. student 路径（Stage 2）同样可训练 ---
try:
    conv, (x, a) = build(), inputs()
    n_past = 4
    keep = n_past + 1
    xs = x.view(B, K, P, C)[:, :keep].contiguous().view(B, keep * P, C)
    out = as_tensor(conv(xs, n_frames=keep, actions=a, n_past=n_past, use_future=False))
    out.square().mean().backward()
    named = dict(conv.named_parameters())
    # student 不走 beta，但 obs_encoder / transition / out_proj 都应参与
    # net.0 同样是零初始化末层的上游，首步无梯度属于设计内的延迟启动
    core = ["out_proj.weight", "transition.net.4.weight", "obs_encoder.to_mu.weight"]
    dead = [n for n in core
            if named[n].grad is None or named[n].grad.abs().max().item() == 0.0]
    res["5_student_path_trainable"] = {
        "PASS": len(dead) == 0 and out.abs().mean().item() > 1e-8,
        "dead": dead,
        "abs_mean": f"{out.abs().mean().item():.3e}",
    }
except Exception as e:
    res["5_student_path_trainable"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 6. 起步扰动足够小 ---
try:
    conv, (x, a) = build(), inputs()
    out = as_tensor(conv(x, n_frames=K, actions=a, n_past=4))
    m = out.abs().mean().item()
    res["6_init_perturbation_small"] = {
        "PASS": 1e-8 < m < 0.1,
        "abs_mean": f"{m:.3e}",
        "note": "small enough not to disturb OpenVLA pretrained behaviour, "
                "but non-zero so gradients flow",
    }
except Exception as e:
    res["6_init_perturbation_small"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_trainability.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
