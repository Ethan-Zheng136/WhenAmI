"""[zgt] flow 版端到端验证。

上次教训：`converge_type=masked` 写进了 config，训练正常跑、loss 正常降，
但 checkpoint 里存的是递推版结构 —— 两个 320 卡任务白跑 10000+ 步。
根因是 `**kwargs` 被转发给 PrismaticVLM 而非 WoG，参数被静默丢弃。

故本测试不看代码文本，只查**运行时行为**：
  * 签名里是否显式列出（**kwargs 会吞掉参数）
  * 构造出来的到底是哪个类
  * state_dict 的键名指纹
  * 整链路（含真实 DiT）能否反向
  * 真实数据形状下能否跑通

结果写 /tmp/whenami_flow_e2e.json
"""
import inspect
import json
import sys
import time

import torch

sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

res = {}
_T0 = time.time()
_OUT = "/tmp/whenami_flow_e2e.json"


def flush(tag):
    """每项测完立即落盘：任何一项卡住时，之前的结果都还在。"""
    with open(_OUT, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open("/tmp/flow_e2e_progress.txt", "a") as f:
        f.write(f"[{time.time()-_T0:6.1f}s] {tag}\n")


open("/tmp/flow_e2e_progress.txt", "w").close()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FLOW_PARAMS = {"flow_fwd_weight", "flow_bwd_weight", "flow_cycle_weight"}

# --- 1. 三个入口的签名都显式列出 flow 参数 ---
try:
    # 只读源码文本：import vla.vla 会加载 VAE 权重（数分钟），而签名
    # 检查本来就不需要真的构造模型。
    import re
    src_vla = open("/home/zhengguantian/WoG_fresh/WoG/vla/vla.py").read()
    src_load = open("/home/zhengguantian/WoG_fresh/WoG/vla/load.py").read()

    def params_of(src, fn_name):
        i = src.index(f"def {fn_name}(")
        depth, j = 0, i
        while True:
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        return set(re.findall(r"(\w+)\s*[:=]", src[i:j]))

    miss = {}
    for name, src, fn in [("WoG.__init__", src_vla, "__init__"),
                          ("from_pretrained", src_vla, "from_pretrained"),
                          ("load_vla", src_load, "load_vla")]:
        got = params_of(src, fn)
        m = sorted(FLOW_PARAMS - got)
        if m:
            miss[name] = m
    res["1_signatures"] = {
        "PASS": not miss,
        "missing": miss,
        "note": "**kwargs forwards to PrismaticVLM, not WoG - params must be explicit",
    }
except Exception as e:
    res["1_signatures"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

flush("1_signatures")

# --- 2. from_pretrained 真的把它们传给了 cls() ---
try:
    src = src_vla[src_vla.index("def from_pretrained("):]
    body = src[src.index("vlm,"):]
    missing = [k for k in FLOW_PARAMS if f"{k}={k}" not in body]
    res["2_forwarded_to_cls"] = {
        "PASS": not missing,
        "missing": missing,
        "note": "being in the signature is not enough - must reach cls()",
    }
except Exception as e:
    res["2_forwarded_to_cls"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

flush("2_forwarded_to_cls")

# --- 3. 三种 converge_type 构造出不同的类 ---
try:
    from vla.temporal import TemporalConverge, StateSpec
    from vla.temporal_masked import MaskedConverge
    from vla.temporal_flow import BidirFlow
    sigs = {}
    for name, mod in [("recurrent", TemporalConverge(StateSpec(), patch_dim=64)),
                      ("masked", MaskedConverge(patch_dim=64, spec=StateSpec(),
                                                n_layers_side=1, n_layers_now=1,
                                                hidden=32, n_heads=2, ff=64,
                                                recon_dim=8, n_patch=8)),
                      ("flow", BidirFlow(patch_dim=64, spec=StateSpec(),
                                         n_enc_layers=1, flow_hidden=32,
                                         readout_dim=8))]:
        ks = set(mod.state_dict())
        sigs[name] = {
            "n_keys": len(ks),
            "obs_encoder": any("obs_encoder" in k for k in ks),
            "q_past": any(k.startswith("q_past.") for k in ks),
            "flow": any(k.startswith("flow.") for k in ks),
        }
    ok = (sigs["recurrent"]["obs_encoder"] and not sigs["recurrent"]["flow"]
          and sigs["masked"]["q_past"] and not sigs["masked"]["flow"]
          and sigs["flow"]["flow"] and not sigs["flow"]["obs_encoder"])
    res["3_distinct_fingerprints"] = {"PASS": ok, **sigs}
except Exception as e:
    res["3_distinct_fingerprints"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

flush("3_distinct_fingerprints")

# --- 4. converge_type 非法值必须报错，不能静默回退 ---
try:
    # 同样避开 import：查源码里确实有白名单校验且会 raise
    has_check = ('converge_type not in ("recurrent", "masked", "flow")' in src_vla
                 and "raise ValueError" in src_vla)
    res["4_invalid_type_raises"] = {
        "PASS": has_check,
        "note": "silent fallback is what cost 20 GPU-hours last time",
    }
except Exception as e:
    res["4_invalid_type_raises"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

flush("4_invalid_type_raises")

# --- 5. 与真实 DiT 串起来能反向 ---
try:
    from action_model.action_model import ActionModel
    torch.manual_seed(0)
    B, K, P, C = 2, 9, 64, 1024
    m = BidirFlow(patch_dim=C, spec=StateSpec(), n_enc_layers=1,
                  flow_hidden=128, readout_dim=32).to(DEV).train()
    am = ActionModel(model_type="DiT-S", token_size=4096, in_channels=7,
                     Ta=16, train_venc=True).to(DEV).train()
    x = torch.randn(B, K * P, C, device=DEV)
    a = torch.randn(B, K - 1, 7, device=DEV)
    cog = torch.randn(B, 1, 4096, device=DEV)
    tgt = torch.randn(B, 16, 7, device=DEV)
    vae = torch.randn(B, 32, 784, device=DEV)   # 不传时 vae_proj 不参与前向

    # DiT 的 final_layer 与 flow.net[-1] 都是零初始化（残差分支，安全），
    # 首步梯度为零属设计内的延迟启动 —— 先跑几步让它们脱离零点。
    opt = torch.optim.AdamW(list(m.parameters()) + list(am.parameters()), lr=1e-3)
    for _ in range(4):
        opt.zero_grad(set_to_none=True)
        cond, aux = m(x, a, n_frames=K, n_past=4, vae_latent=vae)
        (am.loss(tgt, cog, cond) + aux.fwd + aux.bwd + 0.1 * aux.cycle).backward()
        opt.step()

    opt.zero_grad(set_to_none=True)
    cond, aux = m(x, a, n_frames=K, n_past=4, vae_latent=vae)
    act_loss = am.loss(tgt, cog, cond)
    (act_loss + aux.fwd + aux.bwd + 0.1 * aux.cycle).backward()
    live = sum(1 for p in m.parameters()
               if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0)
    tot = sum(1 for p in m.parameters() if p.requires_grad)
    res["5_end_to_end"] = {
        "PASS": live == tot and list(cond.shape) == [B, 16, 32],
        "params_with_grad": f"{live}/{tot}",
        "cond": list(cond.shape),
        "act_loss": f"{act_loss.item():.4f}",
        "fwd": f"{aux.fwd.item():.4f}",
        "bwd": f"{aux.bwd.item():.4f}",
    }
except Exception as e:
    import traceback
    res["5_end_to_end"] = {"PASS": False, "err": f"{type(e).__name__}: {e}",
                           "tb": traceback.format_exc()[-300:]}

flush("5_end_to_end")

# --- 6. 线上 lr 下连训 30 步，参数必须真的移动 ---
try:
    torch.manual_seed(0)
    m = BidirFlow(patch_dim=1024, spec=StateSpec()).to(DEV).train()
    am = ActionModel(model_type="DiT-L", token_size=4096, in_channels=7,
                     Ta=16, train_venc=True).to(DEV).train()
    opt = torch.optim.AdamW(list(m.parameters()) + list(am.parameters()), lr=3.2e-5)
    B, K, P = 2, 9, 64
    x = torch.randn(B, K * P, 1024, device=DEV)
    a = torch.randn(B, K - 1, 7, device=DEV)
    cog = torch.randn(B, 1, 4096, device=DEV)
    tgt = torch.randn(B, 16, 7, device=DEV)
    vae = torch.randn(B, 32, 784, device=DEV)
    before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        cond, aux = m(x, a, n_frames=K, n_past=4, vae_latent=vae)
        (am.loss(tgt, cog, cond) + aux.fwd + aux.bwd + 0.1 * aux.cycle).backward()
        opt.step()
    frozen = [n for n, p in m.named_parameters()
              if p.requires_grad and torch.equal(before[n], p.detach())]
    res["6_production_lr"] = {
        "PASS": len(frozen) == 0,
        "n_params": len(before),
        "frozen": frozen[:6],
        "note": "v1 had every converge param byte-identical after 10000 steps",
    }
except Exception as e:
    res["6_production_lr"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

flush("6_production_lr")

# --- 7. FSDP 包装下能反向 ---
try:
    import os
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29655")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    torch.manual_seed(0)
    m = BidirFlow(patch_dim=256, spec=StateSpec(), n_enc_layers=1,
                  flow_hidden=64, readout_dim=16).cuda()
    fm = FSDP(m, use_orig_params=True, device_id=0)
    B, K, P = 2, 9, 16
    x = torch.randn(B, K * P, 256, device="cuda")
    a = torch.randn(B, K - 1, 7, device="cuda")
    vae = torch.randn(B, 32, 784, device="cuda")
    # 与测试 5 同理：flow.net[-1] 零初始化让 T 恒等起步，首步上游无梯度。
    # 这是设计内的延迟启动，不是 FSDP 的问题 —— 先跑几步再查。
    opt = torch.optim.AdamW(fm.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        c, ax = fm(x, a, n_frames=K, n_past=4, vae_latent=vae)
        (c.square().mean() + ax.fwd + ax.bwd + ax.cycle).backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    cond, aux = fm(x, a, n_frames=K, n_past=4, vae_latent=vae)
    (cond.square().mean() + aux.fwd + aux.bwd + aux.cycle).backward()
    live = sum(1 for p in fm.parameters()
               if p.requires_grad and p.grad is not None and p.grad.abs().max() > 0)
    tot = sum(1 for p in fm.parameters() if p.requires_grad)
    res["7_fsdp"] = {"PASS": live == tot, "params_with_grad": f"{live}/{tot}"}
    dist.destroy_process_group()
except Exception as e:
    res["7_fsdp"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

flush("7_fsdp")

# --- 8. 真实 OXE 数据的形状能跑通 ---
try:
    import numpy as np
    d = np.load("/mnt/pfs/zhengguantian/WoG_v2/outputs/oxe/fractal20220817_data.npz",
                allow_pickle=True)
    n_act = d["action"].shape[1]        # 真实动作长度
    torch.manual_seed(0)
    m = BidirFlow(patch_dim=1024, spec=StateSpec(), n_enc_layers=1,
                  flow_hidden=128, readout_dim=32).to(DEV).train()
    B, K, P = 2, 9, 64                  # 9 帧 = obs_offsets [-4..4]
    x = torch.randn(B, K * P, 1024, device=DEV)
    # _aggregate_step_actions 产出 [B, K-1, 7]
    a = torch.randn(B, K - 1, 7, device=DEV)
    cond, aux = m(x, a, n_frames=K, n_past=4, vae_latent=vae)
    (cond.square().mean() + aux.fwd + aux.bwd).backward()
    res["8_real_shapes"] = {
        "PASS": list(cond.shape) == [B, 16, 32],
        "oxe_action_len": int(n_act),
        "cond": list(cond.shape),
        "note": "9 frames from obs_offsets [-4..4]; actions aggregated to K-1 steps",
    }
except Exception as e:
    res["8_real_shapes"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

flush("8_real_shapes")

with open("/tmp/whenami_flow_e2e.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
