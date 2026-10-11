"""[zgt] PPU 冒烟：use_temporal=True 端到端跑一步真实 forward。

前面 47 个单元测试都用桩替换了 v_enc / vlm。本脚本加载真实的
DINOv2 + Wan VAE + 7B VLM，在 PPU 上跑一次完整前向 + 反向，
验证：

  1. 模型能构造（TemporalConverge 正确替换 QFormer）
  2. 真实视觉编码 -> 时间线 -> 多节点 loss 全链路通
  3. loss 有限、梯度有限（数值稳定性在真实尺度下仍成立）
  4. 显存可控

用假 batch 而非真实 RLDS —— 数据侧已在 block5 验证过，
这里只测模型侧。

用法（PPU）:
    cd WoG && PYTHONPATH=... python3 smoke_temporal_ppu.py
"""
import sys, json, time
sys.path.insert(0, "/home/zhengguantian/WoG_fresh")
import stub_tf
stub_tf.patch(verbose=False)
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import torch

OUT = "/tmp/whenami_smoke.json"
res = {}


def log(k, **kw):
    res[k] = kw
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


B, N_PAST, N_FUT, Ta, A = 1, 4, 4, 16, 7
K = N_PAST + 1 + N_FUT

try:
    t0 = time.time()
    from vla import load_vla

    CKPT = ("/mnt/pfs/zhengguantian/WoG_v2/ckpts/openvla-7b-prismatic/"
            "checkpoints/step-295000-epoch-40-loss=0.2200.pt")
    vla = load_vla(CKPT, load_for_training=True, action_model_type="DiT-L",
                   future_action_window_size=Ta - 1, train_venc=True,
                   use_temporal=True)
    log("1_construct", PASS=True, seconds=round(time.time() - t0, 1),
        cond_module=type(vla.converge).__name__,
        qformer_is_none=vla.qformer is None,
        converge_params=sum(p.numel() for p in vla.converge.parameters()))
except Exception as e:
    import traceback
    log("1_construct", PASS=False, error=f"{type(e).__name__}: {str(e)[:300]}",
        tb=traceback.format_exc()[-600:])
    sys.exit(1)

try:
    dev = "cuda"
    vla = vla.to(torch.float32).to(dev).train()
    torch.cuda.reset_peak_memory_stats()

    tok = vla.llm_backbone.get_tokenizer()
    ids = tok("What action should the robot take to pick up the cup?",
              return_tensors="pt").input_ids.to(dev)
    mask = torch.ones_like(ids)

    def pv(n=None):
        shape = (B, 3, 224, 224) if n is None else (B, n, 3, 224, 224)
        return {"dino": torch.randn(*shape, device=dev),
                "siglip": torch.randn(*shape, device=dev)}

    # action 需覆盖 [-n_past, +19] -> 24 步
    acts = torch.randn(B, N_PAST + Ta + N_FUT, A, device=dev)

    t0 = time.time()
    loss, _, act_loss, obs_loss = vla(
        input_ids=ids, attention_mask=mask,
        pixel_values=pv(), future_pixel_values=pv(N_FUT),
        past_pixel_values=pv(N_PAST),
        actions=acts, labels=ids,
        output_hidden_states=True, repeated_diffusion_steps=2,
    )
    log("2_forward", PASS=bool(torch.isfinite(loss).all()),
        seconds=round(time.time() - t0, 1),
        loss=float(loss), act_loss=float(act_loss), obs_loss=float(obs_loss),
        mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
except Exception as e:
    import traceback
    log("2_forward", PASS=False, error=f"{type(e).__name__}: {str(e)[:300]}",
        tb=traceback.format_exc()[-800:])
    sys.exit(1)

try:
    t0 = time.time()
    loss.backward()
    grads = [p.grad for p in vla.converge.parameters() if p.grad is not None]
    gmax = max(float(g.abs().max()) for g in grads) if grads else 0.0
    log("3_backward", PASS=bool(grads) and all(
            torch.isfinite(g).all() for g in grads) and gmax < 1e4,
        seconds=round(time.time() - t0, 1),
        n_grad_tensors=len(grads), grad_absmax=gmax,
        mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
except Exception as e:
    import traceback
    log("3_backward", PASS=False, error=f"{type(e).__name__}: {str(e)[:300]}",
        tb=traceback.format_exc()[-600:])
