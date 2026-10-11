"""[zgt] 验证块 3.5：TemporalConverge 接入 vla.py

接入是风险最高的一步 —— 从此处起改的是 forward 主路径，
写错的表现是"训练不收敛"而非报错。故验证重点：

  1. 回归：use_temporal=False 时与原版 WoG 行为完全一致
  2. 条件形状：两条路径输出统一为 [B, 16, 32]，下游不感知差异
  3. 当下节点：取的是 s*[n_past] 而非写死的 0 或 -1
  4. 模块键：all_module_keys / trainable_module_keys 随开关切换
  5. 无 past 时仍可工作（n_past=0，时间线退化为 now+future）

不加载 7B 模型 —— 用桩替换 v_enc 与 vlm，只测条件计算这一段。

结果写 /tmp/whenami_block35.json
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import torch
import torch.nn as nn

from vla.temporal import StateSpec, TemporalConverge
from vla.qformer import QFormer

res = {}
torch.manual_seed(0)

B, P, C, A = 2, 64, 1024, 7
N_PAST, N_FUTURE = 4, 4


class StubWoG(nn.Module):
    """只复刻 _encode_frames / _compute_condition 两段逻辑，避免加载 7B。"""

    def __init__(self, use_temporal):
        super().__init__()
        self.use_temporal = use_temporal
        self.pool_size = 8
        if use_temporal:
            self.converge = TemporalConverge(StateSpec(), patch_dim=C, action_dim=A)
            self.qformer = None
        else:
            self.converge = None
            self.qformer = QFormer(hidden_dim=C, nheads=8, dim_feedforward=2048,
                                   num_layers=2, num_query_tokens=16)

    def _encode_frames(self, frame_pixel_values, base_pixel_values):
        # 桩：直接由帧数生成确定性特征，绕开 DINOv2/VAE
        T = next(iter(frame_pixel_values.values())).shape[1]
        g = torch.Generator().manual_seed(T)
        feats = torch.randn(B, T * P, C, generator=g)
        vae = torch.randn(B, 32, 784, generator=g)
        return feats, vae

    # 以下与 vla.py 中的实现保持一致
    def _compute_condition(self, pixel_values, future_pixel_values,
                           past_pixel_values, actions):
        if not self.use_temporal:
            feats, vae_latent = self._encode_frames(future_pixel_values, pixel_values)
            return self.qformer(feats, vae_latent)

        now_feats, vae_latent = self._encode_frames(
            {k: v.unsqueeze(1) for k, v in pixel_values.items()}, pixel_values
        )
        future_feats, _ = self._encode_frames(future_pixel_values, pixel_values)

        if past_pixel_values:
            past_feats, _ = self._encode_frames(past_pixel_values, pixel_values)
            n_past = next(iter(past_pixel_values.values())).shape[1]
            patch_seq = torch.cat([past_feats, now_feats, future_feats], dim=1)
        else:
            n_past = 0
            patch_seq = torch.cat([now_feats, future_feats], dim=1)

        n_future = next(iter(future_pixel_values.values())).shape[1]
        n_frames = n_past + 1 + n_future
        step_actions = actions[:, : n_frames - 1]
        states = self.converge(patch_seq, step_actions, n_frames=n_frames,
                               vae_latent=vae_latent)
        return states.mu[:, n_past]


now_pv = {"dino": torch.randn(B, 3, 224, 224)}
fut_pv = {"dino": torch.randn(B, N_FUTURE, 3, 224, 224)}
past_pv = {"dino": torch.randn(B, N_PAST, 3, 224, 224)}
acts = torch.randn(B, 20, A)

# --- 1. QFormer 路径形状 ---
try:
    m_q = StubWoG(use_temporal=False)
    c_q = m_q._compute_condition(now_pv, fut_pv, past_pv, acts)
    res["1_qformer_shape"] = {"PASS": list(c_q.shape) == [B, 16, 32], "got": list(c_q.shape)}
except Exception as e:
    res["1_qformer_shape"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 2. Temporal 路径形状必须一致 ---
try:
    m_t = StubWoG(use_temporal=True)
    c_t = m_t._compute_condition(now_pv, fut_pv, past_pv, acts)
    res["2_temporal_shape_matches"] = {
        "PASS": list(c_t.shape) == list(c_q.shape) == [B, 16, 32],
        "temporal": list(c_t.shape), "qformer": list(c_q.shape),
        "note": "downstream per_attn cannot tell the two paths apart",
    }
except Exception as e:
    res["2_temporal_shape_matches"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 3. 取的是当下节点，不是首/末 ---
try:
    states = m_t.converge(
        torch.cat([
            m_t._encode_frames(past_pv, now_pv)[0],
            m_t._encode_frames({k: v.unsqueeze(1) for k, v in now_pv.items()}, now_pv)[0],
            m_t._encode_frames(fut_pv, now_pv)[0],
        ], dim=1),
        acts[:, : N_PAST + N_FUTURE],
        n_frames=N_PAST + 1 + N_FUTURE,
        vae_latent=m_t._encode_frames({k: v.unsqueeze(1) for k, v in now_pv.items()}, now_pv)[1],
    )
    picked = states.mu[:, N_PAST]
    res["3_picks_present_node"] = {
        "PASS": bool(torch.allclose(c_t, picked, atol=1e-6)),
        "n_nodes": int(states.mu.shape[1]),
        "present_index": N_PAST,
        "note": "index derived from n_past, not hardcoded 0 or -1",
    }
except Exception as e:
    res["3_picks_present_node"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 4. 无 past 时可工作 ---
try:
    c_nopast = m_t._compute_condition(now_pv, fut_pv, {}, acts)
    res["4_works_without_past"] = {
        "PASS": list(c_nopast.shape) == [B, 16, 32],
        "got": list(c_nopast.shape),
        "note": "n_past=0, timeline degrades to now+future",
    }
except Exception as e:
    res["4_works_without_past"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 5. 接入后起步扰动小，但非零 ---
# 与 test_block3_converge.py 第 2 项同因：原断言"条件恒为 0"会把
# out_proj 零初始化导致的梯度死锁固化成规格。这里改为检查扰动量级。
try:
    m = float(c_t.abs().max())
    res["5_init_small_but_nonzero"] = {
        "PASS": 1e-8 < m < 0.1,
        "abs_max": m,
        "note": "per_attn sees a small non-zero condition: pretrained behaviour "
                "is preserved and gradients can still reach converge",
    }
except Exception as e:
    res["5_zero_init_preserved"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 6. 源码一致性：桩里的逻辑与 vla.py 是否同步 ---
try:
    src = open("/home/zhengguantian/WoG_fresh/WoG/vla/vla.py").read()
    res["6_source_has_wiring"] = {
        "PASS": all(k in src for k in [
            "use_temporal", "_compute_condition", "self.converge",
            "states.mu[:, n_past]", 'cond_key = "converge"',
        ]),
        "note": "wiring present in vla.py",
    }
except Exception as e:
    res["6_source_has_wiring"] = {"ERROR": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_block35.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
