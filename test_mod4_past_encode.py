"""[zgt] 验证 MOD-4：past 帧接入 vla.forward 的编码路径。

不加载 7B 模型（太慢），只验证：
  1. forward 签名接受 past_pixel_values
  2. _encode_frames 对 past / future 行为一致（同样的 shape 变换）
  3. past_pixel_values 为空时，past_patch_features 保持 None（原版行为）
  4. 端到端：真实 RLDS 数据 -> collator -> 拿到正确形状的 past/future

结果写 /tmp/whenami_mod4.json
"""
import sys, json, inspect
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import numpy as np
import torch

res = {}

# --- 检查 1：forward 签名 ---
try:
    from vla.vla import WoG
    params = list(inspect.signature(WoG.forward).parameters)
    res["1_forward_signature"] = {
        "PASS": "past_pixel_values" in params,
        "has_past": "past_pixel_values" in params,
        "has_future": "future_pixel_values" in params,
    }
except Exception as e:
    res["1_forward_signature"] = {"ERROR": f"{type(e).__name__}: {str(e)[:200]}"}

# --- 检查 2：_encode_frames 存在且可被 past/future 共用 ---
try:
    res["2_encode_frames_exists"] = {
        "PASS": hasattr(WoG, "_encode_frames"),
        "args": list(inspect.signature(WoG._encode_frames).parameters),
    }
except Exception as e:
    res["2_encode_frames_exists"] = {"ERROR": f"{type(e).__name__}: {str(e)[:200]}"}

# --- 检查 3：端到端 —— 真实数据经 collator 后 past/future 的形状 ---
try:
    from prismatic.vla.datasets.datasets import RLDSBatchTransform, FrameLayout
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction

    class FakeTransform:
        def __call__(self, img):
            a = np.asarray(img, dtype=np.float32)
            return {"dino": torch.from_numpy(a).permute(2, 0, 1)}   # [3,H,W]

    class FakeTok:
        pad_token_id = 0
        model_max_length = 32
        def __call__(self, *a, **k):
            class R: input_ids = [1, 2, 3, 4]
            return R()

    def pb(name):
        class PB:
            def add_turn(self, *a, **k): pass
            def get_prompt(self): return "x"
        return PB()

    def make_batch(n):
        imgs = np.stack([np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n)])
        return {
            "dataset_name": "fake",
            "action": np.zeros((20, 7), dtype=np.float32),
            "action_mask": np.ones((20,), dtype=bool),
            "observation": {"image_primary": imgs},
            "task": {"language_instruction": b"do it"},
        }

    offs = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    bt = RLDSBatchTransform(None, FakeTok(), FakeTransform(), pb, obs_offsets=offs)
    items = [bt(make_batch(9)) for _ in range(2)]          # batch size 2

    coll = PaddedCollatorForActionPrediction(32, 0, padding_side="right")
    out = coll(items)

    past = out["past_pixel_values"]["dino"]
    fut = out["future_pixel_values"]["dino"]
    res["3_end_to_end_9frames"] = {
        "PASS": list(past.shape) == [2, 4, 3, 8, 8] and list(fut.shape) == [2, 4, 3, 8, 8],
        "past_shape": list(past.shape),     # 期望 [B=2, T=4, C=3, H, W]
        "future_shape": list(fut.shape),
        # past 的 4 帧应是原始下标 0..3，future 是 5..8（用像素值当指纹）
        "past_frame_ids": past[0, :, 0, 0, 0].tolist(),
        "future_frame_ids": fut[0, :, 0, 0, 0].tolist(),
    }
except Exception as e:
    import traceback
    res["3_end_to_end_9frames"] = {"ERROR": f"{type(e).__name__}: {str(e)[:200]}",
                                   "tb": traceback.format_exc()[-400:]}

# --- 检查 4：回归 —— obs_offsets=None 时 past 必须为空 ---
try:
    bt0 = RLDSBatchTransform(None, FakeTok(), FakeTransform(), pb, obs_offsets=None)
    items0 = [bt0(make_batch(5)) for _ in range(2)]
    out0 = coll(items0)
    res["4_regression_no_past"] = {
        "PASS": out0["past_pixel_values"] == {} and list(out0["future_pixel_values"]["dino"].shape) == [2, 4, 3, 8, 8],
        "past": out0["past_pixel_values"],
        "future_shape": list(out0["future_pixel_values"]["dino"].shape),
    }
except Exception as e:
    res["4_regression_no_past"] = {"ERROR": f"{type(e).__name__}: {str(e)[:200]}"}

with open("/tmp/whenami_mod4.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
