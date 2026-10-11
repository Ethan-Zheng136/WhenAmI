"""[zgt] 验证 MOD-3：datasets.py / data_utils.py 的三段切分。

改动前的行为（实测基线）：
    image_primary (5,224,224,3)，img=[0]，future=[1:]
改动后必须满足：
    1. 回归：obs_offsets=None 时行为与改动前**完全一致**
    2. 9 帧：past/now/future 按 offsets 正确切分
    3. 不一致必须报错（最关键 —— 静默错位不会崩，但训练全程用错数据）

自检项（事先知道答案）：
    - 用同一条轨迹，past_pixel_values 的最后一帧（偏移 -1）
      应当等于「把 offsets 右移一格后」的当前帧。
      这里用更直接的办法：直接比对 RLDS 原始数组的对应下标，
      验证 now_idx / past_idx / future_idx 推导正确。

结果写 /tmp/whenami_mod3.json
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import numpy as np
from PIL import Image

# 用一个假的 image_transform，直接返回可比对的指纹，避免依赖真实视觉模型
class FakeTransform:
    def __call__(self, img):
        a = np.asarray(img, dtype=np.float64)
        return {"dino": float(a.mean())}          # 用均值当指纹


class FakeTokenizer:
    def __call__(self, *a, **k):
        class R: input_ids = [1, 2, 3]
        return R()


def fake_prompt_builder(name):
    class PB:
        def add_turn(self, *a, **k): pass
        def get_prompt(self): return "x"
    return PB()


from prismatic.vla.datasets.datasets import RLDSBatchTransform

res = {}

def make_batch(n_frames):
    """构造一个假的 rlds_batch，每帧均值 = 帧序号，便于验证取了哪一帧。"""
    imgs = np.stack([np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n_frames)])
    return {
        "dataset_name": "fake",
        "action": np.zeros((16, 7), dtype=np.float32),
        "action_mask": np.ones((16,), dtype=bool),
        "observation": {"image_primary": imgs},
        "task": {"language_instruction": b"do something"},
    }


def run(obs_offsets, n_frames):
    bt = RLDSBatchTransform(
        action_tokenizer=None,
        base_tokenizer=FakeTokenizer(),
        image_transform=FakeTransform(),
        prompt_builder_fn=fake_prompt_builder,
        obs_offsets=obs_offsets,
    )
    out = bt(make_batch(n_frames))
    return {
        "now_frame": float(np.asarray(out["pixel_values"]["dino"])),
        "past_frames": [float(f["dino"]) for f in out["past_pixel_values"]],
        "future_frames": [float(f["dino"]) for f in out["future_pixel_values"]],
    }


# --- 检查 1：回归。None -> 当前帧=0，未来=[1,2,3,4] ---
try:
    r = run(None, 5)
    r["PASS"] = (r["now_frame"] == 0.0
                 and r["past_frames"] == []
                 and r["future_frames"] == [1.0, 2.0, 3.0, 4.0])
    res["1_regression_none"] = r
except Exception as e:
    res["1_regression_none"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 检查 2：显式等价 offsets，结果必须与检查 1 相同 ---
try:
    r = run([0, 3, 7, 11, 15], 5)
    r["PASS"] = (r["now_frame"] == 0.0
                 and r["past_frames"] == []
                 and r["future_frames"] == [1.0, 2.0, 3.0, 4.0])
    res["2_explicit_equivalent"] = r
except Exception as e:
    res["2_explicit_equivalent"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 检查 3：9 帧。当前帧应在下标 4，past=[0,1,2,3]，future=[5,6,7,8] ---
try:
    r = run([-4, -3, -2, -1, 0, 1, 2, 3, 4], 9)
    r["PASS"] = (r["now_frame"] == 4.0
                 and r["past_frames"] == [0.0, 1.0, 2.0, 3.0]
                 and r["future_frames"] == [5.0, 6.0, 7.0, 8.0])
    res["3_nine_frames"] = r
except Exception as e:
    res["3_nine_frames"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 检查 4【关键】：offsets 长度与帧数不符，必须报错 ---
try:
    r = run([-4, -3, -2, -1, 0, 1, 2, 3, 4], 5)      # 说 9 帧，实际给 5 帧
    res["4_length_mismatch_must_raise"] = {
        "PASS": False, "BUG": "长度不符却没报错，发生了静默错位！", "got": r,
    }
except ValueError as e:
    res["4_length_mismatch_must_raise"] = {"PASS": True, "raised": str(e)[:160]}
except Exception as e:
    res["4_length_mismatch_must_raise"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

# --- 检查 5【关键】：offsets 里没有 0（无当前帧），必须报错 ---
try:
    r = run([-2, -1, 1, 2], 4)
    res["5_no_zero_must_raise"] = {"PASS": False, "BUG": "没有 anchor 帧却没报错", "got": r}
except ValueError as e:
    res["5_no_zero_must_raise"] = {"PASS": True, "raised": str(e)[:160]}
except Exception as e:
    res["5_no_zero_must_raise"] = {"PASS": False, "err": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_mod3.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
