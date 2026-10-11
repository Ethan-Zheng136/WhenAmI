"""[WHENAMI] 验证 MOD-1/MOD-2 的改动。

两个必须过的检查：
  1. 回归：不传 obs_offsets 时，行为必须与改动前**完全一致**
     （5 帧，偏移 [0,+3,+7,+11,+15]）
  2. 新功能：传 obs_offsets 时，能取到 past 侧的帧

自检项（事先知道答案的量）：
  - baseline 的 image_primary 必须是 5 帧    <- 改动前实测过
  - baseline 的 action 必须是 (16,7)         <- 改动前实测过
  - 9帧配置的 image_primary 必须是 9 帧
  - 越界的 offsets 必须**报错**而不是静默取错帧（这是最关键的一条，
    因为静默取错帧不会崩，只会让训练全程用错数据）

结果写 /tmp/whenami_mod1.json
"""
import sys, json, traceback
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import numpy as np
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds import make_interleaved_dataset

ROOT = "/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment"


def build(window_size, future_aw, obs_offsets=None):
    kw, w = get_oxe_dataset_kwargs_and_weights(
        ROOT, [("bridge_orig", 1.0)],
        load_camera_views=("primary",), load_depth=False,
        load_proprio=False, load_language=True,
    )
    tt = dict(
        window_size=window_size,
        future_action_window_size=future_aw,
        skip_unlabeled=True,
    )
    if obs_offsets is not None:
        tt["obs_offsets"] = obs_offsets
    out = make_interleaved_dataset(
        kw, w, train=True, shuffle_buffer_size=8, batch_size=None,
        traj_transform_kwargs=tt,
        frame_transform_kwargs=dict(resize_size={"primary": (224, 224)}, num_parallel_calls=2),
        traj_transform_threads=2, traj_read_threads=2,
    )
    ds = out[0] if isinstance(out, tuple) else out
    for b in ds.as_numpy_iterator():
        return {
            "image_primary": list(b["observation"]["image_primary"].shape),
            "action": list(b["action"].shape),
            "pad_mask": list(np.asarray(b["observation"]["pad_mask"]).shape),
            "pad_mask_vals": np.asarray(b["observation"]["pad_mask"]).astype(int).tolist(),
        }


res = {}

# --- 检查 1：回归。不传 obs_offsets，必须和改动前一致 ---
try:
    r = build(window_size=1, future_aw=15, obs_offsets=None)
    r["PASS"] = (r["image_primary"][0] == 5 and r["action"] == [16, 7])
    res["1_regression_default"] = r
except Exception as e:
    res["1_regression_default"] = {"ERROR": f"{type(e).__name__}: {str(e)[:300]}"}

# --- 检查 2：显式传入与原版等价的 offsets，应与检查 1 完全相同 ---
try:
    r = build(window_size=1, future_aw=15, obs_offsets=[0, 3, 7, 11, 15])
    r["PASS"] = (r["image_primary"][0] == 5)
    res["2_explicit_same_as_default"] = r
except Exception as e:
    res["2_explicit_same_as_default"] = {"ERROR": f"{type(e).__name__}: {str(e)[:300]}"}

# --- 检查 3：9 帧 past+now+future（紧凑间隔）---
# window_size 必须 >= 1+4=5 才能取到 -4
try:
    r = build(window_size=5, future_aw=15,
              obs_offsets=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
    r["PASS"] = (r["image_primary"][0] == 9)
    res["3_nine_frames_compact"] = r
except Exception as e:
    res["3_nine_frames_compact"] = {"ERROR": f"{type(e).__name__}: {str(e)[:300]}"}

# --- 检查 4：9 帧按比例间隔（镜像 WoG 的采样）---
# 取 -15 需要 window_size >= 16
try:
    r = build(window_size=16, future_aw=15,
              obs_offsets=[-15, -11, -7, -3, 0, 3, 7, 11, 15])
    r["PASS"] = (r["image_primary"][0] == 9)
    res["4_nine_frames_proportional"] = r
except Exception as e:
    res["4_nine_frames_proportional"] = {"ERROR": f"{type(e).__name__}: {str(e)[:300]}"}

# --- 检查 5【最关键】：越界必须报错，不能静默取错帧 ---
# window_size=1 时根本没有 past 列，要 -4 必须失败
try:
    r = build(window_size=1, future_aw=15, obs_offsets=[-4, 0, 4])
    res["5_out_of_range_must_raise"] = {
        "PASS": False,
        "BUG": "越界没有报错！静默取到了错误的帧", "got": r,
    }
except ValueError as e:
    res["5_out_of_range_must_raise"] = {"PASS": True, "raised": str(e)[:200]}
except Exception as e:
    res["5_out_of_range_must_raise"] = {
        "PASS": False, "note": "报错了但不是 ValueError",
        "err": f"{type(e).__name__}: {str(e)[:200]}",
    }

with open("/tmp/whenami_mod1.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
