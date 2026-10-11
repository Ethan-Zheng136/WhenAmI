"""第1步-探针：确认开启 past_action_window_size=4 后，RLDS 给出的数据布局。

为什么先探再改：
  datasets.py:48 取的是 image_primary[0]。现在 past=0，[0] 就是当前帧。
  一旦 past=K，[0] 变成最老的历史帧，当前帧挪到 [K]。
  这一行不改会**静默出错**——不报错，但训练全程用错帧。

  所以改代码之前，必须先确认：
    Q1  image_primary 的第 0 维长度 = K + 1 + H ？
    Q2  当前帧到底在第几个位置？
    Q3  action 的第 0 维含义是什么，和 past 有没有关系？
    Q4  开启 past 后，是否会引入 padding（序列开头不足 K 帧时）？

预测（写在跑之前，不许改）：
  Q1  past=4, future=15 -> window_size=5, 加上 future_action_window -> image 应为 5 帧
      （注意：future_action_window_size 影响的是 action，不一定影响 image）
  Q2  当前帧在 index=4（即 K）
  Q3  action 形状 (5+15, 7) 或 (16,7)，不确定
  Q4  会有 padding，且应该有 mask 字段标识

自检项：
  用同一条轨迹、past=0 和 past=4 各取一次，
  past=0 的 image[0] 应当等于 past=4 的 image[4]（同一个当前帧）。
  这个 assert 如果过了，说明我对"当前帧位置"的理解正确。
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import numpy as np
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds import make_interleaved_dataset

ROOT = "/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment"


def probe(past_k: int, future_aw: int = 15, seed_take: int = 1):
    """按指定 past_action_window_size 取一个 batch，报告布局。"""
    kw, w = get_oxe_dataset_kwargs_and_weights(
        ROOT, [("bridge_orig", 1.0)],
        load_camera_views=("primary",), load_depth=False,
        load_proprio=False, load_language=True,
    )
    out = make_interleaved_dataset(
        kw, w, train=True, shuffle_buffer_size=8, batch_size=None,
        traj_transform_kwargs=dict(
            window_size=past_k + 1,              # ← datasets.py:148 的公式
            future_action_window_size=future_aw,
            skip_unlabeled=True,
        ),
        frame_transform_kwargs=dict(resize_size={"primary": (224, 224)}, num_parallel_calls=2),
        traj_transform_threads=2, traj_read_threads=2,
    )
    ds = out[0] if isinstance(out, tuple) else out

    recs = []
    for i, b in enumerate(ds.as_numpy_iterator()):
        img = b["observation"]["image_primary"]
        rec = {
            "past_k": past_k,
            "image_primary": list(img.shape),
            "action": list(b["action"].shape),
            "keys": sorted(b.keys()),
            "obs_keys": sorted(b["observation"].keys()),
        }
        # padding / mask 相关字段
        for mk in ["timestep_pad_mask", "pad_mask", "action_mask", "absolute_action_mask"]:
            if mk in b:
                rec[mk] = list(np.asarray(b[mk]).shape)
            elif mk in b.get("observation", {}):
                rec[f"obs.{mk}"] = list(np.asarray(b["observation"][mk]).shape)
        # 存下每一帧的指纹，用于跨配置比对"当前帧在哪"
        rec["frame_fingerprints"] = [float(np.asarray(f, dtype=np.float64).mean()) for f in img]
        recs.append(rec)
        if i + 1 >= seed_take:
            break
    return recs


results = {}
for k in [0, 4]:
    try:
        results[f"past_{k}"] = probe(k)
    except Exception as e:
        results[f"past_{k}"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}

with open("/tmp/probe_window.json", "w") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
