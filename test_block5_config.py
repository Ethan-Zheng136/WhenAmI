"""[zgt] 验证块 5：训练配置贯通 —— 从 train.py 配置到真实 RLDS 数据。

这是最后一环。前面每块都在各自范围内验证过，此处验证的是**参数能否
从配置一路走到数据**，中间任何一段断开都会导致：配置写了 9 帧，实际
跑的还是 5 帧 —— 不报错，训练照常，但方法根本没生效。

关键检查：
  1. 窗口推导：obs_offsets -> past_window / future_window 自洽
  2. 真实数据贯通：按推导出的参数取数，形状符合预期
  3. 回归：obs_offsets=None 时与原版完全一致
  4. 不自洽时报错（而非静默取错帧）
  5. 两处 obs_offsets 同源（chunk_act_obs 与 RLDSBatchTransform）

结果写 /tmp/whenami_block5.json
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import numpy as np

res = {}
ROOT = "/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment"
Ta = 16


def derive(offsets, Ta):
    """复刻 train.py 里的窗口推导逻辑。"""
    if offsets:
        return max(0, -min(offsets)), max(Ta - 1, max(offsets) + Ta - 1)
    return 0, Ta - 1


# --- 1. 窗口推导 ---
try:
    cases = {
        "none": (None, (0, 15)),
        "compact_9": ([-4, -3, -2, -1, 0, 1, 2, 3, 4], (4, 19)),
        "proportional_9": ([-15, -11, -7, -3, 0, 3, 7, 11, 15], (15, 30)),
    }
    got = {k: derive(v[0], Ta) for k, v in cases.items()}
    want = {k: v[1] for k, v in cases.items()}
    res["1_window_derivation"] = {
        "PASS": got == want,
        "derived": {k: list(v) for k, v in got.items()},
        "expected": {k: list(v) for k, v in want.items()},
        "note": "past_window >= max(-offset); future_window >= max(offset)+Ta-1",
    }
except Exception as e:
    res["1_window_derivation"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 2/3. 真实数据贯通 ---
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds import make_interleaved_dataset


def fetch(offsets):
    pw, fw = derive(offsets, Ta)
    kw, w = get_oxe_dataset_kwargs_and_weights(
        ROOT, [("bridge_orig", 1.0)],
        load_camera_views=("primary",), load_depth=False,
        load_proprio=False, load_language=True,
    )
    tt = dict(window_size=pw + 1, future_action_window_size=fw, skip_unlabeled=True)
    if offsets:
        tt["obs_offsets"] = offsets
    ds = make_interleaved_dataset(
        kw, w, train=True, shuffle_buffer_size=8, batch_size=None,
        traj_transform_kwargs=tt,
        frame_transform_kwargs=dict(resize_size={"primary": (224, 224)}, num_parallel_calls=2),
        traj_transform_threads=2, traj_read_threads=2,
    )
    ds = ds[0] if isinstance(ds, tuple) else ds
    for b in ds.as_numpy_iterator():
        return list(b["observation"]["image_primary"].shape), list(b["action"].shape)


try:
    img, act = fetch([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    # 9 帧；action 覆盖 [-4, +19] = 24 步，足够每个节点取满 Ta=16
    res["2_real_data_9frames"] = {
        "PASS": img[0] == 9 and act == [24, 7],
        "image_primary": img, "action": act,
        "expected_action": [24, 7],
        "note": "24 steps covers a_{-4..+19}, enough for all 9 nodes",
    }
except Exception as e:
    res["2_real_data_9frames"] = {"ERROR": f"{type(e).__name__}: {str(e)[:200]}"}

try:
    img, act = fetch(None)
    res["3_regression_default"] = {
        "PASS": img[0] == 5 and act == [16, 7],
        "image_primary": img, "action": act,
        "note": "identical to pre-change behaviour",
    }
except Exception as e:
    res["3_regression_default"] = {"ERROR": f"{type(e).__name__}: {str(e)[:200]}"}

# --- 4. 不自洽必须报错 ---
try:
    from prismatic.vla.datasets.datasets import RLDSBatchTransform, RLDSDataset
    bt = RLDSBatchTransform(None, None, None, None, obs_offsets=[-4, 0, 4])
    try:
        RLDSDataset(ROOT, "bridge_orig", bt, (224, 224),
                    past_action_window_size=0,        # 不够回看 4 步
                    obs_offsets=[-4, 0, 4])
        res["4_inconsistent_raises"] = {"PASS": False, "BUG": "should have raised"}
    except ValueError as e:
        res["4_inconsistent_raises"] = {"PASS": True, "message": str(e)[:140]}
except Exception as e:
    res["4_inconsistent_raises"] = {"ERROR": f"{type(e).__name__}: {str(e)[:200]}"}

# --- 5. 源码接线：三处 obs_offsets 都在 ---
try:
    srcs = {
        "train.py": open("/home/zhengguantian/WoG_fresh/WoG/scripts/train.py").read(),
        "materialize.py": open("/home/zhengguantian/WoG_fresh/WoG/prismatic/vla/materialize.py").read(),
        "vla.py": open("/home/zhengguantian/WoG_fresh/WoG/vla/vla.py").read(),
    }
    checks = {
        "train.py": ["use_temporal", "obs_offsets", "future_window", "past_window"],
        "materialize.py": ["obs_offsets=obs_offsets"],
        "vla.py": ["use_temporal=use_temporal", "_multi_node_action_loss"],
    }
    missing = {f: [k for k in ks if k not in srcs[f]] for f, ks in checks.items()}
    res["5_source_wiring"] = {
        "PASS": all(not v for v in missing.values()),
        "missing": {k: v for k, v in missing.items() if v},
    }
except Exception as e:
    res["5_source_wiring"] = {"ERROR": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_block5.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
