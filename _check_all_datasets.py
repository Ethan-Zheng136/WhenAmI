"""逐个验证 24 个已下载数据集能否被 WoG 的 RLDS 管线真正读取。

文件完整 ≠ 能读。TFDS 需要 dataset_info.json 的 schema 与 tfrecord 内容匹配，
且 WoG 的 configs.py 里为每个数据集写死了 image_obs_keys / state_obs_keys，
键名对不上会在训练启动后才炸——那时已经占了几百张卡。

本脚本对每个数据集构造 dataset 并取 1 个 batch，逐个记录成功/失败。
结果写 /tmp/ds_check.json（TF 的 GC 噪声会污染 stderr）。
"""
import sys, os, json, traceback
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

ROOT = "/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment"
results = {}

names = sorted(d for d in os.listdir(ROOT) if os.path.isdir(f"{ROOT}/{d}"))

from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds import make_interleaved_dataset

for name in names:
    try:
        kw, w = get_oxe_dataset_kwargs_and_weights(
            ROOT, [(name, 1.0)],
            load_camera_views=("primary",), load_depth=False,
            load_proprio=False, load_language=True,
        )
        out = make_interleaved_dataset(
            kw, w, train=True, shuffle_buffer_size=16, batch_size=None,
            traj_transform_kwargs=dict(window_size=1, future_action_window_size=15,
                                       skip_unlabeled=True, goal_relabeling_strategy="uniform"),
            frame_transform_kwargs=dict(resize_size={"primary": (224, 224)}, num_parallel_calls=2),
            traj_transform_threads=2, traj_read_threads=2,
        )
        ds = out[0] if isinstance(out, tuple) else out
        for b in ds.as_numpy_iterator():
            img = b["observation"]["image_primary"]
            act = b["action"]
            results[name] = {
                "ok": True,
                "image": list(img.shape), "action": list(act.shape),
                "lang": str(b["task"]["language_instruction"])[:50],
            }
            break
        else:
            results[name] = {"ok": False, "err": "iterator empty (0 samples)"}
    except Exception as e:
        results[name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:200]}"}
    # 每个都立刻落盘，中途崩了也有部分结果
    with open("/tmp/ds_check.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

with open("/tmp/ds_check.json", "w") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
