"""验证 WoG 的 RLDS 数据管线能否在 py3.12 + TF 2.16 上工作。

背景：集群镜像是 py312（torchengine-torch280-cu129-py312），
而 WoG 官方要求 py3.10 + TF 2.15。TF 2.15 无 py312 wheel（最低 2.16）。
本脚本验证 TF 2.16 是否能替代——这是能否在集群上训练的前提。

结果写入 /tmp/rlds_result.txt（TF 退出时的 GC 噪声会污染 stderr，故用文件交换）。
"""
import sys
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds import make_interleaved_dataset

ROOT = "/mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment"

kw, w = get_oxe_dataset_kwargs_and_weights(
    ROOT, [("bridge_orig", 1.0)],
    load_camera_views=("primary",), load_depth=False,
    load_proprio=False, load_language=True,
)

out = make_interleaved_dataset(
    kw, w, train=True, shuffle_buffer_size=64, batch_size=None,
    traj_transform_kwargs=dict(window_size=1, future_action_window_size=15,
                               skip_unlabeled=True, goal_relabeling_strategy="uniform"),
    frame_transform_kwargs=dict(resize_size={"primary": (224, 224)}, num_parallel_calls=2),
    traj_transform_threads=2, traj_read_threads=2,
)
ds = out[0] if isinstance(out, tuple) else out

lines = [f"RETURN_TYPE {type(out).__name__}"]
for b in ds.as_numpy_iterator():
    img = b["observation"]["image_primary"]
    act = b["action"]
    lines += [
        f"KEYS {sorted(b.keys())}",
        f"IMAGE {img.shape} {img.dtype}",
        f"ACTION {act.shape} {act.dtype}",
        f"LANG {str(b['task']['language_instruction'])[:60]}",
    ]
    # 自检：动作窗口必须是 16 步（window_size=1 + future 15），7 维 EEF
    assert act.shape[-2:] == (16, 7), f"动作形状异常 {act.shape}"
    assert img.shape[-3:] == (224, 224, 3), f"图像形状异常 {img.shape}"
    break

lines.append("PIPELINE_OK")
open("/tmp/rlds_result.txt", "w").write("\n".join(lines))
