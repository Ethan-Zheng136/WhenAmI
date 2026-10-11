"""[zgt] 验证块 6：动作聚合修复 + Stage 2 双路径蒸馏

两件事：

A. 动作聚合（bug 修复）
   原实现 step_actions = actions[:, :n_frames-1] 假设相邻节点间隔 1 步。
   但 WoG 的采样是按动作窗口 1/4 均匀取（偏移 [0,3,7,11,15]，间隔 3~4 步），
   我们镜像该策略后 past 侧同样是 3~4 步间隔 —— 一个动作驱动不了 4 步转移。
   修复：把区间内的动作求和作为等效位移（delta EEF 可加），gripper 取末值。

B. Stage 2 双路径
   同一 batch 再跑一次无 beta 的 student 路径，逐节点对齐 teacher。
   关键是 student 必须物理截断观测（不只是置零 beta），否则 alpha
   仍会吸收未来帧 —— 那是泄漏而非特权撤除。

结果写 /tmp/whenami_block6.json
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import torch
import torch.nn as nn

res = {}
torch.manual_seed(0)
B, A, Ta = 2, 7, 16


def aggregate(actions, offsets, n_past):
    """复刻 vla.py 的 _aggregate_step_actions。"""
    if offsets is None:
        return actions[:, : actions.shape[1] - 1]
    offsets = list(offsets)
    steps = []
    for k in range(len(offsets) - 1):
        lo, hi = offsets[k] + n_past, offsets[k + 1] + n_past
        seg = actions[:, lo:hi]
        agg = seg.sum(dim=1)
        agg = torch.cat([agg[..., :-1], seg[:, -1, -1:]], dim=-1)
        steps.append(agg)
    return torch.stack(steps, dim=1)


# --- 1. 紧凑 offsets：间隔 1，聚合应等价于直接切片 ---
try:
    offs = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    n_past = 4
    acts = torch.randn(B, 24, A)
    got = aggregate(acts, offs, n_past)
    want = acts[:, :8]     # 间隔 1 时每段只有一个动作
    res["1_compact_equals_slice"] = {
        "PASS": bool(torch.allclose(got, want, atol=1e-6)),
        "shape": list(got.shape),
        "note": "spacing 1 -> each segment holds exactly one action",
    }
except Exception as e:
    res["1_compact_equals_slice"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 2. 按比例 offsets：间隔 3~4，必须聚合而非切片 ---
try:
    offs_p = [-15, -11, -7, -3, 0, 3, 7, 11, 15]
    n_past_p = 15
    acts_p = torch.zeros(B, 46, A)
    for i in range(46):
        acts_p[:, i, 0] = 1.0            # 每步 dx=1，便于验证求和
    got = aggregate(acts_p, offs_p, n_past_p)
    # 节点 k -> k+1 的位移应等于间隔步数
    spacing = [offs_p[i + 1] - offs_p[i] for i in range(8)]
    dx = got[0, :, 0].tolist()
    res["2_proportional_aggregates"] = {
        "PASS": [round(v) for v in dx] == spacing,
        "spacing": spacing,
        "aggregated_dx": [round(v, 1) for v in dx],
        "note": "sum of per-step deltas must equal the gap length",
    }
except Exception as e:
    res["2_proportional_aggregates"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 3. gripper 取末值而非求和 ---
try:
    acts_g = torch.zeros(B, 46, A)
    acts_g[:, :, -1] = 1.0               # 每步 gripper=1
    got = aggregate(acts_g, offs_p, n_past_p)
    res["3_gripper_takes_last"] = {
        "PASS": bool(torch.allclose(got[..., -1], torch.ones_like(got[..., -1]))),
        "gripper_values": got[0, :, -1].tolist(),
        "note": "gripper is absolute, not incremental -> take last, not sum",
    }
except Exception as e:
    res["3_gripper_takes_last"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 4. Stage 2：student 时间线只到当下 ---
try:
    from vla.temporal import StateSpec, TemporalConverge
    conv = TemporalConverge(StateSpec(), patch_dim=1024, action_dim=A)
    # 打破 out_proj 的零初始化：否则任何输入都输出 0，
    # "篡改后不变"会成为必然，测不出泄漏与否
    torch.nn.init.normal_(conv.out_proj.weight, std=0.02)
    K, P = 9, 64
    feats = torch.randn(B, K * P, 1024)
    step_a = torch.randn(B, K - 1, A)
    teacher = conv(feats, step_a, n_frames=K).mu
    student = conv(feats, step_a, n_frames=K, use_future=False, n_past=4).mu
    res["4_student_truncated"] = {
        "PASS": teacher.shape[1] == K and student.shape[1] == 5,
        "teacher_K": int(teacher.shape[1]),
        "student_K": int(student.shape[1]),
        "note": "student covers [0, n_past] only",
    }
except Exception as e:
    res["4_student_truncated"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 5. Stage 2：篡改未来帧，student 必须不变 ---
try:
    tampered = feats.clone()
    tampered[:, 5 * P:] += 1e3
    student2 = conv(tampered, step_a, n_frames=K, use_future=False, n_past=4).mu
    teacher2 = conv(tampered, step_a, n_frames=K).mu
    res["5_student_no_leak"] = {
        "PASS": bool(torch.allclose(student, student2, atol=1e-6)
                     and not torch.allclose(teacher, teacher2, atol=1e-6)),
        "student_unchanged": bool(torch.allclose(student, student2, atol=1e-6)),
        "teacher_changed": bool(not torch.allclose(teacher, teacher2, atol=1e-6)),
        "note": "teacher sees the tampering, student must not",
    }
except Exception as e:
    res["5_student_no_leak"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 6. 源码接线 ---
try:
    src = open("/home/zhengguantian/WoG_fresh/WoG/vla/vla.py").read()
    tr = open("/home/zhengguantian/WoG_fresh/WoG/scripts/train.py").read()
    res["6_source_wiring"] = {
        "PASS": all(k in src for k in [
            "_aggregate_step_actions", "_student_timeline",
            "self.distill_weight", "use_future=False",
        ]) and "distill_weight" in tr,
        "note": "aggregation + stage-2 distillation wired through",
    }
except Exception as e:
    res["6_source_wiring"] = {"ERROR": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_block6.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
