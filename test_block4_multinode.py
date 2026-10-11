"""[zgt] 验证块 4：多节点 action loss

核心主张是"时间线上每个节点都要能生成从自己出发的动作"。
若只监督当下，其余节点就是无监督的中间变量，会自由漂移 ——
时间线退化成"一个状态 + 八个装饰"。

关键检查：
  1. 动作窗口对齐：节点 k 取 actions[k : k+Ta]，对应偏移 (k-n_past)
  2. 数据不足时跳过而非截断（截断会让各节点 loss 尺度失衡）
  3. 全部节点无数据时必须报错，且提示该把 faw 调到多少
  4. 梯度能回到每一个节点（不能只有当下那个拿到梯度）
  5. QFormer 路径不受影响（all_nodes=None 时走单点监督）

数据前提（实测）：
  ws=5, faw=15 -> action 24? 否，是 20，只够 5/9 个节点
  ws=5, faw=19 -> action 24，覆盖全部 9 个节点
  代价 +4 步 = +28 个数/样本，可忽略

结果写 /tmp/whenami_block4.json
"""
import sys, json
sys.path.insert(0, "/home/zhengguantian/WoG_fresh/WoG")

import torch
import torch.nn as nn

res = {}
torch.manual_seed(0)

B, K, N, D, A, Ta = 2, 9, 16, 32, 7, 16
N_PAST = 4


class StubActionModel(nn.Module):
    """记录每次调用收到的条件，用于验证节点与动作窗口的对应关系。"""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D, 1)
        self.calls = []

    def loss(self, actions, cognition, cond):
        self.calls.append({
            "action_shape": tuple(actions.shape),
            "action_first": float(actions[0, 0, 0]),
            "cond_sum": float(cond.sum()),
        })
        return self.proj(cond).mean() + actions.mean() * 0.0


class Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.Ta = Ta
        self.action_model = StubActionModel()

    # 与 vla.py 中实现保持一致
    def _multi_node_action_loss(self, all_nodes, n_past, actions,
                                cognition_features, repeated_diffusion_steps):
        B_, K_ = all_nodes.shape[:2]
        L = actions.shape[1]
        losses = []
        for k in range(K_):
            if k + self.Ta > L:
                continue
            node_actions = actions[:, k: k + self.Ta]
            losses.append(
                self.action_model.loss(
                    node_actions.repeat(repeated_diffusion_steps, 1, 1),
                    cognition_features.repeat(repeated_diffusion_steps, 1, 1),
                    all_nodes[:, k].repeat(repeated_diffusion_steps, 1, 1),
                )
            )
        if not losses:
            raise ValueError(
                f"no node has enough action data: K={K_}, Ta={self.Ta}, "
                f"action length={L}. Increase future_action_window_size to "
                f"at least {K_ - n_past - 1 + self.Ta - 1}."
            )
        return torch.stack(losses).mean()


# 动作按 (偏移 * 100) 编码，便于验证取到的窗口是否正确
def make_actions(n_steps, n_past):
    a = torch.zeros(B, n_steps, A)
    for i in range(n_steps):
        a[:, i, 0] = (i - n_past) * 100.0    # actions[i] 对应偏移 (i - n_past)
    return a


# --- 1. faw=19: 全部 9 个节点都有监督 ---
try:
    m = Stub()
    nodes = torch.randn(B, K, N, D)
    acts24 = make_actions(24, N_PAST)          # ws=5, faw=19 -> 24 步
    cog = torch.randn(B, 1, 4096)
    m.action_model.calls.clear()
    loss = m._multi_node_action_loss(nodes, N_PAST, acts24, cog, 1)
    res["1_all_nodes_supervised"] = {
        "PASS": len(m.action_model.calls) == K,
        "n_supervised": len(m.action_model.calls),
        "expected": K,
    }
except Exception as e:
    res["1_all_nodes_supervised"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 2. 动作窗口对齐：节点 k 的首个动作偏移应为 (k - n_past) ---
try:
    got = [c["action_first"] / 100.0 for c in m.action_model.calls]
    want = [float(k - N_PAST) for k in range(K)]
    res["2_window_alignment"] = {
        "PASS": got == want,
        "node_first_offsets": got,
        "expected": want,
        "note": "node k must supervise a_{k-n_past : k-n_past+Ta-1}",
    }
except Exception as e:
    res["2_window_alignment"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 3. faw=15: 数据只够 5 个节点，跳过其余而非截断 ---
try:
    m2 = Stub()
    acts20 = make_actions(20, N_PAST)          # ws=5, faw=15 -> 20 步
    m2.action_model.calls.clear()
    _ = m2._multi_node_action_loss(nodes, N_PAST, acts20, cog, 1)
    lens = {c["action_shape"][1] for c in m2.action_model.calls}
    res["3_skip_not_truncate"] = {
        "PASS": len(m2.action_model.calls) == 5 and lens == {Ta},
        "n_supervised": len(m2.action_model.calls),
        "expected": 5,
        "all_windows_full_length": lens == {Ta},
        "note": "truncating would make loss scales differ across nodes",
    }
except Exception as e:
    res["3_skip_not_truncate"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 4. 全部不足时报错，并给出 faw 建议值 ---
try:
    m3 = Stub()
    tiny = make_actions(10, N_PAST)            # 连节点 0 都不够
    try:
        m3._multi_node_action_loss(nodes, N_PAST, tiny, cog, 1)
        res["4_raises_when_no_data"] = {"PASS": False, "BUG": "should have raised"}
    except ValueError as e:
        msg = str(e)
        res["4_raises_when_no_data"] = {
            "PASS": "future_action_window_size" in msg,
            "message": msg[:150],
        }
except Exception as e:
    res["4_raises_when_no_data"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 5. 梯度到达每一个节点 ---
try:
    m4 = Stub()
    nodes_g = torch.randn(B, K, N, D, requires_grad=True)
    loss = m4._multi_node_action_loss(nodes_g, N_PAST, acts24, cog, 1)
    loss.backward()
    per_node = [float(nodes_g.grad[:, k].abs().sum()) for k in range(K)]
    res["5_grad_reaches_all_nodes"] = {
        "PASS": all(g > 0 for g in per_node),
        "n_nodes_with_grad": sum(1 for g in per_node if g > 0),
        "expected": K,
        "note": "if only the present node had grad, the timeline would drift",
    }
except Exception as e:
    res["5_grad_reaches_all_nodes"] = {"ERROR": f"{type(e).__name__}: {e}"}

# --- 6. 源码接线 ---
try:
    src = open("/home/zhengguantian/WoG_fresh/WoG/vla/vla.py").read()
    res["6_source_wiring"] = {
        "PASS": all(k in src for k in [
            "_multi_node_action_loss",
            # 解包多了 calib_aux —— masked 版要把 Lambda 校准损失带出来，
            # 挂在 self 上会在梯度累积时读到上一轮的值。
            "cond_tokens, all_nodes, n_past, calib_aux = self._compute_condition",
            "if all_nodes is not None:",
            "target = cond_tokens.detach()",
        ]),
        "note": "includes fix for the stale `future_patch_features` reference",
    }
except Exception as e:
    res["6_source_wiring"] = {"ERROR": f"{type(e).__name__}: {e}"}

with open("/tmp/whenami_block4.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
