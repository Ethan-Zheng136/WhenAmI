# Real Timeline：真实OXE多锚点动作训练

2026-10-03。独立研究分支，不改当前Stage3，也不接手另一agent的主表/部署。

## 不是再等50k，也不是再做合成分类

现有A保存了整条真实轨迹，已有Past基础权重可直接使用。
因此这里建立：

```text
真实OXE episode，三个实际时刻 k0 < k1 < k2
每点：自己的历史图像/当前图像/指令 → 冻结DINO/Wan/VLM/Past → H_k,C_k,z_k

H0,z0 → 状态s0
     └─真实已执行u[0:1)、Δt→ F → 预测s1 → o1证据更新 → s1、b[0|1]
                       └─u[1:2)、Δt→ F → 预测s2 → o2证据更新 → s2、b[0,1|2]

每个合法当前点的状态 → 条件残差桥 → SAME Past DiT → a[k:k+16]动作FM
```

三个锚点每个都有16个query，每query内部hidden128；
**三个真实时间节点与16个query是两个不同轴**。

## 本轮的具体算法选择

- 正向：实际区间动作序列GRU、实际Δt，预测下一锚点条件/状态；
- filter：真实该锚点的因果Past特征融合到预测状态；
- 后向：当前观测特征与预测之间的innovation，通过attention和门控残差更新旧节点；
- 当前动作：读取修订bank，产生当前条件残差，由一个共享的原Past DiT读出；
- 保留每个cutoff的下三角状态`b[k|t]`，但只用`b[<=k|k]`预测当前k的动作；
- 辅助prediction目标是下一点冻结Past条件，不伪造grasp/phase标注，
  也不声称该条件是完美物理状态；
- 门控是研究候选，不是已证明的证据归属/置信度机制。

尚未实现：遥远未来teacher的完整平滑监督、修订后重算整条filter轨迹、
长期有界memory、在线候选动作规划。不能称最初全部WhenAmI愿景已经实现。

## 四个模式

| MODE | 在线动作条件 |
|---|---|
| independent | 每个节点独立Past＋小残差 |
| forward | 正向action-conditioned filter |
| direct_history | 同容量直接读合法历史bank |
| revision | innovation门控、递推修订旧bank |

四组保留相同模块/初始化用于明确比较；其中independent/forward有未参与动作路径的模块，
不能称有效计算量完全相同。核心强对照为direct_history/revision：
相同信息、参数量、bank访问次数与监督。

当前固定基础策略，主要为了让“timeline怎样改变真实动作”可检验且不破坏正在训练的模型。
它不是把冻结端点当永久架构限制。是否需要联合解冻，应是后续独立研究轴，
不在本轮一起改动而失去归因。

## 本机检查与正式研究要分开

`smoke_real.py`：
- 27个域各一条原TRAIN轨迹、共81个真实锚点；
- 实际预训练视觉/VLM/Past/DiT，不是随机小动作头；
- 每种模式8步、batch2条轨迹、每点2次FM；
- 同一小特征集合只用于接线/数值对照，不能比较8步结果宣称泛化收益；
- 检查仅action loss到F/B/门控/读出，基础参数仍冻结；
- 检查未来锚点改变不影响早期动作条件，以及保存重载一致。

`train_real.py`：
- 每一步从完整A索引重新抽真实TRAIN轨迹，现场原图编码；
- 不训练在上述27条固定特征上；
- 独立的研究internal_dev来自原TRAIN按episode hash的5%，不访问主line dev/sealed；
- internal dev仅监测固定样本的共享DiT动作FM，不是主表/闭环评测；
- 基础端点曾在OXE预训练，不能称研究internal_dev是全训练历史未见测试。

## 命令（读取LOCAL_VERIFICATION确认本地验收后使用）

每个对照任务 **2节点×16卡=32 PPU**：

```bash
TIMELINE_MODE=direct_history SEED=42 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_whenami_timeline_research.sh
```

```bash
TIMELINE_MODE=revision SEED=42 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_whenami_timeline_research.sh
```

这是两个独立研究任务，可以分配不同节点同时运行。
使用`TIMELINE_MODE`，不要写`MODE`：PPU平台的shell初始化会将`MODE`覆盖为`CUDA`。
每组5000更新，global64条轨迹×3锚点×2次FM；
每500保存，每250监测内部动作FM，先2步保存/退出，再新进程恢复到5k。
新checkpoint只含约1.26M timeline及Adam，不反复存8B冻结模型。
这些任务不替代或中断当前Stage3。

第一对任务确认吞吐、梯度和机制诊断后，再以seed43/44做独立复核；
不能把seed42单次内部loss变化直接当作方法成立。

## 关键产物

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline/
  real_smoke_01/
  raw_runner_single_01/
  <mode>_seed<seed>_32ppu_5k_01/
```

第一次多节点研究任务仍是新的配置，需要检查两步保存恢复，不因本机通过就保证32PPU全部兼容。
`LOCAL_VERIFICATION.json`是本地代码与权重来源检查，不是伪造的集群成功标记。
