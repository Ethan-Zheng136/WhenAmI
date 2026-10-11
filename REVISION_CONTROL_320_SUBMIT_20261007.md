# 新修订实验：真实OXE三节点后见修订 vs 直接历史注意力

命名统一：本实验称 **WhenAmI-D**。
**D0=`direct_history`，D1=`revised_nodes`**；脚本、variant和任务目录不重命名。
实际pipeline见`WHENAMI_D_PIPELINE_20261007.md`。D不是C之后的第四训练stage。

日期：2026-10-07 UTC。**这是新的修订实验，不是pooled-B，也不是C继续训练。**
独立代码：`whenami_revision_control/`；不修改既有A/B/C或另一服务器正在评测的模型。

本地验收已完成：26项测试（含4-rank嵌套FSDP恢复）、真实B1/A50k加载与单步SGD、
零残差B输出一致、旧节点到动作的梯度/受控干预、27域原始执行动作核验、
410个DEV样本动作区间核验及两组preflight。
真实单卡smoke峰值allocated约49.62GiB，不是320卡完整Adam的显存测量。
源码绑定证明：`whenami_revision_control/LOCAL_VERIFICATION.json`。

## 1. 提交命令

每条分别申请 **20节点 ×16 PPU=320卡**。可以同时或先后提交。

### 对照：直接读取完整合法历史

```bash
REVISION_VARIANT=direct_history PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_revision_control_320.sh
```

### 实验：显式修订同一旧节点

```bash
REVISION_VARIANT=revised_nodes PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_revision_control_320.sh
```

平台启动器入口`WORLD_SIZE=20`表示节点数，不手工设为320。
两条不是一个40节点任务中的两条顺序命令。无需重新准备OXE，不重训A。

启动器要求源码绑定的`whenami_revision_control/LOCAL_VERIFICATION.json`；
本地证明未完成或源码/来源变动时拒绝提交，不可绕过该检查。

## 2. 这版实现了什么

```text
窗口：[旧事件0，旧事件1，当前事件2]
每个事件有独立图像/指令编码、真实时间和此前完整已执行动作区间

                 原B1的最近两帧路径
                         │
                  base_condition + z
                         │
已观测三节点 → 节点模块 → 修订后的旧节点condition
                         │
                   零初始化条件残差
                         │
                 原DiT预测当前事件2的动作
```

- 原B动作路径保留；初始残差为0时复现原B的condition/cognition/velocity。
- 记忆分支每个节点都有**独立逐帧VLM cognition和local-Q视觉特征**，
  不把当前帧cognition广播给过去版本，也不是缺少任务语义的纯视觉memory。
- 两组拥有相同参数集合、初始值、数据、已执行动作、teacher目标、损失和预算。
- `direct_history`每次从原始节点query重新读取全部合法历史；
  `revised_nodes`用上一步更新后的旧节点作query，再接入新证据。
  两组都能访问完整已观测原始memory，不通过减少baseline信息制造优势。
- 被辅助监督的旧节点condition直接进入动作残差读出，不是旁路辅助头。

**范围边界：** 这是最多三个已观测锚点的窗口内重建/修订原型。
每个窗口内部按证据到达顺序扫描，并保留同一旧节点的不同版本；
尚不是跨整段任务维护的无界/长期bank，没有动作条件未来预测F或完整执行重规划R。
不把本次工程接线或一次SGD称为方法已有效/新颖性已成立。

## 3. 两个来源，不要混淆

### 学生策略初始化：已经有旧轮闭环结果的B1-20k

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b/training/
timeline_b_condition_align_A10k_320_20k_01/main/checkpoints/step-020000

COMMITTED SHA256:
4d9f3b3e0722a570a8f8e17744652a8239509e65f7db4f9c25156d40f92805bb
```

即 **A10k→B1-20k**，不是新A50k→B1。
加载其完整1550项模型状态；原forecast_head保留但冻结、不再用于本轮损失。
memory/bridge为新参数，优化器重新建立，不恢复旧B optimizer。

### 固定教师：Timeline-A50k

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_joint/training/
timeline_joint_timeline_320_50k_01/main/checkpoints/step-050000

COMMITTED SHA256:
f785848d08a55bc5aee4e821b3feac61b3543665e6117982b7eeb0cbbb301714
```

教师加载1548项状态，全程冻结。每条训练样本做两次教师前向：

```text
只看事件[0,1] → 对旧节点0、1的判断 C_before*
已看到[0,1,2] → 对同一旧节点0、1的新判断 C_after*
```

当前决策是事件2；两次教师都不看事件2之后的真实图像或动作。
事件2是旧节点的后续证据，但对当前动作是合法当前观测。

## 4. 已执行动作与训练标签严格分离

采样仍使用既有OXE27三真实锚点，每段间隔1–12，片段内指令一致。

```text
事件0            事件1             当前事件2
  └── a[0:1) ────┘  └── a[1:2) ────┘
         已执行命令          已执行命令
                                     └── a[2:...]：仅动作监督目标
```

已执行区间从前一个锚点的真实动作窗口取完整区间，不平均成一个7D向量，
也不把未执行动作尾巴当成执行历史。
数据审计逐域与原始OXE动作数组核对区间，不仅检查张量shape。
篡改当前事件2的未来动作标签不会改变模型输入。

## 5. 损失与训练配置

令`N`为每个condition token的L2归一化：

```text
L = 当前事件2的action FM
  + 0.1 × cosine_distance(student_after_old_nodes, teacher_after_old_nodes)
  + 0.1 × mean ||(N(student_after)-N(student_before))
                  -(N(teacher_after)-N(teacher_before))||²
```

最后一项对每个token的condition维求和，再平均节点/token/batch。
教师两次输出detach；A的latent只是动作相关教师表征，不是物理事件真值。
两组使用相同损失，区别不是“有无辅助监督”。

|项目|值|
|---|---|
|更新数|20,000|
|global batch / micro|320 / 1|
|动作FM draws|每样本2次|
|原B主体峰值LR|**1e-6**，保守适配已有策略|
|memory/bridge峰值LR|**1e-4**|
|warmup|1,000，原cosine floor=0.1|
|可训练|原VLM、QFormer、Timeline、DiT及新memory/bridge|
|冻结|A教师、独立DINO/Wan、旧forecast_head|
|分布式|hybrid_zero2，LLM activation checkpointing|
|常规保存/DEV|2,500 / 1,000步|
|显示|tqdm、秒/步、ETA、显存、动作/节点/修订loss、残差RMS|

这是新实验的统一配方，不是声称与旧B只差一个参数。
当前实现优先保证原B路径一致和逐节点因果性，存在额外逐帧VLM计算：
原B最近两帧路径之外，memory独立编码三帧；教师另做2帧/3帧两次前向。
**预计比原B更重，不承诺仍是约2秒/步。** 速度/显存以实际320卡验收和ETA为准；
之后若合并重复编码，必须重新验证数值与因果性，不能热改运行任务。

## 6. 自动硬件验收

```text
本地证明/固定双来源/数据分割预检
→ 实际320-rank小模型保存恢复
→ 完整学生+教师训练100步，保存/DEV
→ 新进程从100恢复到120，保存/DEV
→ 检查完整更新、八组梯度、非零桥更新、optimizer/RNG和评测协议
→ 从120继续到20,000
```

前120步属于总20k，不额外加20k。任一关失败不进入后续大训练。
启动/存盘/DEV/重启期间会暂时没有训练进度增长。
本地BF16主体单步SGD不等同320卡全FP32-master Adam已验收。

输出：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_revision_control/training/
  revision_control_direct_history_B1A10k20k_A50k_320_20k_01/
  revision_control_revised_nodes_B1A10k20k_A50k_320_20k_01/
```

每组有`admission1/`、`admission2/`、`main/`。
fresh任务拒绝覆盖已有段；任意中断后先检查checkpoint，不盲目重提同RUN_ID。

## 7. 重要评测变化：不能与旧B/C的MSE直接混表

仍用同410个DEV episode、27域，但本轮预测的是**第三/最后锚点**的动作，
前两个锚点和中间真实已执行命令都可见。
旧B/C的DEV通常预测第二/中间锚点，因此数值不能直接拿来排名。

本轮结果写明：

```text
protocol = revision_last_anchor_executed_history_v1
student_observed_nodes = 3
executed_past_commands_as_input = true
```

- `admission1/development/step-000000/`为同协议零残差B初始化基线。
- 中间DEV使用seed42，最终20k使用采样seed42/43/44；不是三个训练seed。
- 两组的新DEV可以配对比较；若要比较原B/WoG，必须在这套同协议重新评测。
- 新SIMPLER wrapper还需支持三真实锚点和**实际已执行**命令的时间对齐。
  不能拿旧B的两帧wrapper直接换权重，也不能输入仿真专家未来动作。
- episode reset必须清空窗口；不满三节点是冷启动扩展，不伪造历史。
  本轮smoke检查接口/初始B保持，不等于训练覆盖了所有冷启动分布或已完成NVIDIA验收。

## 8. 本地证据

```text
代码/证明:
/home/zhengguantian/WoG_fresh/whenami_revision_control/

测试、真实smoke、数据核验:
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_revision_control/local_validation_20261007_01/
```

以`LOCAL_VERIFICATION.json`的`local_passed=true`和绑定源码为准。
smoke中的受控非零bridge干预用于证明“旧节点→动作”通路，
不是已经学出更好的动作；实际更新只有一次SGD工程检查。
首次visual-only memory smoke被主动停止，随后补齐独立逐帧VLM cognition，
重新执行`real_smoke_v2`，旧日志保留且不作为通过证明。

**本轮提交是验证新机制的真实训练，不是效果保证、首创保证或完整执行时间线已经完成。**
