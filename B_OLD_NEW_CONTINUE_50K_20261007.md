# 旧B / 新B：从各自20k真正恢复，累计训练到50k

日期：2026-10-07 UTC。
**四个独立续训任务；每个20节点×16 PPU=320卡。**
全部同时运行需1280卡；资源不够可以分批提交，不需要四组同时启动。
这不是重新从A warmstart，也不是对原20k实验直接篡改`--steps`。

**2026-10-07 15:12 UTC本地验收完成：** 22项测试、四份真实B20k模型及对应
教师的前向/反向检查、64份Adam/1280份RNG审计和四case本地preflight均已通过。
实际320rank完整恢复/回放仍在提交后执行。本机没有执行正式续训。
六项B/C汇总入口见`BC_SIX_CONTINUE_50K_20261007.md`，C的验收状态独立记录。

## 1. 四条提交命令

### 旧B0：A10k起点，action-only

```bash
B_CONTINUE_CASE=old_b0 PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_b_continue_320.sh
```

### 旧B1：A10k起点，condition-align

```bash
B_CONTINUE_CASE=old_b1 PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_b_continue_320.sh
```

### 新B0：A50k起点，action-only

```bash
B_CONTINUE_CASE=new_b0 PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_b_continue_320.sh
```

### 新B1：A50k起点，condition-align

```bash
B_CONTINUE_CASE=new_b1 PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_b_continue_320.sh
```

每条作为一份独立20节点任务提交。
平台入口`WORLD_SIZE`表示节点数20，torchrun内部才是320rank；
不要手工设为320，也不要把四条放进一个80节点任务执行。
沿用之前的容器、挂载和MASTER_ADDR/RANK注入方式。

## 2. 到底恢复什么

|选择|恢复学生|冻结教师|
|---|---|---|
|old_b0|旧A10k→B0的step-020000|原A10k，不升级|
|old_b1|旧A10k→B1的step-020000|原A10k，不升级|
|new_b0|新A50k→B0的step-020000|原A50k|
|new_b1|新A50k→B1的step-020000|原A50k|

恢复内容包括：

- 完整B模型；
- 对应16份Adam优化器状态，包括moment、step计数和两组参数布局；
- 各320个rank自己的RNG状态；
- 使用连续的全局step：第一步是**20,001**，最后一步50,000；
- 数据采样和FM噪声种子继续按全局step计算，不重新从step1取样。

父checkpoint：

```text
旧B0：
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b/training/
timeline_b_action_only_A10k_320_20k_01/main/checkpoints/step-020000

旧B1：
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b/training/
timeline_b_condition_align_A10k_320_20k_01/main/checkpoints/step-020000

新B0：
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b50/training/
timeline_b50_action_only_A50k_320_20k_01/main/checkpoints/step-020000

新B1：
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b50/training/
timeline_b50_condition_align_A50k_320_20k_01/main/checkpoints/step-020000
```

每份COMMITTED及教师身份都在新代码中固定哈希，不允许自动选择“最新”权重。
不改老师、不把旧B换成A50k，不改变model/data/objective。

## 3. 学习率：保持父20k的末端低LR，不重启warmup

四组统一：

```text
原0–20k：保留原实验的历史日程，不重新解释

20k → 50k：
  主体 LR = 1e-6
  辅助头 LR = 1e-5
  恒定保持各自父checkpoint中实际保存的末端值
  无新warmup，不升回1e-5 / 1e-4
```

目的是先检验“在已下降的末段低学习率继续优化30k”是否改善，
不同时混入高LR重启或更改结构。
这是**20k原日程 +30k低LR续训**，不等同从step0按50k cosine训练的实验。
旧/新对照采用相同的续训合同并记录额外预算，不把额外训练收益当新架构收益。

其它配置：

- global batch320，micro1，仍为每样本2次FM噪声采样。
- 原数据分割、1–12可变时间间距、教师当前条件目标、对齐权重0.1保持。
- 原AdamW、gradient clipping、hybrid_zero2、LLM activation checkpointing保持。
- 每2500步保存模型+Adam+RNG；每1000步做DEV。
  每个存盘点也额外做DEV（例如22.5k），保证中断后任何保存点都有恢复回放参照。
- 25k/30k/35k/40k/45k/50k使用3个采样seed42/43/44，
  其它常规DEV为seed42。比较中间走势时只取相同seed，不能混均值。
- tqdm显示累计step，progress同时记录新增更新数和ETA。

## 4. 为什么需要独立续训入口

原`CheckpointIO`严格要求同world、同布局、同recipe。
本次不放松它，而是：

```text
首次恢复20k：
验证固定父checkpoint
→ 用父checkpoint原recipe恢复model/Adam/RNG
→ 建立包含父完整recipe/manifest、额外预算和新LR规则的续训recipe

续训checkpoint保存/再次恢复：
全部使用新的续训recipe，继续严格一致性检查
```

这样不会通过“把父recipe改成50k”绕过原校验，
也不会误用model-only加载而丢弃Adam/RNG。

## 5. PHASE=run自动执行

```text
本地证明/父checkpoint/教师/数据预检
→ 实际320-rank小模型topology保存恢复测试
→ 按父recipe完整恢复20k模型、Adam、逐rank RNG
→ 重跑父20k的3-seed DEV，与已存预测逐样本比较
→ 训练20,001–20,100，保存/DEV
→ 新进程恢复20,100，核对保存预测，训练到20,120，保存/DEV
→ 检查全部320rank的Adam/RNG恢复、梯度、固定LR、连续step
→ 新进程恢复20,120，再继续至50,000
```

共新增30,000个更新；其中前120个更新用于实际模型验收，不是额外重跑20k。
恢复回放使用同一FSDP后端，逐预测最大绝对差容差1e-6，不是跨PPU/NVIDIA比较。
任何加载/回放/验收失败都停止，不自动换老师、重置optimizer或放宽数值门槛。

初始化、评测、checkpoint保存和新进程加载需要时间，期间tqdm可能不动。
不承诺实际速度一定等于原B，需以提交后的日志为准。

## 6. 输出与任务恢复

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_b_continue/training/
  b_continue_old_b0_20k_to50k_01/
  b_continue_old_b1_20k_to50k_01/
  b_continue_new_b0_20k_to50k_01/
  b_continue_new_b1_20k_to50k_01/
```

每组包含：

```text
topology_probe/
admission1/                  # 从父20k恢复
  restore-rank-*.json        # 320份Adam/RNG恢复计数证明
  CHECKPOINT_REPLAY_VERIFIED.json
  development/step-020000/   # 新进程重放原20k
  checkpoints/step-020100/
admission2/
  checkpoints/step-020120/
main/
  recipe.json
  LOAD_VERIFIED.json
  RESUME_VERIFIED.json
  progress.json
  train.jsonl
  checkpoints/step-050000/
  development/
  COMPLETE.json
```

原20k结果、正在运行的D、pooled-B和其它源码/评测包都不覆盖、不修改。
重复fresh提交会拒绝已有输出。`PHASE=full`仅适合20,120验收已通过且`main`尚未创建。
中断时不要盲目重提默认命令；需要依据实际续训checkpoint显式恢复到新输出段，
保持新recipe不变。不是从最新模型随意重新初始化optimizer。

## 7. 验证边界

本地证据以`whenami_b_continue/LOCAL_VERIFICATION.json`为准：

- 原模型/数据/损失复用及四case启动命令检查；
- CPU4-rank嵌套FSDP验证原recipe恢复、续训recipe保存/恢复、
  下一次模型更新与Adam moments逐值相同，RNG恢复；
- 读取原四组共64份Adam、1280份RNG；核对moment shape/dtype、实际step计数；
- 真正加载四组B20k及对应原教师，验证实际下一步样例前向/反传。

单卡不装载8B完整FP32 Adam状态，不能把这些测试称为实际320卡大模型恢复已经通过。
该项由上述cluster gate完成。父Adam审计不对全部moment元素做有限性扫描，
源码与checkpoint恢复边界均如实记录。

这四条命令只针对旧/新B，不包含C续训，也不改变D的现行20k任务。
