# 640卡提交：两组各320卡的全VLM Timeline-A

2026-10-05。本轮联合训练代码已经实现并完成本地验收，
不是原来冻结VLM的32卡脚本改节点数。

## 1. 已完成的检查

- 原先pooled/independent/timeline三组实际32卡均完成100→120步保存、重启恢复、27域评测。
- 新联合模型可训练参数：8,062,383,335。
- 本地实际完整预训练模型的FSDP前向/反向通过：
  VLM视觉、投影、LLM、局部QFormer、时间模块、原始DiT均有有效动作梯度。
- 隐藏节点的原图/特征改变不会从旁路影响条件。
- 原图重新编码VLM的27域开发评测通过。
- 18项测试通过，包括两组小模型的CPU4rank、2个分片组、
  hybrid ZeRO2 model/Adam/RNG保存恢复后下一更新一致。
- 本地完整模型反向峰值约62.62GiB。它不是320卡吞吐或完整Adam显存的估计。

**未完成的检查：新完整模型在实际320卡上的Adam更新/保存恢复。**
单96GiB PPU无法容纳全部FP32模型、梯度和Adam状态；
因此提交脚本先在实际拓扑强制验收，通过后才继续。

## 2. 你要提交的两条命令

每条命令使用独立的**20节点×16PPU=320卡**任务；
两条同时运行合计40节点640卡。不要作为一个40节点任务运行本脚本。
平台提供WORLD_SIZE=20、RANK和MASTER_ADDR。

### Timeline

```bash
JOINT_MODE=timeline PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_joint_320.sh
```

### 同预算、目标时间可查询的聚合对照

```bash
JOINT_MODE=pooled_query PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_joint_320.sh
```

`pooled_query`不是旧pooled：同样有各目标的局部基态、目标时间查询和相同上下文，
但先压缩来源节点。不会强制所有目标输出相同条件；两组参数、初始权重与零残差
起始行为匹配。它也不是官方WoG复现，不能用名称替代科学对照分析。

## 3. 一次提交会做什么

```text
实际320rank的小拓扑保存恢复测试
                ↓
全模型训练1–100步，保存、27域评测
                ↓
新进程恢复101–120步，保存、27域评测
                ↓
自动检查：loss/梯度、全部学习路径、配方、checkpoint、恢复、评测
                ↓ 只有通过才继续
新进程从120步续训到50,000步
```

任一步失败脚本停止，不会悄悄跳过。总训练schedule从第1步就是50k，
100/120步只是阶段检查点，不会重新计学习率或重头训练。

若希望验收后人工决定，可把`PHASE=run`换成`PHASE=admit`；
验收结束再使用`PHASE=full`。不要对已经生成输出的任务重复用run/admit覆盖。

## 4. 固定配方与输出

| 项目 | 每组配置 |
|---|---|
| 训练范围 | VLM视觉/投影/LLM＋局部QFormer＋时间模块＋完整DiT |
| 冻结 | 独立DINO/Wan |
| 初始化 | 共同Future Stage1-20k；不是Stage3-50k或32卡optimizer续训 |
| 数据 | 既有OXE27原图TRAIN；不重新下载/全量准备 |
| 开发集 | 固定410episode/27域，零本轮TRAIN重叠，基础预训练暴露不排除 |
| batch | 单卡1轨迹；global320轨迹×3锚点=960观测 |
| FM | 每节点2次draw |
| 优化 | 预训练参数LR1e-5；时间模块1e-4；warmup1000、cosine |
| 训练 | 50k更新；两组同样预算 |
| 数据增强 | 本轮两组均无image augmentation，明确不同于旧Stage3增强配方 |
| 执行 | hybrid ZeRO2、LLM activation checkpointing |
| 保存/评测 | 每5k保存、每2.5k评测，另有100/120阶段检查 |
| 进度 | tqdm、loss/LR、最大显存、滚动步时、包含已测保存评测开销的ETA |

输出：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_joint/training/
  timeline_joint_timeline_320_50k_01/
  timeline_joint_pooled_query_320_50k_01/
```

两者均分`topology_probe/`、`admission1/`、`admission2/`、`main/`。
不要把旧32卡DDP状态跨拓扑直接加载为320卡FSDP。

## 5. 科学边界

这是**完整联合训练的A阶段特权教师**：遮挡节点可借用后来时刻的真实图像。
它仍不等于可部署的最终WhenAmI；B阶段撤除未来仍需实现和验证。
本轮本地测试证明的是梯度、信息边界和工程入口，不证明创新性或性能。
此前主Stage3与32卡实验的源码、模型和日志均保留不动。

代码：`/home/zhengguantian/WoG_fresh/whenami_timeline_joint/`

证据：

```text
outputs/whenami_timeline_joint/
  LOCAL_VERIFICATION.json
  frozen32_admission_audit.json
  full_model_smoke_02/RESULT.json
  full_model_smoke_02/development27/RESULT.json
  tests_delivery.log
  submission_preflight_and_dryrun.txt
  source_snapshot/
```
