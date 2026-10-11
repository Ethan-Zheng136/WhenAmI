# 新增640PPU：固定A10k的因果B转化实验

2026-10-05。现有A两组640卡保持运行；以下是**额外两组各320卡**，
不是替换A任务，也不是将两条命令放进同一个40节点任务。

## 实验目的

固定同一A-10k模型做教师和学生初始化：

```text
老师看：历史、当前、后续真实观测 → 当前节点的特权条件
                                            ↑ 对齐
学生看：历史、当前 → 当前条件 → 辅助预测头
                       └── 动作DiT → 当前动作
```

比较：

- B0 `action_only`：只用当前动作FM。
- B1 `condition_align`：相同动作FM＋0.1教师条件余弦对齐。

两组完整训练VLM/QFormer/时间模块/DiT；教师与独立DINO/Wan冻结。
辅助预测结果不输入动作头，不允许teacher target/未来图像成为student输入。
两组都计算教师前向以匹配计算预算，B0不使用它的target作为loss。

本轮是当前决策转化pilot，学生输入为**一个历史锚点＋当前锚点**，
不是旧Stage3四张历史帧接口，也不是最终全时间节点蒸馏。
A10k被固定为本轮中间教师，之后A15k/50k不会自动替换它。

## 已通过的本地验收

- 13项CPU/入口测试通过；
- CPU4rank、嵌套FSDP：A-DCP到B严格模型加载（仅新增辅助头缺失）；
- 同样的分片B model/Adam/RNG保存恢复，下一更新逐值一致；
- 实际A10k老师和学生加载，二者权重不共享存储；
- 真实TRAIN图像上B0/B1各条动作梯度和对齐梯度检查；
- 一次真实SGD工程更新，老师保持不变；
- 改变未来原图/语言不改变学生输出；
- 27域因果学生开发评测路径通过；
- 现有A源码与运行配方一致，无改动。

实际smoke约47.60GiB，使用BF16学生和单步SGD；
**这不等于实际320卡FP32Adam验收通过**。该检查由提交脚本强制执行。

## 任务1：动作监督对照B0

申请20节点，每节点16PPU：

```bash
B_VARIANT=action_only PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_b_320.sh
```

## 任务2：教师条件监督B1

另申请20节点，每节点16PPU：

```bash
B_VARIANT=condition_align PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_b_320.sh
```

平台提供WORLD_SIZE=20、RANK、MASTER_ADDR，不要设成320。
默认RUN_ID均为新目录；已存在时拒绝覆盖。

## 自动执行顺序

```text
320rank拓扑恢复测试
       ↓
B全模型1–100步 → 保存＋27域因果评测
       ↓
新进程101–120步 → 保存＋评测＋恢复检查
       ↓ 通过后才继续
从120步续训到20,000步
```

任一步失败停止，不跳过。`PHASE=admit`可只跑到120；
之后同配置用`PHASE=full`继续。

固定配方：global320轨迹、每轨迹一个当前动作块、2次FM抽样；
已有body LR1e-5、新辅助头LR1e-4、warmup1000/cosine；
20k更新、每2500保存、每1000评测，最终评测3个采样seed。
数据直接复用OXE27训练分片和410个DEV episode，不需要重下/预处理全OXE。

## 产物

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b/training/
  timeline_b_action_only_A10k_320_20k_01/
  timeline_b_condition_align_A10k_320_20k_01/
```

源码：`/home/zhengguantian/WoG_fresh/whenami_timeline_b/`
本地验收：`outputs/whenami_timeline_b/LOCAL_VERIFICATION.json`

## 解释边界

这轮只检验在共同A10k预训练之后，额外教师条件监督能否改善因果控制。
不能单独证明全部timeline预训练的价值，更不能直接证明闭环成功率。
完整研究仍需固定最终老师、pooled/timeline完整A→B对比、多训练seed及闭环评测。
研究证据安排见`WHENAMI_RESEARCH_EVIDENCE_PLAN_20261005.md`。
