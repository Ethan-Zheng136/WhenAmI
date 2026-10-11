# Timeline-C：从A50k出发的下一版学生实验

2026-10-06。C是新实验代号，不是必须串在A→B之后的第三个训练阶段。
这两组直接从A50k初始化，不从B20k resume，不修改已交付B10k/B20k或正在执行的SIMPLER任务。

## 1. A、B、C的“未来”分别在哪里

- A训练老师：允许真实过去、当前、未来图像，共享动作监督塑造时间节点条件。
- 本轮旧B：学生只看过去＋当前，B1对齐老师的当前条件；没有显式未来节点进入动作。
- C学生：仍只看过去＋当前，但查询当前与未来t+4两个时刻的动作条件；
  未来节点是预测，绝不是真实未来输入，也不是生成未来图像。
- C老师：冻结同一个A50k，读取三张真实图，给当前与未来节点提供监督目标。

```text
固定 A50k 老师：
真实 past/current/future(t+4) → [C*_current, C*_future]
                                      │
                               两组相同的对齐监督
                                      │
学生（从同一个A50k初始化，主体可训练）：
真实 past/current → 时间查询 → [C_current, predicted_C_future]
                                  │             │
       future_aux：动作仅读当前 ───┘             │
       future_read：动作读当前＋预测未来 ────────┘
```

future_aux动作头使用两份相同当前条件，future_read使用当前和逐渐接入的预测未来；
两组条件token数量都是32、模型参数集合相同，初始读出相同。
read组的接入比例由0在前1000步逐渐升到1，aux组始终为0。

两组损失均为：

```text
L = 当前动作Flow-Matching损失
  + 0.1 × mean(当前条件余弦距离, 未来条件余弦距离)
```

唯一主要实验变量是“动作是否读取预测未来节点”。
这不是物理动力学模型、逆模型、完整未来轨迹生成或持续在线时间线修订。

## 2. 已完成的本地验收

机器可读：

```text
/home/zhengguantian/WoG_fresh/whenami_timeline_c/LOCAL_VERIFICATION.json
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_c/
  final_cpu_tests.log
  real_smoke_01/RESULT.json
  train27_data_audit.json
```

- 18项测试通过，包括已有时间查询等价性、截止时间隔离、未来输入拒绝、读入比例、
  初始两组输出一致、未来节点的动作梯度路径、teacher detach、launcher检查。
- CPU4-rank嵌套FSDP验证A→C model-only加载、C的Adam/read_scale保存恢复、下一更新逐项一致。
- 真实A50k加载：学生1548项A状态＋新read_scale；老师1548项，无遗漏主体状态。
- 真实OXE样例反传：VLM视觉/投影/LLM、QFormer、Timeline、DiT都有有限非零梯度。
- 动作loss到预测未来的直接梯度：aux为0，read大于0。
- 完成一次真实BF16主体SGD更新；teacher不变，篡改隐藏未来图像/指令不改变学生输出。
- 27个TRAIN域各检查一条合法固定未来间隔样本，不需要重新准备OXE。

**真实320-rank完整模型Adam训练/保存/恢复仍需下面的集群验收。**
本机SGD smoke不替代这项，也不代表C的效果已经得到验证。

## 3. 提交：两个独立任务，每个20节点×16卡

两条命令各提交一个20节点任务，共640PPU。
不要把两条放到同一个40节点任务里，也不要将平台的WORLD_SIZE手工设置成320。
沿用此前成功的DLC容器/挂载/MASTER_ADDR/RANK配置。

### 任务1：未来仅作辅助监督

```bash
C_VARIANT=future_aux PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_c_320.sh
```

### 任务2：未来预测直接参与动作

```bash
C_VARIANT=future_read PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_c_320.sh
```

`PHASE=run`会自动执行：

```text
320-rank小型拓扑/保存恢复证明
→ 完整C训练至100步、保存、因果DEV评测
→ 新进程恢复至120步、保存、因果DEV评测
→ 检查梯度、教师身份、读入比例、checkpoint与恢复
→ 全部通过才继续到20k
```

任一步失败就停止，不绕过验收硬跑20k。
只想跑验收时使用 `PHASE=admit`，之后有效120步checkpoint可用 `PHASE=full` 继续。
同一任务不可重复使用已有RUN_ID；需要重提时明确使用新RUN_ID，或先核查已有恢复点。
提交后不要修改C源码或被记录的依赖，否则source guard会阻止恢复。

## 4. 固定配方

|项目|值|
|---|---|
|学生与老师初始化|同一个Timeline-A50k|
|老师|冻结|
|学生训练部分|VLM视觉/投影/LLM、QFormer、时间查询、完整DiT|
|冻结视觉基础模块|独立逐帧DINO/Wan|
|学生输入|一张历史＋当前|
|训练真实未来间隔|4个数据帧索引，不是统一物理秒数|
|历史间隔|1–12，受轨迹长度约束|
|数据|原全量OXE27 TRAIN，筛选能提供该时距与一致指令的片段|
|global batch / micro|320条轨迹 / 每卡1条|
|Flow Matching draws|2|
|更新数|20,000|
|主体与时间模块峰值LR|均1e-5，warmup1000后cosine|
|read组渐进接入|前1000步0→1|
|保存/评测|2500/1000步，另有100/120步验收|
|分片与重计算|hybrid_zero2、LLM activation checkpointing|

固定future-gap会改变相对旧B的样本筛选，A初始化也从10k变为50k。
因此C与旧B的差异不只有读出结构；最干净的本轮比较是future_aux与future_read。

## 5. 输出与检查

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_c/training/
  timeline_c_future_aux_A50k_h4_320_20k_01/
  timeline_c_future_read_A50k_h4_320_20k_01/
```

关注 `train.jsonl` 中：

- action_fm、current_alignment、future_alignment；
- read_scale：aux恒0；read按step/1000升至1；
- 六组梯度与teacher无梯度检查；
- progress.json / tqdm 的速度、ETA和显存；
- COMMITTED.json、RESUME_VERIFIED.json、development/step-*/RESULT.json。

DEV只读取过去＋当前，预测未来查询固定为当前+4，不读取DEV中第三张未来图像。
条件误差降低不等于动作或任务成功率提高；最终仍需同协议SIMPLER和更多独立训练seed。

## 6. 研究定位与已有工作

这是为检验原始Timeline思路而做的下一版在线学生，不宣称“预测未来/门控注入”本身首创。
已核查WoG、MTM、HiF-VLA、PFD；本轮还读取了2026-10-01发布的ATI-VLA相关方法，
其共享码本动作对齐与预测latent自适应注入是直接近邻。
我们的时间查询/动作条件语义差异仍需强对照和闭环收益支持，不能仅凭模块组合确认新颖性。

原文档案：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_followup_20261006/literature/ATI_VLA.txt
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_novelty_recheck_20261005/
```
