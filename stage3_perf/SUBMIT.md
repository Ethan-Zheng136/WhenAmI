# Stage3 吞吐版本：320卡、新训练输出

## 最新选择：320卡、Hybrid ZeRO-2、每5000步保存

用户决定不换640卡，进一步测试利用显存降低通信的320卡方案。
入口：`run_stage3_perf_320_zero2.sh`，具体说明见根目录
`STAGE3_ZERO2_320_SUBMIT.md`。这是从原初始化开始的新输出；
保持micro4/global1280/50k/LR，复用现有Python实现，新增CPU梯度与恢复检查通过。
是否比原HYBRID_SHARD的约2.62秒/步更快需新320任务实测，尚未宣称已加速。
一次100步保存/评测/恢复后会自动继续完整50k。

```bash
RESUME=0 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_perf_320_zero2.sh
```

## 更新：用户选择640卡主训练、每5000步保存（2026-10-02）

正式入口（不是100步后退出的`640_probe`）：

```bash
RESUME=0 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_perf_640.sh
```

申请40节点×16卡。micro2×accum1，global仍1280，50k步，LR不变；
输出`outputs/stage3_joint/oxe27_forecast_perf640_50k_save5k_01`。
停止旧320任务后提交；原checkpoint、日志保留，A不重跑。
新任务从原始端点/ReaderStudy初始化，不携带旧320的已训练更新。
一次性的100步保存/评测/新进程恢复检查仍保留，随后自动继续到50k。
周期保存点为5000、10000……50000；动作评测仍每2500步一次，
所以2500、7500等非保存点也有评测结果（没有该步权重快照）。
如新640任务中断，使用同一入口`RESUME=1`，不能传旧320的RUN_ID。
已准备和本地检查，不代表集群任务已由助手提交，也不保证640线性加速。

---

2026-10-02。实现位于独立`stage3_perf/`；原`stage3_scale/`和正在运行的
`oxe27_forecast_320_50k_01`不修改。当前版本的320卡实测提速尚待提交，
不能把本地单卡测试或旧B验收当作新混合分片已在320卡验证。

## 一条提交命令

可以停止旧任务后，重新申请 **20节点×16卡**，各节点运行相同命令：

```bash
RESUME=0 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_perf_320.sh
```

输出`outputs/stage3_joint/oxe27_forecast_perf320_50k_01`。
这是从原Stage1/Stage2及ReaderStudy初始化重新训练，不从旧320卡任务迁移。
A无需重跑；原任务中尚未保存的更新不会带入新任务，旧日志/权重不删除。

脚本顺序：
1. 在当前320个物理设备上先运行tiny混合分片保存/恢复验证。
2. 使用真实全量OXE27、真实8B模型跑正式配方的前100步。
3. 保存100步checkpoint，并做现有完整开发集动作评测。
4. 检查完整ckpt、各模块有限非零梯度、动作评测、源码一致性。
5. **退出训练进程并新建进程恢复100步状态**，继续到50,000步。
   恢复失败会退出，不会静默丢弃Adam/RNG。

因此前100步不是丢弃的试跑，也不用再人工提交一个“正式训练”任务。
100步之后出现`RESUME_VERIFIED.json`和step101日志，证明新进程已成功更新。
若tiny或前100步失败，脚本不会继续。不要自动原样重启`RESUME=0`覆盖旧目录。

## 实现改动

- micro-batch **4**、累积 **1**，global batch仍 **1280**；50k步、LR和4次FM抽样不变。
- **HYBRID_SHARD**：节点内16卡分片、相同本地卡号的20个节点之间复制同步。
  不是全320卡做参数分片，也不是恢复成8卡配置。
- 默认不做activation checkpointing：用显存减少重计算。
  已在本机真实8B、micro4、4次FM的BF16-master无Adam反向测试通过，
  峰值约33.1GiB；这不是多机FP32-master+Adam内存证明，提交后记录实际全rank峰值。
- 可以用新RUN_ID和`ACTIVATION_CHECKPOINT=llm`开启更保守的LLM重计算；
  不可在同一任务resume时改变该参数。
- 1个CPU线程、最多2个batch预取，不在worker线程调用GPU。
  沿用同一采样/增强/动作mask函数，未改变数据范围、归一化或科学目标。
- 313MB索引每rank只解析一次，构建域索引由27次全遍历改为一次遍历。
- 模型只保存第一组分片；Adam只保存16份本地分片，每个全局rank仍独立保存RNG。
  避免20个节点将相同Adam状态重复写20遍。
  同拓扑精确恢复，不支持直接改变320→640。

默认只训练forecast；模型/loss完全复用原`stage3_scale/model.py`。
不是更小模型，不是减少训练数据，也没有减少global batch冒充加速。
合并微批会改变具体随机数分组，不宣称与旧执行顺序逐bit相同。

## 进度、ETA与性能诊断

直接出现在平台训练日志中：

```text
Stage3: ... 123/50000 [..., L=..., lr=..., ...s/step ETA_train=...]
PROGRESS 123/50,000 (...) | ...s/step | ...obs/s
         | ETA_train ... | ETA_total ... | next_ckpt ...
         | loss ... action ... | LR ... R/K ... | peak_max ...GiB
PHASE_BEGIN save/eval ...
PHASE_END save/eval ... elapsed=...
```

- tqdm + 每10步永久文本ETA，平台吞掉回车时仍有可读进度行。
- `ETA_train`依据最近100步；`ETA_total`待保存/评测有实测后增加其预计开销，
  并在100步自动恢复后保留这些历史。都只是估计，不包含未知故障停机。
- `train.jsonl`记录全rank最大allocated/reserved峰值。
- 首步/恢复首步/每100步采样分段计时：
  数据等待、CPU→设备、冻结视觉/teacher、学生前向、反向（包含通信）、梯度裁剪、optimizer。
  采样时显式同步，有额外开销；不能将每项跨rank最大值简单相加当总步时。
- `FIRST_SEGMENT_COMPLETE.json`给出第11—99步非profile步的平均速度。
  这是第一次实际320卡吞吐判断点，不能预先保证比旧版快几倍。

## 同配置中断恢复

```bash
RESUME=1 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_perf_320.sh
```

仅用于该新任务已有完整checkpoint且未完成的情况，不是恢复旧`stage3_scale`任务。
脚本和Python源码被recipe绑定；开始后不要原地修改代码或模型/数据配方。
若原任务停在100步保存后但评测中断，需检查并补做该评测，而不能假定已经完成。

## 本地验证边界

- CPU4-rank：full/hybrid/hybrid_zero2，保存恢复后下一更新逐值相同；
  hybrid验证复制的分片一致、Adam只保存2份而非4份，每rank RNG恢复独立。
- 单PPU真实8B：micro4、llm/none重计算设置均完成反向，7个训练模块梯度有限非零。
  为本地内存测试使用BF16 master且无Adam，不是完整集群训练证明。
- 单PPU tiny：实际新trainer的两步训练→保存→新进程恢复→四步训练、
  tqdm/ETA和阶段日志。评测在此测试中明确用synthetic占位；它不证明真实动作效果。
- 全量OXE产物16个实际观察：同步/预取读取的图像增强、动作、mask、身份等逐值一致。
- 单元/原模型回归测试记录于`outputs/stage3_joint/perf_regression_final_01.log`。
- 主入口检查`LOCAL_VERIFICATION.json`的源码指纹，不把旧B形状验收冒充新性能验收。
