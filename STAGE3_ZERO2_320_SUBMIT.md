# 320卡进一步吞吐候选：Hybrid ZeRO-2

2026-10-02。用户选择留在320卡，维持global batch1280、每5000步保存。
本次只新增launcher及独立测试，不修改已验收的`stage3_perf/`、`stage3_scale/` Python源码。

## 正式训练入口

申请 **20节点×16卡**，确认旧任务已停止，所有节点使用相同命令：

```bash
RESUME=0 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_perf_320_zero2.sh
```

新目录：
`outputs/stage3_joint/oxe27_forecast_zero2_320_50k_save5k_01`

这是重新从原Stage1/Stage2及ReaderStudy初始化开始，不是续跑旧320已经训练的463步。
A无需重跑，旧权重/日志/结果不删除。新入口不接受旧hybrid配方的resume，
不会静默重置优化器冒充无损续训。

## 配方

- 320卡 × 每卡4 × 累积1 = global1280；50,000 optimizer updates。
- LR2e-5，R/K LR1e-4，warmup1000，4次FM抽样，原模型及loss完全不变。
- `hybrid_zero2`：节点内16卡分片，跨节点复制；在前向后保留完整参数，
  相比普通HYBRID_SHARD减少反向前的参数all-gather，代价是更高峰值显存。
  本机PyTorch2.8源码文档明确支持该行为，旧WoG策略也有此选项。
- 沿用取消activation checkpointing、CPU两batch预取、只解析一次索引、
  去重的分片checkpoint，不为加速减少数据/模型/动作抽样次数。
- 周期保存5000、10000……50000；评测仍每2500步。
- 保留一次100步保存/评测及新进程恢复验证，随后自动继续到50k。
- 内置tqdm、每10步PROGRESS/ETA、显存峰值、每100步分段计时。

## 哪些已经验证，哪些没有

已完成：
- 额外18项CPU信息流/梯度/运行工具测试（包括hybrid_zero2对比未分片全batch的
  梯度平均、clip范数及Adam更新），日志`zero2_gradient_and_runtime_tests_20261002.log`。
  CPU测试设置无CUDA设备，日志有一条PPU/CUDA初始化不可用告警；不把它当成PPU验证。
- 重跑4-rank CPU hybrid_zero2去重保存/恢复，分rank RNG正确、恢复后下一更新逐值相同。
- launcher语法、fresh/resume命令、错误拓扑拒绝、模拟完整运行顺序、
  任一阶段失败不继续、恢复时后续评测结果保留不覆盖，均通过。
- 正式模型/数据preflight门控通过，原模型/训练Python源码hash不变。

未完成：
- 320个不同PPU上的hybrid_zero2吞吐和峰值内存实测；
  CPU正确性测试不证明会快于现有约2.62秒/步。
- 不能承诺提速比例，也不应依据旧HYBRID_SHARD峰值保证新配置不会OOM。

新任务先用同一320设备tiny验证，然后真实100步落盘`FIRST_SEGMENT_COMPLETE.json`，
读取其中`steady_seconds_per_step`和`peak_allocated_gib_max`判断实际收益；
主训练会在正确性检查通过后自动继续，不需要再提交第二个任务。
如果不快或显存不合适，应基于实测再选择，不应连续增加未测选项。

## 新配方同拓扑中断恢复

```bash
RESUME=1 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_perf_320_zero2.sh
```

仅用于这个新zero2任务已有完整checkpoint时。不能改RUN_ID指向旧hybrid/640任务。
评测比保存更频繁，回滚时超出checkpoint的旧评测会移入`replayed_development`保留，
避免重算到2500/7500等未保存点时输出目录冲突。不中断正常运行的任务执行此命令。
