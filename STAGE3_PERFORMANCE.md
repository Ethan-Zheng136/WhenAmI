# Stage3运行监控与吞吐检查（2026-10-02）

**更新：独立优化训练器已实现于`stage3_perf/`，提交说明见
`stage3_perf/SUBMIT.md`，入口为`run_stage3_perf_320.sh`。包括平台日志内
tqdm/ETA、micro4×accum1、节点内16卡混合分片、取消重计算及CPU预取。
下文的“建议实验”是实现前记录；多节点提速幅度仍需新任务前100步实测。**

当前任务：`outputs/stage3_joint/oxe27_forecast_320_50k_01`，320卡，
micro-batch2、accum2、global1280、4次FM抽样、50k步。

## 已实现：不中断当前任务的只读进度/ETA

```bash
python /home/zhengguantian/WoG_fresh/monitor_stage3.py
# 单次显示 / JSON快照：
python /home/zhengguantian/WoG_fresh/monitor_stage3.py --once
python /home/zhengguantian/WoG_fresh/monitor_stage3.py --json
```

在登录节点独立终端运行，不占PPU。只读recipe/train.jsonl/complete.json。
滚动最近100步（不足100则全部）；ETA不包含保存和开发评测，不能冒充准确完工时间。
长时间无日志会提示检查，不能单凭日志静默判定死锁。Ctrl-C只停止监控。
不是给已启动训练进程注入tqdm，也不会自动出现在当前平台的训练输出里。
脚本/测试位于stage3_scale外，未修改正在训练的源码指纹，6项CPU测试通过。

13:44 UTC快照：59/50000，平均11.67秒/步，最近20步12.79秒/步；
线性外推约6.8—7.4天纯训练，距离首个2500步保存约8小时，均非承诺。

## 实际工程差异：不应宣称与WoG完全相同

对照本地`WoG_v2/refs/code/WoG/training/strategies/{fsdp,base_strategy_wog}.py`：

- WoG训练循环已有tqdm/metrics；当前`stage3_scale/train_joint.py`是独立循环。
- WoG实际策略映射为HYBRID_SHARD / _HYBRID_SHARD_ZERO2；
  当前为默认全world的FULL_SHARD，320卡参与分片。
- WoG启用activation checkpointing时只选择LLM层；
  当前同时覆盖LLM/视觉/BERT/DiT适配块/R/K Transformer层。
- WoG在线RLDS流水线处理并行数据；当前从A产物同步取batch和解码，
  尚无专用异步预取路径。A避免重复解析RLDS，但不等于读取无开销。
- 学生VLM、Past QFormer和DiT来自原有权重/模块；R/K、时间输入、
  目标函数、data loader、恢复格式和训练调度均有独立改动。
- WoG此处的动作loss在训练时读取真实Future条件，另以cosine目标学习未来预测；
  当前forecast用预测Future修正Past条件后参与动作预测，
  真Future只提供辅助目标，loss为action FM +0.1标准化Future MSE。

这些是代码事实；哪个差异主导速度，现有总步时日志尚不能证明。
监控的平均显存占用也不等于每卡峰值显存，不能据此保证更大batch不OOM。

## 建议的独立性能实验（未实施，勿当成已验证加速）

先保持模型、50k更新预算与global1280，分别测稳态秒/步及峰值显存：

1. 320卡micro4×accum1 对比 当前micro2×accum2：
   减少分批执行/collective轮次，但不减少每更新观察数，不保证两倍提速。
2. 减少activation checkpointing范围，测用显存换少重算是否划算。
3. 节点内16卡分片、节点间复制的混合分片，对照当前全320卡分片。
4. 单独计时data、frozen visuals、student forward/backward、collectives、
   optimizer；必要时做异步预取。不能只凭GPU利用率图判断FLOPs效率。

上述需独立benchmark、有限梯度/保存恢复验证；本次没有改变正在跑的任务。
不应先假设640卡会线性提速，也不能仅增加卡数就保持batch预算不变。

## 恢复限制

当前v2 checkpoint的Adam状态按rank保存，要求相同world、参数布局及recipe。
直接320→640不支持精确resume；micro/accum/分片方式变化也不能绕过原门控。
如需迁移，先实现并验证明确的转换流程；仅加载模型不等于完整恢复。
不要为加速静默丢掉optimizer/RNG或修改已验收源码。
