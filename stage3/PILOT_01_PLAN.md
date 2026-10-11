# Pilot 01：本地单PPU对照计划

2026-10-02。开始运行后、查看正式对照结果前记录。

- 固定cache：`outputs/stage3_pilot/cache_pilot_01/`，
  813训练窗口/211验证窗口，按episode隔离。
- 两组：`s3_forward_pilot_01`、`s3_bidir_pilot_01`，顺序使用同一张PPU。
- 配置：seed=42、batch=8、LR=1e-4、1000更新步、R=256宽/3层、
  alignment/action权重均1、BF16训练。
- 对比主对象：**最终1000步checkpoint**，不在验证曲线上挑最好的一步。
- 主指标：每数据集211验证窗口中的对应子集，
  predicted_future_causal动作的归一化前6维MSE，双向减单向；
  同一噪声seed=42、fp32推理，同一固定Future动作读出。
- 同时报告r/R²、gripper、首步动作误差；训练均值条件、真实Future条件、
  原Past条件作为参照。逆向动作读出只作oracle诊断。
- 按episode配对bootstrap区间，只表示本次固定模型/固定噪声下的样本不确定性。
- 单向/双向forward输入批次、初值、噪声和更新次数相同；
  双向有额外反向任务，**不是等算力对照**。
- 只有一个训练seed，且验证集被周期性查看，不声称最终泛化/论文结论。
- 不修改Stage1/2权重，不启动外部集群任务。
