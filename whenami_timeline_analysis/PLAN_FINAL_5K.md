# Timeline最终5k权重：真实动作采样检查（执行前固定）

2026-10-03。两组5000步已完成，内部action FM分别为direct_history 0.121729866、
revision 0.121583804。这不是动作采样结果，不据此选择checkpoint或调参。

## 本轮固定协议

- 固定两组最后5000步checkpoint，不挑内部dev最佳步；
- 原始Past作为不加timeline参照，不是等预算重训控制；
- 三个主要输出：Past、direct_history、revision；
- 一个额外诊断：相同revision权重，推理时relevance gate强制为0，
  即取消旧节点残差修订但保留当前filter和历史读取；不是重训baseline。
- 使用现有A原TRAIN中的timeline internal_dev角色，不打开主线dev或sealed test；
- 27域，各取至多16条不同episode，固定选样seed20261003，最多512候选；
  若域内样本不足如实报告实际数，不重复episode补齐，不依赖模型输出来选样；
- 每条三个合法真实锚点，实际历史/当前图像和指令经过同一冻结Backbone；
- 用采样seed42/43/44，从FP32标准Gaussian噪声开始，9个Euler区间，
  BF16共享DiT计算，动作状态FP32；无专家动作加噪输入；
- 每锚点初始噪声由dataset/episode_key/真实frame_index/seed决定，所有模式相同；
- 各点只读取当时合法输入，未来锚点特征不能影响早期条件。

## 指标与统计

- 主指标：每个锚点按有效动作mask计算前6维归一化动作MSE，
  预测按既有动作接口截断到[-1,1]，unused dims按原统计置0；
- 同时报未截断MSE，避免裁剪掩盖异常动作；
- MAE、首动作MSE、有效动作位置夹爪准确率；
- 各锚点分别报告，域内episode等权，主聚合是27域macro；
- 每个采样seed单独算误差后平均，不能平均预测形成隐性ensemble；
- 先在同episode内平均三个锚点/三个采样seed，
  再做按域分层的episode配对bootstrap，2000次；
- 只含一个训练seed，不把动作采样seed当多个训练seed；
- expanded internal-dev不等于从未看过的最终test；
  原基础预训练也可能见过这些轨迹；
- 不做主表闭环，不替代另一agent负责的评测。

输出保留完整normalized预测、labels、mask、IDs、anchor times、各次噪声seed、
checkpoint/protocol/source hashes和所有域指标。

本轮不重训、不改变任何已验收Python包、checkpoint、数据或运行参数。

执行补充：首次本机进程在约300条轨迹时收到SIGTERM，未形成完整结果；
保留该日志，原因尚未识别。第二次仅改评测I/O为逐轨迹原子落盘及同协议恢复，
新输出目录，不改样本、模型、seed、指标或选择规则。完整结果前不报告动作效果。
