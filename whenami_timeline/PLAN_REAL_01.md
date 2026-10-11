# Real Timeline 01：推进真实图像—真实动作，不等待主训练

2026-10-03。用户指出前几轮停留合成实验，timeline仍未兑现。
本轮明确交付真实OXE多锚点模型训练链路，不再重复合成分类来代替机器人动作验证。

## 不改动范围

- 不修改`stage3_perf/`、`stage3_scale/`、`stage3/`、原权重、A产物和封存数据。
- 不接手另一agent的主表、仿真环境或模型导出。
- 全部新增代码在`whenami_timeline/`，产物在`outputs/whenami_timeline/`。
- 现有Stage1/Past权重足够作为独立研究初始化，不必等当前50k结束。

## 实际图

每条真实轨迹抽3个实际时间锚点，保存真实Δt和区间动作序列。
每个锚点分别只读自己的历史/当前/指令，冻结原DINO/Wan/VLM/Past得到H/C/z。

```text
H[k0],z[k0] → filter[k0] → F(actions[k0:k1), Δt) → prior[k1]
H[k1],z[k1] → 真实证据innovation → filter[k1]、revision[k0|k1]
H[k2],z[k2] → 真实证据innovation → filter[k2]、revision[k0,k1|k2]

每个时间点的合法状态 → 残差桥 → 同一个冻结原Past DiT → 该时间点16步真实动作FM
```

动作loss为所有合法锚点等权平均，padding按原协议mask。
只用`b[<=k|k]`预测a[k:k+16]，不能用已见a[k:t)的`b[k|t]`去预测旧a[k:k+16]。
R/F转移的辅助目标是下一锚点的冻结Past条件，目标detach，
不把其视为完美物理状态，不构造未经验证的抓取状态标签。

## 候选与控制

- `independent`：各锚点独立Past表征+同一小残差读出，无跨节点信息。
- `forward`：只做因果filter和跨锚点动作条件预测。
- `direct_history`：同容量MLP/attention直接读因果历史，不累计旧事件修订。
- `revision`：由当前真实证据innovation做相关性门控的旧事件残差修订。

revision/direct-history同参数、同信息、同更新时间、相同bank访问次数。
全部共用同一原始Past DiT和基础特征，不以额外标注/更大动作头制造收益。
关掉gate的控制留到后续独立计划；本轮不声称新gate解决了证据归属。

## 固定本地验收

1. CPU测试信息边界、Δt、实际动作序列、全部锚点动作loss、梯度及控制容量。
2. 读取27域各1条轨迹，实际冻结DINO/Wan/VLM/Past前向，产出27×3锚点。
   只在TRAIN-role episode中取样，短轨迹使用合法锚点，不伪造多个未来证据。
3. 对同一真实特征集、同一seed42、同一采样序列和FM噪声，
   每种模式各8步、batch2条轨迹、每锚点2次FM抽样，lr1e-4。
   此27条固定诊断集只用于接线/数值/保存恢复验证，不是泛化评测。
4. 基础网络冻结，timeline参数FP32、reader BF16计算；不训练8B optimizer。
5. 首次残差桥零初始化，第一步与独立原Past条件一致；
   输出层开放后检查仅action loss到F/B/桥的真实梯度。
6. 单独检查改变未来锚点不改变早期条件，以及同一模型保存加载一致。

任何8步loss改善都不能当作方法胜利。这次验收的价值是：
**已经有真实图像、明确多时间节点、同一动作读出、合法逐节点动作训练**。

## 正式研究实验（需要提交，不在本地偷偷扩大）

- 在现有A产物上流式采样真实轨迹，多锚点逐步编码，不是无限重放上述27条。
- 默认2节点×16PPU，timeline DDP，冻结基础策略每rank1份；
  global64条轨迹×3锚点，2次FM。完整OXE27域权重保持原mix。
- 独立内部开发划分来自原train episode哈希，与主线dev/sealed不重叠；
  原基础预训练可能见过，不能称整个学习历史无污染。
- 对照每组5000步；目标先验证真实条件下作用，再决定联合解冻。
- 完整研究runner需数据/梯度/保存恢复验收，本轮不把未经验证的集群命令称为可提交。
