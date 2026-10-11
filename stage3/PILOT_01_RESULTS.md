# Pilot 01：本地单PPU完整对照结果

日期：2026-10-02。执行前计划见 `PILOT_01_PLAN.md`。

## 实验确已完成

- 缓存：`outputs/stage3_pilot/cache_pilot_01/`，1024窗口。
- Stage-3训练813窗口；验证211窗口，Fractal 105窗口/27 episodes，
  Bridge 106窗口/28 episodes。训练/验证无重复episode。
- 两组均单PPU、batch=8、LR=1e-4、seed=42、R hidden=256/layers=3，
  条件对齐/action损失权重各1，训练1000步。
- `s3_forward_pilot_01` 03:53:24–03:55:05 UTC；
  `s3_bidir_pilot_01` 03:55:05–03:57:55 UTC。
  含载入、周期验证、checkpoint；非纯训练吞吐。
- 前1000步连续无缺失，无NaN/Inf，固定DiT参数无梯度/版本变化。
- 两组第一步forward alignment和forward action loss逐值相同；
  source cache、初值、forward样本和噪声匹配。
- 两个最终小checkpoint均严格重载通过，connector所有浮点张量有限。
- 推理使用预先指定的最终1000步checkpoint、fp32、同一seed=42，
  完整10点Euler采样，保存所有逐样本预测。
- 原Past/Future/均值参照的预测数组在两组之间逐值一致，样本ID和真值也一致。
- Stage1/2 checkpoint及参考代码未改动，缓存/源代码hash检查通过。

## 主结果：实际动作误差，而不只是条件loss

下面的单/双向R均指：
`Past H/z -> R -> 预测Future条件 -> 固定Future DiT(z_F) -> 动作`。
不读取真实Future观测，但依然不是Stage4的Past融合策略。

| 数据集 | 路径 | r_6 | R²_6 | normalized MSE_6 ↓ |
|---|---|---:|---:|---:|
| Fractal | 原Past策略 | 0.721972 | 0.458424 | 0.039149 |
| Fractal | R仅正向 | 0.656146 | 0.343282 | 0.045012 |
| Fractal | R双向训练 | 0.657755 | 0.346131 | 0.044680 |
| Fractal | Future真实条件 | 0.678920 | 0.379705 | 0.041517 |
| Fractal | Future训练均值条件 | 0.617698 | 0.275532 | 0.050469 |
| Bridge | 原Past策略 | 0.766771 | 0.531804 | 0.046949 |
| Bridge | R仅正向 | 0.770458 | 0.532615 | 0.045693 |
| Bridge | R双向训练 | 0.770094 | 0.531276 | 0.045758 |
| Bridge | Future真实条件 | 0.778003 | 0.537336 | 0.043438 |
| Bridge | Future训练均值条件 | 0.762854 | 0.503793 | 0.047596 |

双向相对仅正向，Fractal MSE降低0.738%，Bridge MSE升高0.142%。

按episode配对bootstrap，5000次、bootstrap seed=20261002，
差值定义为双向MSE减单向MSE（负数更好）：

| 数据集 | 点估计 | 95% percentile interval |
|---|---:|---|
| Fractal | −0.00033214 | [−0.00076098, −0.00002949] |
| Bridge | +0.00006494 | [−0.00013984, +0.00028596] |

Fractal区间在本次固定模型/固定噪声下为负，但效应小；
Bridge区间跨零且点估计相反。**这不是不同训练种子上的稳健性证据**，
不能概括为“逆向已经产生显著稳定收益”，也不应把Fractal说成完全没有改善。

## 验证集FM损失与训练速度

| 指标 | 仅正向 | 双向 |
|---|---:|---:|
| forward alignment | 0.292841 | 0.291223 |
| forward action FM | 0.189226 | 0.189052 |
| forward action FM（均值条件参照） | 0.236993 | 0.236993 |
| forward action FM（真实条件参照） | 0.181665 | 0.181665 |
| 最后100步中位训练秒/步 | 0.05282 | 0.10417 |

双向额外逆向任务约使训练步时翻倍；匹配的是forward更新预算，不是总算力。
不要比较两组total loss绝对值：双向总loss包含额外任务。

## 当前可支持的结论

1. R确实学会了一定程度的条件预测，实际动作优于训练均值条件。
2. 双向与仅正向非常接近；额外逆向任务尚未显示有实质性的跨数据集收益。
3. 预测Future条件这条直接读出路径未全面优于原Past：Fractal明显落后，
   Bridge略好。它不等于原Past在线策略已被改善。
4. 无需据此推倒已有效的Past/Future。现在应讨论/检验怎样让回溯任务
   实质影响在线状态，而不是继续增加同中心条件回归的训练规模。

## 局限

- 一个训练seed和一个推理noise seed；无全量OXE、闭环或Stage4实验。
- 验证集被每100步观测，不是未接触过的最终test；主指标使用预先指定最终1000步。
- 端点预训练可能见过这些episode。
- 不是先前256窗口的评测集；不能将r=0.72等直接与旧表横比。
- R输出条件，不是严格的逐时刻状态；没有动作条件动力学或物理可逆性证明。

## 产物

根目录：`/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_pilot/`

```text
s3_forward_pilot_01/{train,validation}.jsonl
s3_forward_pilot_01/checkpoints/step-001000.pt
s3_forward_pilot_01_eval/results.json
s3_forward_pilot_01_eval/*.npz

s3_bidir_pilot_01/{train,validation}.jsonl
s3_bidir_pilot_01/checkpoints/step-001000.pt
s3_bidir_pilot_01_eval/results.json
s3_bidir_pilot_01_eval/*.npz

comparison_pilot_01.json
comparison_pilot_01.log
```

新增 `stage3/compare.py` 可重复运行配对分析（需新output路径）；
`stage3/test_compare.py` 两项测试通过。
