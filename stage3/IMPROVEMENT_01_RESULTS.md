# Stage3 改进第一轮：诊断、动作优先对照与 WoG Stage II 基线

2026-10-02。本轮已完成，不修改 Stage1/2 checkpoint 或参考源码。
预先计划见 `IMPROVEMENT_01_PLAN.md`；下一版架构候选见 `NEXT_DESIGN.md`，
**该架构仅写成设计，尚未实现/验证**。

## 1. 先诊断，不先猜测“梯度打架”

在 Pilot01 双向最终1000步 checkpoint 上：

- 验证集211窗口：分别交换来源 H、来源 z，固定目标 reader 的 z 与动作。
  donor 仅来自同数据集、不同训练episode，优先匹配语言。
- 每数据集16个训练batch（batch8）：FP32测共享blocks/final_norm/query_position
  上的方向梯度及 action/alignment 梯度。无优化器、无权重更新。

| 训练梯度诊断 | Fractal | Bridge |
|---|---:|---:|
| 正/逆 total 梯度余弦均值 | .00317 | .00633 |
| 正/逆 action 梯度余弦均值 | .00227 | .00201 |
| 正向 alignment/action 梯度范数比均值 | 3.40 | 3.19 |

在本次checkpoint的抽样训练批次上，两个方向近乎正交，而非强烈反向。
这与“共享参数但基本各学各的”一致，但不能单凭局部梯度证明成因。
对齐监督的梯度比动作监督大，值得作单一权重对照。

| 验证 action FM（不是实际采样动作MSE） | Fractal | Bridge |
|---|---:|---:|
| 来源 H/z 完整 | .163178 | .200498 |
| 仅交换来源 H | .162220 | .200943 |
| 仅交换来源 z | .185672 | .222910 |

**解释边界**：
H 中含历史和当前证据，交换 H 不等于只抹掉历史；跨源交换本身也产生
不一致输入。指令匹配比例仅 Fractal49.5%、Bridge3.8%，不能把这项试验
当作严格“只换历史”的因果证据。它只是提示进一步做静态/倒序历史控制，
不能据此宣布历史无用。

## 2. 一个最小改动：alignment 1 → .1，action 保持1

不搜索更多权重，不改变网络结构、LR、数据、batch、种子或更新次数。
单向和双向均从相同初始化训练1000步，仍使用813训练窗口与211验证窗口。
主评测用预先固定的最终1000步checkpoint，FP32动作采样、noise seed42。
双向仍增加计算，不是等算力对照。

实际UTC时间：
- forward训练：04:15:46—04:17:24；
- bidirectional训练：04:17:24—04:20:09；
- 两组动作评测：04:20:09—04:24:52。

## 3. 缺失的 WoG Stage II 对照已补齐

checkpoint：
`ckpts/WoG-A/checkpoints/50k-1229.pt`，config中 `train_venc=False`。
严格加载在线使用的 action_model、projector、LLM、vision 四组权重；
模型 `whenami_stage=0` 且 DiT无per-attention。

只读取当前图像+语言，**不传入历史、真实未来或预测未来条件**。
保留原预处理/VLM推理；动作采样用与 Stage3相同的逐窗口seed和10点Euler网格。
先按WoG-A自身统计反归一化，再用共同的Past统计报告动作误差。
评测于04:24:53—04:27:46完成。

该checkpoint与我们的端点训练预算不同，是现有模型的性能参照，
不是消除了预训练/训练步数差异的架构对照。

## 4. 实际动作结果

主指标：整个16步动作块前6维 normalized MSE，越低越好。

| 路径 | Fractal | Bridge |
|---|---:|---:|
| 原 Past | **.039149** | .046949 |
| WoG Stage II / WoG-A | .039541 | **.038758** |
| 真实 Future（Stage I特权参照） | .041517 | .043438 |
| 旧单向：align=1 | .045012 | .045693 |
| 旧双向：align=1 | .044680 | .045758 |
| 新单向：align=.1 | .043467 | .046675 |
| 新双向：align=.1 | .043403 | .046579 |

新双向相对旧双向：Fractal误差−2.86%，Bridge +1.80%。
新双向相对同设置单向：Fractal −.147%，Bridge −.206%；
episode配对bootstrap的两个区间均跨零：

| 新双向 MSE − 新单向 MSE | 点估计 | 95%区间 |
|---|---:|---|
| Fractal | −.00006404 | [−.00045447, +.00025414] |
| Bridge | −.00009610 | [−.00026928, +.00006439] |

降低alignment权重并未带来稳定的跨数据集改善，也未使逆向监督变得有价值。
不替换默认参数，不将这轮标成“Stage3已改善”。

第一步动作也是重要的次指标：Fractal新双向 .045894，旧双向 .045363，
原Past .036345。即使整个动作块误差有所下降，也不能据此声称在线执行改善。
完整r、R²、首动作与gripper指标都在机器可读结果中，未只保留有利指标。

## 5. 可以得出的结论与下一步

1. 补齐WoG-A后，不能笼统说“只有WoG好”：这批Fractal上原Past也有价值；
   Bridge上WoG-A的点估计更好。样本小、预算不同，不作全局架构排名。
2. 此轮不支持继续调大R或简单增加逆向权重。
3. 更值得检验的是：**保留Past动作路径，令未来预测通过一个在线使用的
   回溯校正模块影响当下决策**。设计与必要普通适配器对照见 `NEXT_DESIGN.md`。
4. 新设计不能直接算成功；必须先证明真实Future校正比普通Past适配器强，
   再证明预测Future与回溯监督的增量价值。

## 6. 边界与审计

- 只有一个训练/采样种子；bootstrap不覆盖种子方差。
- 这是已经看过的验证集上的探索，不是锁定方法后的新test。
- 验证episode与R训练episode隔离，但可能出现在冻结端点预训练中。
- 所有固定Past/Future/均值参照在四次评测中预测数组**逐值一致**，
  对照最大绝对偏差0，样本ID、时间和动作标签一致。
- 两个训练均1000步无中断；冻结reader参数/梯度检查通过；
  最终checkpoint严格重载，实际动作预测有限。
- 21项测试通过，包含真实CPU双进程DDP、诊断donor隔离、共享梯度范围、
  当前帧选择及WoG-A配对采样与原Euler算法的一致性。
- 未做闭环；没有实现Stage4，也没有实现 `NEXT_DESIGN.md` 中的K模块。

## 7. 产物

根目录 `/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_pilot/`：

```text
diagnose_bidir_01/{results,gradient_batches,intervention_donors}.json
s3_forward_actionprio_01/{train,validation}.jsonl
s3_forward_actionprio_01/checkpoints/step-001000.pt
s3_bidir_actionprio_01/{train,validation}.jsonl
s3_bidir_actionprio_01/checkpoints/step-001000.pt
s3_{forward,bidir}_actionprio_01_eval/{results.json,*.npz}
wog_stage2_pilot_01_eval/{results.json,*.npz}
comparison_actionprio_01.json
improvement_01_comparison.json
improvement_01_tests.log
```

新增脚本：
`diagnose.py`、`evaluate_wog_stage2.py`、`compare_improvement.py`、
`test_diagnose.py`。没有修改原训练/评测入口或已有checkpoint。
