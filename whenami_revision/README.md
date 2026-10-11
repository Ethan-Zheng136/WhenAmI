# WhenAmI Revision：独立新方法探索

2026-10-03。**本目录不替换当前Stage3，不负责NVIDIA部署/主表评测。**
评测与新服务器准备由另一agent承担；本目录只建立新的事件时间—证据时间模型、
真实数据输入合同和研究诊断。

**追加进展（2026-10-03）：** 已完成`PLAN_03_DELAYED_EVIDENCE.md`规定的
长历史/延迟/噪声/无关证据实验，六模型各1200步，20项测试通过。
当前决策与direct-history基本持平；旧事件概率的长历史稳定性有迹象，
但无关证据的扰动更大，不支持直接宣称新方法有效。
详见`RESULTS_03_DELAYED_EVIDENCE.md`。
新颖性边界见`NOVELTY_STATUS_20261003.md`：不能声称具身领域无人做过。

研究依据：`../WHENAMI_DEEP_RESEARCH_20261003.md`。阶段编号不改：
当前Stage3仍可直接产出部署策略，新方法是独立的研究候选，不是再加一道必须的训练关卡。

## 1. 已经实现的最小完整接口

```text
同一episode的实际锚点：       k0 ───── k1 ───── k2 ───── t
区间动作：                 u[k0:k1) u[k1:k2) u[k2:t)

forward filter:
  上一个filtered状态 + 完整区间动作序列 + 实际Δt → 预测当前观测
  真实当前观测到来 → 更新当前filtered状态

incremental revision:
  新证据与预测之间的innovation → 逐次修订旧锚点bank b[k|t]

decision:
  当前filtered状态 + 修订bank读出 → 当前决策
```

- `b[k|k]`保留用于对照，`b[k|t]`保存修订后的版本，事件身份不改变。
- 后向更新器不是物理转移的函数逆，没有硬加cycle可逆约束。
- 模型接口不能接收未来图像、状态标签或目标动作。
- 只接收到当前为止的观测与严格早于当前的已执行动作。
- 动作输入保留区间序列，不做“旋转/增量求和就是精确合成”的假设。
- 使用真实frame-index差；频率未知时不伪造秒数。
- 时间采用相对差值，不输入episode总长/归一化进度。
- 逐步流式运行与整段前缀回放一致；重复同一timestamp的证据直接拒绝。
- 增加一个`direct_history`控制：同容量、同初始化、同旧节点计算量，
  用当前完整filter hidden直接读取旧事件，不递推旧revision。

**这些是程序接口和可学习原型，不是已验证的8B策略或机器人方法增益。**

## 2. 文件

| 文件 | 作用 |
|---|---|
| `contracts.py` | 事件锚点、cutoff、区间已执行动作、与标签分离 |
| `raw_data.py` | 读取现有A的OXE分片，构建真实多锚点前缀 |
| `model.py` | filter/revision/当前读出原型及可选条件残差桥 |
| `objectives.py` | 合成诊断的决策、事件状态和预测目标 |
| `synthetic.py` | 共享前缀的空抓/抓住后滑落/抓住保持三分支 |
| `train_diagnostic.py` | 固定预算CPU可学习性诊断，自动保存源码快照 |
| `test_revision.py` | 时间、可见性、梯度、流式一致性及桥接测试 |
| `audit_raw.py` | 27域真实训练轨迹的数据合同审计 |
| `PLAN_01.md` | 第一轮实验前固定计划 |
| `PLAN_02.md` | 看到第一轮限制后追加的同容量重训控制计划 |
| `RESULTS_20261003.md` | 本轮结果及限制（单独记录） |

## 3. 真实数据接口与当前不能做的事

`RevisionEpisodes`只读取完整OXE27训练索引，不打开dev或sealed数据。
`prefix()`只取不晚于cutoff的图像，逐区间暴露`actions[previous_anchor:anchor]`；
`action_labels()`单独返回当前16步动作标签及有效性mask。

`EvidencePrefix`包含身份和绝对frame index用于审计，
`tensorize()`只把相对时间与合法观测/动作张量交给模型，避免把审计metadata喂成捷径。
早期重复边界帧合为同一个真实事件节点，不伪造多次独立观测。

如果要预测事件k的动作，evidence cutoff必须等于k。后见状态`b[k|t] (t>k)`
可能已经见过`u[k:t)`，因此不能再把预测该区间动作的好成绩当作无泄漏策略能力。
它可用来监督/分析事件状态，并辅助**当前t**的合法决策。

目前没有为真实OXE伪造“抓住/空抓/滑落”标签；原A没有保存这些明确标签。
真实模型训练前还需落实：

- 视觉/任务编码器接入（当前小原型用低维观测，不是VLM）；
- 语言/proprio等输入合同，按所有baseline可见信息公平设置；
- OXE无状态标签时的合法训练目标；
- 有可信状态的仿真分支数据，或经验证的独立后见teacher；
- 同信息、同预算memory与forward对照；
- 独立的多机训练配置和保存恢复验收。

## 4. 运行研究测试

原服务器环境：

```bash
cd /home/zhengguantian/WoG_fresh
export PYTHONPATH=/mnt/pfs/zhengguantian/WoG_v2/pylibs-train:$PWD
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -m pytest whenami_revision/test_revision.py -q
```

真实训练数据读回审计（输出必须用新路径）：

```bash
python -m whenami_revision.audit_raw \
  --output /mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_revision/raw27_contract_audit_NEW.json
```

低维合成可学习性实验（**不是机器人评测，也不是大训练提交命令**）：

```bash
python -m whenami_revision.train_diagnostic --mode incremental \
  --output /mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_revision/diagnostic_NEW
```

默认三种子、各800步，CPU、batch48、hidden64。使用互不重叠的生成group做训练/诊断。
将输出`recipe.json`与源码快照一起保留，不从多个运行中挑最好的一组。

## 5. 与另一agent的边界

- 不修改其部署接口、模型导出、仿真安装、主表或评测任务。
- 不要求当前Stage3中途增加loss或热改模型。
- 研究若需要有状态标签的受控轨迹，以数据合同交接，不让环境端猜测模型需求。
- 当前主表分数只描述原Stage3；不能混入此原型的合成分数。
- `RevisionConditionBridge`只验证接口和零初始化/梯度开放，没有宣称已接入生产DiT。

## 6. 验证含义

本轮核心目标是把想法变成不会混淆信息时序的可运行程序，并尽早发现旁路。
合成诊断能通过不代表机制成立；若普通history也能解决，要记录这个事实，
进一步检验更有代表性的任务与训练，而不是故意删除baseline已有的信息。
