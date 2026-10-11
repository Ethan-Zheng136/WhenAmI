# 当前Stage3、Revision原型和候选完整WhenAmI：不能混为一张图

2026-10-03。本文依据当前源码及已经保存的诊断结果。
不修改现有Stage3或接手另一agent的eval/部署工作。

## 1. 新颖性结论：不是“没人做过”

**新观测修订过去状态的思想是已有的smoothing，不是本项目首次提出；
`b[k|t]`双索引也是状态估计中自然的表示，不是单独的创新点。**

应正面比较的研究：

- Bayesian smoothing / fixed-lag smoothing：同一过去状态的估计随后续证据更新。
- TD-VAE（2018/ICLR2019）：belief、local smoothing和跨时间点状态预测。
- HiF-VLA（CVPR2026）：历史运动、未来运动和动作联合建模。
- RB-VLA（2026）：action-conditioned递归belief与动作策略。
- ChainVLA（2026）：可修订的跨query执行状态、进程上下文和未执行motion tail。
- 状态估计/规划统一的factor-graph工作也表明，“同一模型前后两段”本身不是全新概念。

不能因为没有查到逐模块完全相同的代码，就宣称“首次”。
当前原型也没有足够的新算法/结果来支持已经形成论文贡献。

**待验证的具体方向**：
在可部署VLA中显式保留事件身份和原始/修订版本，用新证据纠正对先前事件的解释，
并证明该机制在相同观测、动作历史、参数和计算条件下提高当前决策，尤其失败恢复。
还需要机制本身、监督设计及实验，而不只是更换术语。

一手来源：

```text
Särkkä smoothing lecture:
https://users.aalto.fi/~ssarkka/course_k2016/handout7.pdf
TD-VAE:
https://arxiv.org/abs/1806.03107
HiF-VLA:
https://openaccess.thecvf.com/content/CVPR2026/html/Lin_HiF-VLA_Hindsight_Insight_and_Foresight_through_Motion_Representation_for_Vision-Language-Action_CVPR_2026_paper.html
RB-VLA:
https://arxiv.org/abs/2602.20659
ChainVLA:
https://arxiv.org/abs/2608.02326
Frank Dellaert, factor graphs and world models（作者技术说明，非新实验证据）:
https://gtsam.org/2026/04/21/factor-graphs-and-world-models.html
```

## 2. 图A：当前大模型Stage3（已训练，独立保留）

```text
历史图像 + 当前 ──> 冻结DINO/Wan ──> Past QFormer ──> H_P、C_P
                                                        │
当前图像 + 指令 ──> VLM ──> z ───────────┐               │
                                       │               │
                          H_P、z ──> R ──> C_F_hat      │
                                       │               │
                          H_P、z ──> K <─┘               │
                                    │                  │
                                    └─ delta_C_P ──────┤
                                                       v
                                              C_P + delta_C_P
                                                       │
                                         z ──────────> DiT ──> 动作

训练监督：
真实Future + 当前 ──> 冻结Future编码 ──> C_F_target
                                           │
                           对齐C_F_hat <────┘

Loss = action FM + 0.1 * Future alignment
```

当前修正的是**当前动作条件**，由自己预测的Future驱动。
不存储同一事件在不同evidence cutoff下的多个解释，没有独立后向状态更新器。
R输入不包含候选未来动作，不应把它叫作一般的动作条件物理后果模拟器。

## 3. 图B：现在已经写出的Revision小原型（不是8B）

```text
上个filtered状态 + 两锚点间已执行动作序列 + 实际Δt
                     │
                     F
                     │
                  预测当前观测 ───> 编码 ───┐
                                          │
真实当前观测 ──> 编码 ─────────────────────┤
                                          v
                                  innovation = 实际-预测
                                          │
原始事件记录 b[k|k] ────────────────┐      │
此前修订记录 b[k|previous] ────────┼──> B <┘
已执行动作/锚点年龄 ───────────────┘     │
                                       v
                                   b[k|t] bank
                                       │
当前filtered状态 ────────────────> attention读取 ──> 当前离散决策
```

对于每个旧事件节点，B做残差更新；它不是逐边执行一个可逆动力学算子，
也不是已经实现了严格Bayes smoother或校准的概率belief。
`state_head`在合成任务中读出二分类标签，不等于整个hidden是概率分布。

**重要实现边界**：

- 下一时刻F仍从`state.filtered[:, -1]`起步，不是用修订bank重算整个filter轨迹；
- 修订bank进入当前决策的attention读出；
- 暂无有界长期memory/事件选择与遗忘策略，不能声称已解决无限时长效率；
- 观测是12维合成特征，hidden64、参数127,890；
- 决策是confirm/retry/recover/transport，不是机器人7维动作；
- 没有加载大模型checkpoint，没有接入真实视觉/语言编码器或主模型DiT；
- 未实现尚未到来的Future teacher对该新分支的训练；
- 当前真实OXE数据接口与合成模型诊断是两项独立验证，尚未连接成真实训练。

## 4. 两个时间索引究竟是什么

```text
事件时间       k=10                  k=13                 k=16
观测           抓取尝试              抬起/看见结果        搬运/可能滑落

证据截止10     b[10|10]

证据截止13     b[10|13]              b[13|13]

证据截止16     b[10|16]              b[13|16]             b[16|16]
```

下三角可用，上三角在在线策略中不可用。时刻16可以修订10/13，但时刻10
不能知道16的真实观测。相同旧事件身份不能因后续变化而被替换。

比如“10时抓住，16时滑落”不能被改写成“10时没抓住”。
这些事件标签在合成数据中定义明确；真实OXE目前没有相应统一标注。

## 5. 图C：候选接回WhenAmI的方法（尚未接入）

可先采用保守的附加残差接口，不丢掉当前已学动作能力：

```text
原Stage3：
历史/当前/指令 ──> Past + VLM + R/K ──> C_base、z ─────────┐
                                                         │
新增分支：                                               │
多锚点真实观测 + 已执行动作 + Δt                          │
        │                                                │
        F + B ──> 修订事件bank ──> 条件桥 ──> delta_revision
                                                         │
                                  C_base + delta_revision
                                             │
                                  z ───────> 同一个DiT ──> 动作
```

已实现`RevisionConditionBridge`：attention＋零初始化输出，能输出16×32残差。
已测试初始不改变base condition，以及输出层打开后梯度能回到revision。
**图中大模型集成、真实训练目标、完整动作FM联训尚未实现。**

另一条候选是由事件修订替换旧K或旧预测条件路径，而非继续叠模块。
必须与附加同容量通用memory残差比较后选择；不预先认定“叠得越多越好”。

## 6. 真正可能改善什么（是假设，不是已有结果）

| 维度 | 当前Stage3 | 候选Revision |
|---|---|---|
| 修正对象 | 当下动作condition | 有身份的过去任务事件解释 |
| 触发依据 | 历史推得的Future condition | 真实新证据与预测的差异 |
| 时间表示 | 单中心窗口 | 明确event time / evidence time |
| 动作关系 | R/K不读已执行区间动作 | F/B显式读已执行动作序列 |
| 在线记忆 | 本次窗口内表征 | 可修订事件bank |
| 未证实的价值 | Future监督增益 | 修订是否优于同信息同计算的普通memory |

它应作为WhenAmI更完整研究分支，而不是声称当前Stage3的已实现改进已经涨分。
现有VLM、Past和DiT值得继承；新方法不应迫使用户再从零训练所有基础能力。

## 7. 验证程度，按层级分开

1. **数据合同：通过。** 15项测试覆盖prefix因果性、动作可见性、真实Δt、
   重复证据拒绝、流式一致性、梯度、条件桥等。
2. **实数据读取：通过。** 每域1条训练episode，27域原图/动作多锚点读回。
   不是“在27域上训练了模型”，也没有伪造状态标签。
3. **合成可学习性：通过。** 三种子、各800步，区分空抓/抓住后滑落/保持。
4. **相对普通memory的机制优势：未通过/未发现。**
   同容量direct-history重训控制和incremental均达到该简单任务100%，
   关闭revision也不影响当前决策。现有toy已饱和，不能证明新增机制必要。
5. **真实视觉/语言、8B集成、机器人收益：未验证。**
6. **新颖性：有明确近邻，不能宣称首次。**

低维可学习性只说明程序具备做这件事的能力。没有重训控制的“100%”不能当论文成果。

## 8. 下一步的真实研究问题

先确定哪些任务需要修订旧事件，并建立可信状态/后果监督，
再用同信息、同标签、同计算的memory baseline比较。

新颖性应来自有实质差异的修订/监督/决策机制及可信证据，
不是`b[k|t]`记号、一个残差MLP或“hindsight”名称本身。

当前原型的有用之处是已经把信息时序和梯度通路落地，并暴露了普通memory也能解决
简单任务这一事实。应保留该负证据继续研究，不能删掉基线路径使它人为变难。
