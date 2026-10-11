# WhenAmI 深入研究：从“未来条件修正”到“可被新证据修订的任务状态”

**日期：2026-10-03。性质：文献、既有设计、源码与证据综合研究，不是已实现的新模型。**

本文不以缩减投稿工作量为出发点，而以做成一个完整、可解释、可证伪的方法为标准。
当前Stage3长训练不改、不停，不解封测试集，不覆盖历史失败结果。
新方法应在独立分支实现，与当前版本公平比较。

本文引用的`[Lxx]`见末尾一手文献清单；`[Cxx]`见源码清单。
原文/提取文本/校验值及本轮微型检查在：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_deepresearch_20261003/
```

阅读范围有深有浅，已明确标记；不宣称穷尽所有相关工作，不宣称候选方案已有性能或新颖性证明。

---

## 0. 执行摘要：这次研究真正改变了什么判断

### 判断一：一个月的结果没有白做，但不能合并成一个过强结论

已有证据分别支持：

- Past能利用实际历史，相对重复当前帧在既有离线协议中有收益；
- 更大Reader Study中，真实Future在Past路径上有额外价值；
- 当前全OXE联合训练的在线动作误差在下降；
- 但早期正逆共享R、简单oracle混训和额外采样loss都没有稳定增量；
- 当前尚无与50k主线同预算的latent、辅助预测但不接入K、闭环等完整对照。

这说明值得继续做的不是“再训练一次看会不会赢”，而是明确哪些信息、在哪个时间点、
以什么方式改变决策。

### 判断二：当前R/K是可用基线，不足以承载完整WhenAmI理论

它是：

```text
Past + current → R预测Future条件 → K修正Past条件 → action
```

不是：

```text
对明确时间锚点的状态进行正向预测、后向证据更新，再形成可校验的当前判断
```

两者应在论文和代码中分开命名。保留当前Stage3的工程产出，不把缺失理论当成代码故障。

### 判断三：真正值得研究的对象应是“同一个时刻、在不同证据到达时间下的判断”

建议引入双时间索引：

\[
b_{k|t}\equiv\text{在时刻 }t\text{ 已获得的证据下，对事件时刻 }k\text{ 的任务状态判断},\quad k\le t.
\]

例如：

- `b[k|k]`：当时以为抓住了；
- `b[k|k+Δ]`：后来抬起手，看到物体没动，修正“刚才抓住了”的判断；
- 修正不仅被记录，还影响当前应“继续搬运”还是“重新抓取”。

这个双时间索引比单一`z_t`更忠于“我当时处在什么状态、现在该怎样行动”。
**它是候选方法的核心接口，不是仅给query加一个时间embedding。**

### 判断四：不要把预测出来的未来当作第二份独立观测

预测未来可以提供计算结构、训练正则或候选计划，但不能凭空消除当前信息中不可辨识的歧义。
更坚实的在线双向路径是：

```text
真实新观测到来
    → 对过去锚点的判断进行后见修订
    → 通过已执行动作和时间关系更新当前决策
```

未来教师还保留，但承担“训练更好的因果判断”的职责，而不是要求学生提前知道随机结果。

### 判断五：资源优势应主要用于强对照、可辨识诊断与闭环分支数据

仅用成功轨迹不断放大一个预测头，难以证明“状态理解”。
我们可以利用充足训练资源、3090/A100仿真能力同时获得：

- 同预算多种子对照；
- 相同场景不同历史、相同尝试不同结果的配对轨迹；
- 失败与恢复过程；
- 状态修订、动作修订和成功率之间的证据链。

这些证据通常比再增大一个模型更能区分想法是否正确。

---

## 1. 先把既有证据放回正确位置

### 1.1 证据账本

| 既有结果 | 可以支持 | 不能据此支持 |
|---|---|---|
| Past真实历史相对static：Fractal/Bridge动作MSE约−5.16%/−5.62% | 当前Past策略利用历史 | 已胜公平重训的current-only；所有新环境泛化 |
| Reader Study：真实Future相对均值，固定读出约−3.55%/−2.28%，可适应读出约−3.73%/−3.68%，四训练种子方向一致 | 真实Future条件在该协议中有增量 | 预测Future同样有用；收益全部来自时间而非额外current表征 |
| 小R双向pilot变化极小、跨域不一致，正逆梯度近乎正交 | “共享参数即相互促进”的实现未成立 | 所有后向状态推理无效 |
| 普通K和等参数latent能解释早期大部分增益 | 适配器容量/微调是必须排除的解释 | 必须完全放弃Past修正 |
| 当前20k相对100步开发动作MSE约−15.7%/−13.8% | 在线学生在两个dev域学到更好动作 | 改善由Future监督导致；20k优于同预算latent |
| 时序progress probe曾优于WoG，部分其他状态probe变差 | 表示中可能有时序信息 | query天然对应阶段；归一化episode进度等于任务状态 |

来源：`evaluation/README.md`、`stage3/READER_STUDY_RESULTS.md`、
`stage3/{PILOT,IMPROVEMENT,REFINEMENT,ORACLE_EVIDENCE}_01_RESULTS.md`及
`STAGE3_EFFECTS_20261003.md`。旧样本集、精度、初始化、训练预算不同，禁止跨表直接排名。

### 1.2 对历史失败的解释也需要控制变量

原design把几次失败直接归为压缩、递推或精度融合，部分只是合理嫌疑，不是已识别因果。
例如Reader Study同时改变了数据量、训练预算和可训练范围；不能把后来的成功唯一解释为“解冻解决了问题”。

原始代码中的真实错误需要修复，科学假设的错误需要实验排除，两者不能混为一谈。
也不能因为某次错误构造导致的负结果，否定整个理论。

---

## 2. 文献地图：哪些叙事已经不够新，哪些问题仍值得研究

### 2.1 必须正面处理的邻近工作

| 工作 | 原文中的机制 | 对WhenAmI的约束/启发 |
|---|---|---|
| **WoG** [L01] | Stage I学习动作相关Future条件；Stage II预测该条件并训练只读当前z的动作头 | Future对齐不是新贡献。应借用“动作决定有用的预测空间”，而非把其条件等同完整状态 |
| **FLARE** [L02] | action与future token在动作DiT中交互，未来latent对齐，目标EMA适应 | “让预测辅助目标进入真实决策计算”已有明确方案 |
| **HiF-VLA** [L03] | 历史motion vector prior、未来motion reasoning、联合action/motion专家 | “过去＋未来＋动作”或“hindsight/foresight”不能单独作为创新 |
| **RB-VLA** [L04] | action-conditioned递归belief、随机latent prior/posterior、扩散策略 | “增加belief state/记忆来解决部分可观测”已有直接VLA工作 |
| **ChainVLA** [L05] | recurrent working state＋event memory＋未执行motion tail，形成可修订跨query执行状态 | “维护可修订状态/跨调用连续性”也不够。需要明确修订的是同一过去事件的解释及其训练依据 |
| **PALM** [L06] | interaction-affordance预测与within-subtask progress | 单加progress head或阶段标签不是独立贡献 |
| **SG-WAM** [L07] | geometry约束，预测policy自身的EMA未来状态，条件含intervening action sequence | “policy-derived action-relevant latent＋动作条件预测”已有近邻 |
| **TD-VAE** [L08] | filtering belief、local smoothing posterior、两个不同时刻的jumpy预测 | 双时间点、后见学习、状态级预测有成熟理论前身，必须引用而不是包装成首次 |
| **RMA** [L09] | 用历史状态和动作估计训练期特权环境latent | “特权latent蒸馏到历史编码器”不是新范式 |
| **A2D / To Distill or Decide** [L10,L11] | 特权专家可能不适合部分信息学生；随机动力学影响可蒸馏性 | oracle更强不意味着学生能或应该精确模仿它 |
| **FACT** [L12] | 用失败动作的真实后果训练world branch，不把失败动作当正例BC目标 | 想证明动作后果理解，成功示范之外的数据很重要 |
| **OneWorld** [L13] | 不同action-outcome分支应兼容同一潜在物理机制；按信息强弱保留不同不确定性 | 不同后见结果不能被强迫成相同latent；分支一致性不是逐向量相等 |

这些工作不是都应原样复现为主表baseline。主表基线需按可用代码、任务、预训练预算和执行协议选择；
上述表首先用于**确定我们不能重复声称的贡献和必须排除的解释**。

补充阅读：TCC、V-JEPA2、STARRY、PAD、Dense Policy、τ0-WM、WISE、DAWN [L14–L21]。
例如Dense Policy的bidirectional指动作序列展开，不是对同一隐状态做后见推断；
DAWN是自动驾驶工作，不能把其中概念直接视为操控上的已验证方案。
STARRY/PAD/τ0-WM的联合预测及WISE的按阶段调度都使“再加未来生成/再加gate”
成为很弱的创新描述。

### 2.2 这不是“别人做过，所以缩小目标”

相反，资源和已有基础允许我们处理更难、也更明确的问题：

> **在部分可观测操控中，对过去事件的错误解释怎样被后续真实证据纠正，
> 纠正又怎样改变当前状态估计和动作，而不是仅追加一段历史或预测一个好看的未来？**

需要验证的差异是**显式事件时间索引＋证据到达时间索引＋修订路径＋决策后果**，
不是“有后向模块”。目前只能说这是值得验证的研究方向，不能宣称已完成新颖性证明。

---

## 3. 原始design中必须修正的理论表述

### 3.1 后向推理不是物理转移的函数逆

在受控Markov状态模型下，转移是：

\[
p(s_{k+1}\mid s_k,u_k).
\]

后向推断要结合先验和已经看到的证据。例如标准平滑递推（略去给定控制的记号）：

\[
p(s_k\mid o_{1:T})=
p(s_k\mid o_{1:k})
\int p(s_{k+1}\mid s_k)
\frac{p(s_{k+1}\mid o_{1:T})}
     {p(s_{k+1}\mid o_{1:k})}\,ds_{k+1}.
\]

对应的计算关系可简写成：

```text
smoothed_k = filtered_k *
             integral( forward_transition *
                       smoothed_(k+1) / predicted_(k+1) )
```

依据Särkkä平滑推导 [L22]。逆向推断可以共享同一个**转移概率因子**，
但不必、通常也不能满足`T_minus(T_plus(s,a),a) = s`。

抓取后接触变化、遮挡、丢落等情况下，多个过去状态可以导致相似后续观测。
模型应保留歧义，而不是被cycle loss强迫到唯一逆解。

### 3.2 原v1的alpha定义存在重复观测风险

若`alpha_k`已经定义为`p(s_k|o_1:k)`，它已经吸收当前观测`o_k`；
再乘`ell(o_k)`会重复计数。

只有把alpha定义成**更新前的预测分布**`p(s_k|o_1:k-1)`，
才有`alpha_pred × current_likelihood × future_likelihood`这种写法。

这首先是原文符号需要修正；不能在未核对每个历史版本实现时宣称所有模型都实际重复计数。

### 3.3 “未来更多，所以每个样本一定更确定”不成立

更多真实信息，在合适损失下不会使最优Bayes决策的**期望风险**变坏，但：

- 不是任何近似模型都更好；
- 不是每个样本熵都下降；
- 后续证据可能推翻原来过度自信的判断，使单个样本更不确定。

本轮离散例子：原先`p=0.99`，一个有效但出乎意料的观测可把后验改成`p=0.5`，
熵从约0.056升至0.693 nats。故不能强制每次修订只提高confidence。

### 3.4 自预测Future不能当成独立证据加精度

设历史为`H`、真实未来为`F`。对同一一致概率模型：

\[
\mathbb{E}_{F\sim p(F|H)}[p(s|H,F)] = p(s|H).
\]

这是条件概率的边缘化关系。若从自身模型采样未来再无条件拿来“验证自己”，
平均意义上没有新观测信息。把相同信息当独立专家相乘，还会假性自信。

本轮二状态数值检查：先验0.8，正确平均不同后见后验仍为0.8；
错误地把同一信念独立相乘会变成约0.941。

**这不否定想象的计算价值**：有限容量模型可借助多步计算、action planning和训练监督变好。
但要把价值表述为计算/归纳偏置/训练迁移，不能写成凭空获得额外观测证据。

### 3.5 更强oracle不自动是更适合的动作老师

如果两条轨迹具有相同可见历史，却因为不可见随机因素应执行不同动作，
学生不可能在看到新证据之前逐样本复现全知老师。

A2D及To Distill or Decide明确研究了这种问题 [L10,L11]。
应该学习对当前可见信息适用的分布/策略，必要时获取信息，
而不是把全知老师的确定动作无条件蒸馏过去。

对于当前项目，需要区分：

- **representation supervision**：后见信息帮助形成更好的任务表示；
- **decision supervision**：老师的动作在学生信息条件下是否仍合理；
- **information-gathering actions**：确认、观察、重试，不应被全知老师排除。

### 3.6 误差、query正交和动作窗口不能被过度解释

- `1/(单次特征误差+eps)`不是自动校准的状态后验精度；
  它随特征尺度、背景、可见性和模型误差变化，还可能直接依赖当时不可见目标。
- query之间正交不能证明语义独立、条件独立或各自对应一个状态维度。
- 独立latent坐标中的cycle失败不等于世界物理不可逆；
  CycleOCL [L23]也提醒潜在表示多解时，显式对齐可能惩罚合法解释。
  其结论针对OCL，不应扩张为“所有latent一致性都错误”。
- 重叠动作块不必预测完全一致：后一个时刻有新证据和新的闭环动作需求。
  应比较同一物理时间目标及相同证据条件；真实反馈导致的重规划不能被惩罚。
- 旋转动作不能普遍逐维相加当精确SE(3)合成；混合OXE动作坐标、控制器、
  归一化后更不能假设sum保留了完整执行含义。优先编码实际动作序列及mask。

---

## 4. 源码核查：当前能复用什么，历史上还藏着什么问题

### 4.1 当前Stage3的实际限制

**[C01] `stage3_scale/model.py:40–78,142–169`**

- 一个中心时刻的`H_P,C_P,z`；
- `r.predict(...,"forward")`，逆向专用adapter被冻结；
- `K(H_P,z,C_F_hat)`产出条件残差；
- 仅action FM和Future condition MSE，当前forecast没有逆向重建目标；
- student不接收真实Future、动作真值不进入R/K。

因此它是一个实现清楚的单中心因果策略，可作为`F0`稳定基线。
不能仅改名字就把它当作新的`b[k|t]`结构。

**[C02] `stage3/connector.py`**

R能预测另一端的条件，但输入没有未来候选动作序列。
因此当前`C_F_hat`描述的是示范分布下的预测条件，不是一般的
`p(future | history, do(candidate_action))`。
不能把它直接用于候选动作物理后果评分。

**[C03] `stage3_scale/model.py:173–230`**

初始化保留了已学好的VLM/Past/DiT，K和adapter来自adaptive_mean；
Future坐标均值/scale来自之前Reader Study统计。
这有利于稳定，但两个域的固定统计是否适合27域，需要分域检查，
不是一条全局alignment下降就能确认。

### 4.2 时间信息通道需要正确归因

**[C04] `WoG_16rank_probe/vla/qformer.py:24–38`及`FrozenVisualTargets.encode`**

当前DINO patch展平后作为cross-attention memory，没有显式帧时间编码；
Wan VAE仍有时间顺序。**不能说整个模型无时间信息**。

本轮用真实QFormer类的小尺寸CPU实例，保持VAE输入不变，只置换4组DINO帧patch：
输出最大绝对差约`2.38e-7`，符合cross-attention对memory置换的不变性。
这说明若要新增明确锚点，不能只对外声称“4帧输入自然就是4个时间状态”。

候选动作：保留frame/token和真实相对时间；先做分支隔离控制，不直接改当前checkpoint。

### 4.3 旧flow实现中发现的具体设计—代码不一致

以下指**当前仓库可见的历史源码**，未证明某个指定旧checkpoint一定由该文件版本训练。
关联到历史效果前需核对当时config、源码hash或快照。

**[C05] `WoG_16rank_probe/vla/temporal_flow.py:456–458`**

`dt`被硬编码为4。原采样`[-15,-11,-7,-3,0,3,7,11,15]`中，
中间跨度包含3，并非全部4；这不符合v2“显式真实Δt”的设计。

**[C06] 同文件`387–391`**

逐token alignment的target是所有patch的均值，再广播给每个token。
这会让每个query的读出朝同一个目标收敛，不能支持“每个query各司其职，
所以逐token误差就是其语义置信度”的解释。

**[C07] 同文件`368–372,441–462`**

alignment target取自加了可训练`frame_embed`的`x`，而不是纯冻结视觉特征。
不应把这个目标描述为完全固定、不能共同漂移的监督坐标。

**[C08] 同文件`450–452,461–489`**

`use_future=True`时，帧状态经跨帧编码已经能看到整段观测。
因此后续所谓“从第一帧预测未来”的源状态可能已含未来，
其低预测误差不能单独证明因果动力学能力。

这些发现意味着：原来失败的模型中有可解释的实现风险，
不能将所有负结果都当成对“完整双向思想”的干净证伪。
但当前稳定Stage3没有走旧`temporal_flow.py`，所以**不构成停掉当前任务的理由**。

### 4.4 数据层其实已给新方法留了空间

**[C09] `stage3_scale/oxe_data.py:128–168`**

A产物保留完整episode的压缩图像、动作序列、指令、frame spans和合法中心，
不是只剩一段固定特征。可在新loader上抽取多锚点，**不需要重新下载或重做全OXE**。

还缺：

- 显式`event_time`与`evidence_available_until`；
- 每条监督的动作可见性mask；
- 实际Δt/frame-index差，数据频率已知时再换算秒；
- episode内指令变化、失败/恢复事件和可信状态标签；
- 多锚点是否都具有合法动作窗口的mask。

短轨迹要单独报告：边界重复不能变成“多个真实后续证据”。

---

## 5. 我建议的完整研究对象：Evidence-Time State Revision

暂用描述性名称“按证据时间修订任务状态”，不急于给方法取正式缩写。

### 5.1 用`b[k|t]`取代只讨论`z_t`

区分三个轴：

1. **事件时刻k**：我们正在解释哪一次接触、抓取、转移或任务阶段；
2. **证据截止时刻t**：模型实际可以看到哪些观测/已执行动作；
3. **任务语义**：全局指令及当前子任务，不等于episode百分比。

`k<t`时，用后续但已经到来的观测解释过去是完全因果的。
`k=t`时，尚未发生的Future只在训练教师或明确的想象分支中存在。

**重要理论边界**：给定全部相同历史，精确Bayes filter对当前状态已经是充分的。
我们不宣称一个精确smoother能凭同一信息再次提高当前Bayes最优性。
研究的是有限容量、部分历史压缩、非精确模型中的结构化修订是否更易学习、更稳健。
必须与相同历史、相同计算预算的普通memory模型比较。

### 5.2 状态应该表达什么

建议两层，不把整个环境硬塞成一个阶段分类：

- **task/event层**：正在尝试、已确认完成、尚未确认、失败后需要恢复；
  哪个对象/子目标是当前相关对象，哪些前置条件已成立；
- **continuous residual层**：位姿关系、运动、接触趋势等不能离散枚举的任务相关信息。

初版可用宽latent承载两者，再用有限的状态/关系probe和监督锚定。
不能把模型中每个维度或query预先宣称为“物理状态”。
如果增加概率头，就用NLL/Brier/校准检查验证，不把一个任意gate称为不确定性。

### 5.3 后见修订应该修正“当时的解释”，而不是篡改真实历史

必须区分：

```text
情形A：当时根本没有抓住，后来才看清。
情形B：当时确实抓住，后来发生滑落。
```

后来看到空夹爪，不能都反推成“当时没抓住”。
这恰好需要动作区间、后续证据以及多个时间锚点。
仿真应保存每个锚点真实状态，检查模型有没有把结果倒灌成错误的历史。

---

## 6. 候选完整架构：真实证据驱动的“预测—核对—回溯—决策”

### 6.1 在线流程

```text
                         事件时间：
                   k0 -------- k1 -------- k2 -------- t
存储的因果判断：   b[k0|k0]   b[k1|k1]   b[k2|k2]
已执行动作：          u[k0:k1)    u[k1:k2)    u[k2:t)

新观测 o_t
   │
   ├─ 当前任务/视觉证据 e_t
   │
   └─ 与“执行实际动作后应看到什么”的预测核对 → innovation
                                                    │
              对仍有歧义的旧事件进行后向更新 <─────────┘
                   b[k0|t]、b[k1|t]、b[k2|t]
                                  │
                       汇总修订后的关系/状态与当前证据
                                  │
                         b[t|t] → 共享动作DiT → 动作
```

三个模块职责：

- **Forward transition `F`**：在已执行/候选动作和真实Δt条件下预测状态/任务后果；
- **Backward evidence updater `B`**：利用新到达的证据修订旧锚点解释，
  不是求`F`的函数逆；
- **Decision reader `D`**：从修订后的当前任务状态生成动作，
  必须可被ablation证明用到了修订，而不是只走current-z捷径。

可以共享转移因子、状态坐标或部分trunk；正向生成和后向推断的头允许非对称。
不要为了“一个算子”的口号强制两任务完全共享每一层参数。

### 6.2 和当前R/K的关系

不是把所有现有模块扔掉：

- 保留VLM、动作DiT及其训练权重；
- Past宽特征可作为初始事件编码，不强迫先压到16×32再做动力学；
- R的当前无动作Future预测保留作baseline/初始化，但不能直接称动作条件转移；
- K的残差读出思路保留，新K应读取**被证据支持的修订消息**及其来源时间；
- 连接到原动作条件时使用小尺度/残差接口，避免破坏现有动作能力；
- 真实Future条件teacher可保留为辅助，但不再垄断“任务状态正确性”的定义。

### 6.3 多轮推理何时有意义

只有中间解释、证据分配或候选动作确实更新时，第二轮才是新的计算。
固定同一`H,z,C_hat`重复喂R/K不构成时间推理。

测`0/1/2/4`轮，同时提供同参数、同调用次数的普通residual refinement对照。
若修订轮数增加只让模型更加自信、状态/成功率不改善，判为机制失败或校准问题。

### 6.4 想象分支的正确位置

如果新增候选动作想象：

```text
当前belief → 候选动作a1,a2,... → 各自条件化的后果分布 → 决策
```

它用于比较动作后果和可达性，不用于把自生成的好结果当作已观测事实。
每个候选共享同一初始世界假设，不允许一个候选隐含重物、另一个隐含轻物却不报告差异。
OneWorld [L13]是这里必须对照的近邻；FACT [L12]说明失败后果数据的重要性。

这一分支应是独立扩展；即使先不做规划，真实证据的在线修订也能构成可检验核心。

---

## 7. 训练方案：既有三阶段不作废，新增研究分支有独立协议

### 7.1 不覆盖正在训练的Stage3

当前Stage3作为`F0`继续完成训练，随后导出、跑主表和SFT适配。
新分支暂称`WhenAmI-Revision`，不是强迫当前策略再训练一个Stage4才能部署。

### 7.2 数据样本合同

从同一episode取多个锚点`k0<k1<...<t<T`，每个样本记录：

```text
episode_id
event_times / frame indices / actual deltas
observations
executed_action_sequences and masks
instruction_at_each_anchor
evidence_cutoff
state/progress/relation labels when legitimately available
action_target_windows and validity masks
success/failure/recovery annotations when available
```

对每条分支独立构造可见性，而不是只在顶层传一个`use_future=False`。
图像编码、跨帧attention、VAE视频、缓存和teacher features都必须遵守同一个cutoff。

### 7.3 建议损失：主目标明确，辅助各有意义

\[
L = L_{\text{causal-action}}
  + \lambda_{\text{dyn}}L_{\text{transition}}
  + \lambda_{\text{rev}}L_{\text{revision}}
  + \lambda_{\text{ground}}L_{\text{grounding}}.
\]

1. **`L_causal-action`**：每个选定锚点只用当时合法信息，监督其动作窗口。
   当前动作允许使用已发生历史动作，不能使用其目标动作片段本身。
2. **`L_transition`**：同一转移关系连接明确锚点，使用区间动作序列；
   预测任务相关特征/后果的分布。包含失败转移，不只拟合成功未来。
3. **`L_revision`**：同一事件k在后续证据到达后的修订，要求更符合可核验状态、
   观测关系或校验后的冻结teacher解释；不无条件硬逼因果学生复现每个随机后见结果。
4. **`L_grounding`**：防止坐标任意漂移或忽略任务状态。
   可由固定视觉/几何目标、可验证事件标签、任务状态诊断共同提供。
   使用额外标签/teacher时，各对照获得相同数据资源，并明确披露。

首轮不把四个权重和所有结构同时搜索。先固定因果action、做一个修订目标增量，
再逐项加入distribution/dynamics/规划。这是可归因实验设计，不是为了省卡。

### 7.4 Teacher目标与不可辨识性

Teacher可使用更长真实观测窗口，但需有稳定坐标和质量检查。
先冻结已验证的teacher作为控制，再比较EMA teacher，避免目标和学生共同漂移产生假对齐。

对相同可见历史、不同不可见结果的样本：

- 用后见目标训练条件分布，允许多模态/不确定；
- 保留真实在线动作任务；
- 不把低MSE的条件均值误当“已经知道正确分支”；
- 不能因为teacher置信度高就无条件高权重蒸馏，它可能依赖学生不可见的信息。

### 7.5 动作标签泄漏：必须单独建立合同

世界模型在训练时用`a[k:k+Δ)`预测后果是合法的，因为该分支任务就是建模动作后果。
但是：

```text
把真实a[k:k+15]送入后见状态编码
再让D(state)输出a[k:k+15]
```

会形成近似标签通道，不能当作状态表示“特别有用”的证据。

因此：

- **状态/动力学teacher**可以知道某些区间已执行动作，但其成绩必须由状态/后果检验；
- **时刻k的动作teacher或动作因果分支**不得把待预测动作窗口作为条件；
- 多锚点监督时，各锚点的mask不同；
- 最终部署只访问`a_<current`和自身候选动作，不访问专家未来动作。

### 7.6 不强制动作重规划一致

同一个物理时刻的预测可以因新证据而改变，这是正确行为。
只有“证据相同、对应目标相同”的预测才适合一致性约束。
若采用重叠动作块一致性，要屏蔽新信息引起的合法修正或通过损失明确定义其适用范围。

---

## 8. 决定成败的实验，不是只补一张更大的MSE表

### 8.1 主表：完整可部署策略能力

- SIMPLER按WoG协议先接通，得到工程可比较的成功率；
- 加入真正记忆相关基准，例如RMBench [L24]；若拓展LIBERO/CALVIN，按它们规定的SFT与评测协议；
- 若目标基准需SFT，所有比较方法使用一致数据与预算；
- 报告各任务、总体macro结果、训练种子差异、推理延迟、显存和总训练成本；
- 外部发布checkpoint是性能参照，不自动等于公平架构对照；
- 当前完整模型是否超过同预算无Future，是独立于跨论文排行榜的问题。

### 8.2 核心机制表

| 对照 | 排除什么解释 |
|---|---|
| 当前Past联合微调 | 是否只是延长训练 |
| 同结构latent | 是否只是R/K容量和计算 |
| Future辅助预测但不走修订路径 | 是否普通辅助监督已足够 |
| 同历史/同动作输入的recurrent memory | 是否普通记忆已足够 |
| 仅forward vs forward+backward revision | 后向消息是否有独立价值 |
| 同预算通用多轮residual refinement | 是否只是多算几轮 |
| 同state supervision但无revision | 是否额外标签/teacher信息带来的收益 |
| 真后见teacher与可部署student | 监督上限、可迁移性及剩余差距 |

不必同一时刻把所有组合堆起来，但决定论文主张的关键比较都要完整预算、多训练种子。

### 8.3 “视觉近似、决策不同”的配对诊断

构建两类：

**A. 历史可辨识**：当前画面相似、历史不同、正确下一步不同。
例如已放过一个物体又返回，与尚未抓取前返回同一位置。

**B. 当时不可辨识、后来可辨识**：同样的抓取尝试前缀，
抓取成功/空抓/稍后滑落在下一观测或触觉可区别。

要求：
- 输入模态公平；不能一个模型额外有force/proprio，另一个没有；
- 仿真状态仅作为teacher或评价标签，不能偷偷给在线策略；
- 不用绝对timestep、episode总长或固定程序顺序泄露答案；
- 在配对组内匹配/记录语言、相机、位置等变量；
- 预先定义任务分布，不按我们模型失败或成功来挑最终测试样本。

### 8.4 修订本身必须有指标

对同一事件k分别评`b[k|k]`和`b[k|k+Δ]`：

- event/relation accuracy、Brier/NLL和校准；
- **revision gain**：后见正确率/评分相对当时判断的变化；
- **false revision rate**：把原本正确的旧事件误改错的比例；
- **correction latency**：有效新证据到达到改正判断需要几次观测；
- **recovery latency / recovery success**：判断修正后何时改变行为并恢复任务；
- 不可辨识配对中，证据到来前是否保留合理不确定性；
- “当时没抓住”与“抓住后滑落”是否被混淆。

不能只测“能否线性解码归一化progress”：一个时钟也能拿高分，却不理解状态。
也不能只测latent漂移大，漂移大可能只是表示不稳定。

### 8.5 干预与重训对照搭配

干预：

- 正确顺序/倒序历史；
- 真实新证据/重复旧证据；
- 同事件正确后续/匹配外观但错误的后续；
- 对应动作/打乱区间动作；
- 开启/关闭反向消息；
- 修订后的memory/修订前memory；
- 图像延迟、drop、遮挡、意外接触、失败恢复。

分布外干预用于解释依赖，不自动等于公平因果效应；
关键结论要用相应重训控制或仿真中的受控分支实验确认。

### 8.6 利用千卡资源真正稀缺的部分：失败与分支数据

OXE成功演示适合动作能力与表征初始化，但通常不提供相同前缀下的大量随机分支。
在仿真中保存同一初始状态，采集不同动作/扰动下的后果：

- 正常抓取、轻微偏差、空抓、滑落；
- 成功放置、物体弹出、目标被遮挡；
- 一次尝试失败后的恢复路径。

失败轨迹可训练状态、动力学和后见修订；
**不要默认把失败动作当expert imitation目标**。恢复行为需成功恢复示范、
可验证规划、在线纠正或适当RL等真实来源，不能从标签中凭空制造好动作。

---

## 9. 必须接受的否证标准

1. **Teacher不比无后见控制更好**：先修teacher/目标/任务，不无限增加蒸馏量。
2. **只预测Future更准，动作不改善**：不能称决策理解改善。
3. **后见只更新latent、不改变当前动作**：修订路径没有实际价值。
4. **与等历史等计算memory模型持平**：不能把新增参数/上下文收益写成后向推理贡献。
5. **只在知道真实Future的oracle好，在线差**：特权迁移问题未解决。
6. **证据到来前在不可辨识分支上高置信度“猜对”**：先排查泄漏或数据偏置。
7. **始终高置信、失败时不修订**：belief解释不成立。
8. **只提升普通仿真任务，关键修订任务无增益**：重新限定主张，而不是用总均值遮盖。
9. **闭环下降、离线MSE上升/下降不一致**：以真实任务指标判断策略价值。
10. **收益只来自新的teacher/标注量**：对照同资源普通辅助学习后再谈结构贡献。

否证不是失败管理，而是避免在资源足够的情况下还做出不可归因的结果。

---

## 10. 并行研究组织：不再串行押注一条结构

### A线：当前主线完整交付

不改当前训练；完成50k、独立推理包导出、3090/A100环境闭环接入、
同协议WoG/Past/current-Stage3结果。已有20k可用于开发，但最终模型选择规则要先固定。

### B线：公平归因

优先`latent`、仅辅助Future目标、Past联合微调。
同一套OXE27、初始化、FM次数、global batch、训练步数；
关键比较至少三个训练种子，报告同观察预算和实际计算成本两种口径。

### C线：信息与修订诊断

在现有20k/50k上做干预、分域/阶段错误分析；建立配对可辨识数据，
先确认后见修订对什么状态和决策有用。
不需要等待新大模型训练完成。

### D线：新架构研究

独立目录实施双时间索引、多锚点、合法action序列和非对称后向更新。
用完整预算比较forward-only / revision / 同计算memory；
小测试只验证合同，正式效果不以几百窗口的toy胜负决定。

### E线：分支数据与闭环

3090侧先跑环境、数据采集和基础策略；A100侧跑独立推理和大规模评测。
GPU资源也投入统计重复和失败恢复数据，而不只投入单模型训练吞吐。

**依赖关系：** D的新颖性与效果取决于B/C/E的结论，而不是“论文截止逼着减少模型”。
如果C显示没有有效修订信号，就应先修问题定义；如果B已证明F0足够，
新结构必须给出额外可测价值，不能靠更复杂获胜。

---

## 11. 建议的代码落地边界（本轮未实施）

```text
whenami_revision/                 # 独立新目录，不能放入现有recipe哈希范围
  data_contract.py                # 事件时间、证据截止、动作可见性、Δt、mask
  state.py                        # b[k|t]，同锚点不同证据版本
  transition.py                   # action-sequence-conditioned F
  evidence_update.py              # 非对称B；不要求函数逆
  policy_bridge.py                # 保留已训练VLM/DiT的残差接口
  objectives.py                   # causal action / dynamics / revision / grounding
  train.py
  inference.py                    # 流式memory、真实动作、reset、无未来读入
  tests/
```

优先合同测试：

1. Future随机化不改变任何合法prefix输出；
2. 更改未来动作不影响过去动作策略分支；
3. 同一锚点在不同evidence cutoff下身份不变；
4. 重复同一证据不产生虚假的额外confidence；
5. 纯动作loss确实到达修订模块，关闭路径能改变可测行为；
6. 旧事件label与后来状态变化不混淆；
7. 变速/停顿/回退不让“任务进度”退化成时钟；
8. 未知Δt、无动作、边界重复具有明确mask，而非伪造正常输入；
9. 推理缓存reset、分批执行与离线前缀回放一致；
10. 导出前后固定输入中间量与动作一致，跨硬件误差报告透明。

先稳定数据与信息合同，再开展多种架构比较。不是先写一个大Transformer，
最后再从结果中解释每个token“原来代表什么”。

---

## 12. 本轮实际完成与未完成

### 已完成

- 重读原始`DESIGN.md`、v2、Stage3设计、pipeline和多轮正/负实验报告；
- 对当前joint模型、旧flow、QFormer、数据产物接口作逐段源码核查；
- 一手论文定向检索、方法段阅读与本地归档，阅读深度见下面；
- 真实QFormer类的小尺寸CPU帧置换检查；
- 三个离散概率数值检查：后验边缘化、重复证据、反常证据的熵变化；
- 形成明确候选机制、反例、对照矩阵、并行研究路线。

### 未完成，也不冒充完成

- 没有新Revision模型实现或训练；
- 没有证明新机制优于现有F0或任何文献方法；
- 没有穷尽最新论文/所有代码的新颖性审计；
- 没有完成NVIDIA独立导出与闭环接入；
- 没有修改正在运行的训练源码/optimizer/数据集/封存测试。

**研究主张应保持一句话：**

> WhenAmI研究的不是“猜出一个未来再让自己相信它”，
> 而是“在新的真实证据到来时，修订对同一任务事件的理解，
> 并把可验证的修订能力转化为当前的因果决策能力”。

这个主张既保留原idea，又避免把未来预测、记忆、时间阶段和物理可逆性混成一件事。

---

## 附录A：一手文献与本轮阅读范围

版本以实际归档为准；预印本的性能数字未复现，不作为本项目成绩或确定事实背书。
本文不依赖第三方论文摘要站作技术依据。

| ID | 文献/版本 | 阅读范围 |
|---|---|---|
| L01 | World Guidance: World Modeling in Condition Space for Action Generation, arXiv:2602.22010，本地20页PDF | §3.2/3.3、消融、附录A/B；与动作模型代码核对 |
| L02 | FLARE: Robot Learning with Implicit World Modeling, arXiv:2505.15659v1 | 复用前轮归档，方法、future/action交互、EMA及相关消融 |
| L03 | HiF-VLA, arXiv:2512.09928v2，2026-04-09；CVPR 2026原文入口 | §3.2–3.4、历史/未来控制与效率消融 |
| L04 | Recursive Belief Vision Language Action Models, arXiv:2602.20659v2，2026-02-25 | §III系统/状态/训练目标；v1也归档，但本文采用v2 |
| L05 | ChainVLA, arXiv:2608.02326v2，2026-08-04 | 方法状态结构、motion tail、episode-order训练及关键消融 |
| L06 | PALM, arXiv:2601.07060v1，2026-01-11 | 摘要、方法的affordance/progress部分及评测组织 |
| L07 | SG-WAM, arXiv:2608.01397v1，2026-08-02 | §3.2/3.3的policy-derived目标、geometry teacher、区间动作接口 |
| L08 | Temporal Difference Variational Auto-Encoder, arXiv:1806.03107v3，2019-01-02 | §2–4的belief/smoothing/jumpy模型；不能把这些基础说成新提出 |
| L09 | RMA: Rapid Motor Adaptation for Legged Robots, arXiv:2107.04034 | 两阶段训练、history/action适应模块、部署信息边界 |
| L10 | Robust Asymmetric Learning in POMDPs, ICML 2021 / PMLR139 | 导论、非对称模仿问题与A2D目标 |
| L11 | To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable RL, arXiv:2510.03207v1 / NeurIPS2025 | 导论、decodability/belief contraction和随机性结论；未逐条复核所有证明 |
| L12 | FACT: Failure-Aware Causal Training for World-Action Models, arXiv:2608.10232v1 | §1/3问题、action-conditioned后果与失败loss mask |
| L13 | OneWorld: Learning Consistent Physics Across Actions in World Models, arXiv:2609.30946v1，2026-09-25 | 摘要/方法问题设定、共享机制与不等信息强度；不将其生成任务结果当操控证据 |
| L14 | Temporal Cycle-Consistency Learning, CVPR2019 | 前轮原文归档的§3时序对应；非任意latent函数逆 |
| L15 | V-JEPA 2, arXiv:2506.09985v1 | 前轮§3.1动作条件预测与多步训练记录复核 |
| L16 | STARRY, arXiv:2604.26848v2，本地PDF | 摘要/导论/结构；不复现报告的性能 |
| L17 | Prediction with Action (PAD), arXiv:2411.18179，本地PDF | 摘要/方法概览；联合去噪近邻 |
| L18 | Dense Policy, arXiv:2503.13217v1，本地PDF | 摘要/导论，确认“bidirectional”的不同含义 |
| L19 | τ0-WM, arXiv:2606.01027，本地PDF | 摘要/方法概览，统一video/action/evaluation近邻 |
| L20 | WISE, arXiv:2609.03681v1，本地PDF | 摘要/导论，阶段性想象调度近邻 |
| L21 | DAWN, arXiv:2605.11550，本地PDF | 摘要/导论，驾驶领域交互world/action背景 |
| L22 | Simo Särkkä, Lecture7: Bayesian Smoother, Gaussian and Particle Smoothers, 2016-03-16 | pp.3–10 smoothing定义、推导及边界 |
| L23 | Cycle Consistency in Video Object-Centric Learning, arXiv:2605.30211v1 | §3的显式slot一致性局限、implicit reconstruction一致性 |
| L24 | RMBench, arXiv:2603.01229v3，2026-07-30 | 摘要、memory-dependent任务定义与benchmark组织 |

原文地址（便于离线交接）：

```text
L01 https://arxiv.org/abs/2602.22010
L02 https://arxiv.org/html/2505.15659v1
L03 https://arxiv.org/html/2512.09928v2
    https://openaccess.thecvf.com/content/CVPR2026/html/Lin_HiF-VLA_Hindsight_Insight_and_Foresight_through_Motion_Representation_for_Vision-Language-Action_CVPR_2026_paper.html
L04 https://arxiv.org/html/2602.20659v2
L05 https://arxiv.org/html/2608.02326v2
L06 https://arxiv.org/html/2601.07060v1
L07 https://arxiv.org/html/2608.01397v1
L08 https://arxiv.org/pdf/1806.03107v3
L09 https://arxiv.org/html/2107.04034
L10 https://proceedings.mlr.press/v139/warrington21a/warrington21a.pdf
L11 https://arxiv.org/html/2510.03207v1
L12 https://arxiv.org/abs/2608.10232
L13 https://arxiv.org/html/2609.30946v1
L14 https://openaccess.thecvf.com/content_CVPR_2019/papers/Dwibedi_Temporal_Cycle-Consistency_Learning_CVPR_2019_paper.pdf
L15 https://arxiv.org/html/2506.09985v1
L16 https://arxiv.org/abs/2604.26848
L17 https://arxiv.org/abs/2411.18179
L18 https://arxiv.org/abs/2503.13217
L19 https://arxiv.org/abs/2606.01027
L20 https://arxiv.org/abs/2609.03681
L21 https://arxiv.org/abs/2605.11550
L22 https://users.aalto.fi/~ssarkka/course_k2016/handout7.pdf
L23 https://arxiv.org/html/2605.30211v1
L24 https://arxiv.org/html/2603.01229v3
```

## 附录B：源码与证据定位

`C01` `stage3_scale/model.py`：JointPolicy与目标。  
`C02` `stage3/connector.py`：正向/逆向adapter与R输入。  
`C03` `stage3_scale/model.py`：load_joint初始化与固定统计。  
`C04` `WoG_16rank_probe/vla/qformer.py`、`FrozenVisualTargets.encode`：时间信息通道。  
`C05–C08` `WoG_16rank_probe/vla/temporal_flow.py`：旧flow的Δt、逐query目标、frame_embed、可见性。  
`C09` `stage3_scale/oxe_data.py`：episode原图/动作存储与窗口。  

其他核对：

```text
DESIGN.md
DESIGN_v2.md
DESIGN_stages.md
stage3/DESIGN.md
stage3/NEXT_DESIGN.md
stage3/PIPELINES.md
stage3/RESEARCH_DISCUSSION_20261002.md
stage3/READER_STUDY_RESULTS.md
stage3/IMPROVEMENT_01_RESULTS.md
stage3/ORACLE_EVIDENCE_RESULTS.md
stage3/REFINEMENT_01_RESULTS.md
stage3/CAUSAL_STUDENT_INTERRUPTION_20261002.md
stage3_scale/JOINT_DESIGN.md
stage3_perf/train.py
WoG_16rank_probe/prismatic/models/backbones/vision/dinosiglip_vit.py
WoG_v2/refs/code/WoG/vla/vla.py
WoG_v2/refs/code/WoG/action_model/action_model.py
```

所有历史源码发现均应先核对实际run版本再归因；禁止通过覆盖历史代码“修正过去的结果”。
