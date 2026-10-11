# WhenAmI：新颖性核查、Stage3证据边界与立即实验

2026-10-04。用户要求：任何新方案先核查已有工作；明确当前Stage3能证明什么；
现在开始验证，不以新的架构名词或等待替代实验。

## 0. 结论

1. “动作条件belief + 观测更新 + 有限窗口平滑”不是可以直接声称的新方法。
   相关思想在具身策略、神经状态估计和可微平滑中已经有直接先例。
2. 当前Stage3确有固定27域离线动作预测增益，但尚无匹配重训因果归因，
   用户提供的50k闭环结果整体落后本机WoG-A。
3. 不抛弃现有成果，也不立即把已知组件拼成新的大训练。
   先做固定50k机制诊断，并补一个同结构无Stage3 Future对齐监督的完整控制。
4. 真正的新方法只允许以“明确差异 + 可证伪机制预测 + 强近邻对照”立项。
   此刻不能宣称“具身中没人做过”。

## 1. 已有工作：本轮不仅看标题

| 工作 | 已经覆盖什么 | 对我们方案的约束 |
|---|---|---|
| TD-VAE, arXiv:1806.03107v3 | 神经belief，局部smoothing posterior，跨时间预测 | 双时点/后见推断与belief表示本身不是新概念 |
| Recurrent Kalman Networks, ICML2019 | 高维学习隐空间的递归预测/观测校正、显式不确定性 | “预测→观测误差→更新latent”不能单独当创新 |
| Differentiable Factor Graph Optimization for Learning Smoothers, IROS2021, arXiv:2105.08257v3 | 学习观测/动态因子，展开平滑优化端到端训练 | “共享因子＋可微平滑”不能单独当创新 |
| RB-VLA, arXiv:2602.20659v2 | 动作条件的递归belief，动态学习，belief条件化扩散控制 | 与上一轮提案中的forward/state/control主干直接相邻 |
| HiF-VLA, arXiv:2512.09928v2 | hindsight运动表示、foresight推理、联合动作expert | “past/future联合指导动作”与当前Stage3都必须面对这个近邻 |
| ChainVLA, arXiv:2608.02326v2 | Working State、稀疏事件记忆、可修订Motion Tail串联多次动作查询 | “可修订执行状态”或跨步状态传递本身不足以区分 |
| TRACE, arXiv:2606.14551v3 | 固定容量visual/robot-state证据记忆，以执行轨迹path signature写入/读取，支持后来模糊分支决策 | “迟到证据任务”“固定记忆”“历史相关分支选择”均不能泛称空白 |

TRACE尤其值得注意：其delayed-evidence主要指**早期可见的cue留到后面才用于决策**。
我们若继续研究，应区分**后来的观测修订对过去事件的解释**，而不是仅仅记住旧cue。
这个区分仍然没有自动建立新颖性，因为平滑研究已包含后一种推断语义。
必须证明具体任务状态/证据路由/有限计算算法与上述方法不同，并影响真实控制。

本轮还检索到Divide-and-Remember、EventVLA、行动历史memory等更近的工作；
其中未完成正文核查的只能算筛查命中，不能拿来声称已经全面排除重合。
搜索覆盖有限；没有检索到完全相同的系统也不等于不存在。

### 阅读和来源范围

- 本轮下载TRACE v3全文，阅读摘要、问题定义、方法与memory路由段落；
  已下载RB-VLA v2、RKN、可微平滑原论文，阅读方法相关章节。
- TD-VAE、HiF-VLA、ChainVLA使用前次固定的原论文档案并再次阅读相关章节，
  同时本轮重新查询arXiv条目，不把先前聊天概述当原论文。
- 未独立复现任何外部论文；不引用它们的指标作为我们同协议baseline。
- 检索词包括paper titles、VLA belief revision、retrospective memory、
  differentiable smoothing、delayed-evidence visuomotor等。

原文与hash：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_novelty_20261004/
  TRACE.pdf / TRACE.txt
  RBVLA.pdf / RBVLA.txt
  RKN.pdf / RKN.txt
  DynaSmoother.pdf / DynaSmoother.txt
  FETCH_MANIFEST.json

/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_deepresearch_20261003/
  TDVAE.txt / HiF_VLA.txt / ChainVLA.txt / SOURCE_MANIFEST.json
```

准确名称说明：ChainVLA的题名是
“Chaining Vision-Language-Action Queries through a Unified Execution State
for Long-Horizon Manipulation”，不是先前评审中误写的“Steering Frozen ...”。

## 2. 当前Stage3能证明什么

### 已有证据

- 完整50k已训练并严格加载，工程链路可用。
- 1348个独立dev episode、27域、同噪声纯采样评测：
  macro MSE6比本地WoG-A低19.41%，比原Past低20.95%；
  分别19/27与24/27域点估计有利。
- 本轮Stage3训练index与评测episode键零重叠；基础预训练是否见过不保证。
- 这是固定多域开发协议下的实际动作预测证据，不是闭环成功率，也不是盲测主表。

### 尚不能证明

- 增益来自Future对齐而非联合微调/新增容量/训练预算；
- R预测的样本相关Future在部署时具有独立有益贡献；
- K是对过去事件的真正后见修订；
- 双向状态推理、已执行动作动态、迟到证据处理机制已成立；
- 闭环整体优于WoG-A：用户给出的50k结果恰恰没有支持这一点。

### 当前流程与提案区别

当前源码：

```text
H_P,C_P = Past(抽样历史、当前)
z       = VLM(当前、指令)
C_F_hat = R(H_P,z)
C_action= C_P + K(H_P,z,C_F_hat) × scale
action  ~ DiT(z,C_action)

训练才加：预测C_F_hat与冻结真实Future条件的对齐
```

`R`没有执行动作输入，`K`没有后来到达的新观测输入。
没有跨调用持久维护的显式事件belief，也没有对旧事件重新推断的目标。
因此它可以准确称为“历史条件策略＋预测Future辅助监督/条件修正”，
但不能称为已经完成的动作条件双向belief smoother。

提案要求的关键新增事实是：

```text
执行动作之后获得新观测
→ 检验过去/过程的假设
→ 更新事件后验及当前决策
```

这不是把现在的R/K多迭代几次就自动获得的功能。
要作为研究贡献，还需超过同信息、同监督、同初始化的强memory/filter/smoother对照。

## 3. 立即实验A：固定50k机制诊断

已实现于独立目录`stage3_mechanism/`。不改任何生产源码、权重或已完成评测。

5个条件：

| 条件 | 修改 | 检查什么 |
|---|---|---|
| full | 无 | 与原50k逐样本回放，校验加载/采样 |
| no_k | 最终条件仅C_P | K修正的部署依赖与误差影响 |
| mean_future | K读固定Future均值，H_P/z/C_P不变 | R的样本相关输出是否额外有用 |
| repeat_history | 四张历史替换为当前图像，完整重编码 | 真实历史相对静态输入的作用 |
| repeat_history_no_k | 同时静态历史/去K | 历史依赖与K修正是否耦合 |

同一27域dev1348episode、2694窗口、同三噪声数组、同采样精度/9Euler。
全量结果逐episode保存，发生中断可恢复；每个full输出对比既有50k输出，
出现加载/数值漂移即失败，不带着错误协议继续跑。

为了不给共享主机再加一份32GB CPU state，独立加载器先把模型移入私有PPU存储，
再加载DCP；不会写入mmap的原权重文件。需要通过全模型回放才算入场成功。

**解释限制：**

- 这是推理干预，可能分布外；不能把其效果当重训因果贡献。
- no_k不是原Past，也不是WoG-A：VLM/Past/DiT早已更新。
- mean_future不是“从未训练过Future监督”：其训练影响仍在其他权重中。
- 分支一旦被使用，不自动等于分支正确或新颖。
- 本地离线诊断不会取代下一步在SIMPLER上做对应受控干预。

## 4. 最优先完整实验B：同结构latent对照

这是已有Stage3贡献验证，不是为了声称新方法而复现已有模块。

保留：

- 同一Past与adaptive_mean5k warm start、同一seed42；
- 相同VLM/Past/R/K/DiT结构及可训练参数；
- 相同OXE27 index、采样/增强、global1280、50k更新、LR/scheduler；
- 相同320卡20×16拓扑和保存恢复策略。

只改变Stage3阶段是否提供真实Future target并计算alignment loss：

```text
forecast（已完成）: action FM + 0.1 future alignment
latent（待提交）:   action FM
```

“latent”仍有R/K、仍可经action loss训练；不是删结构后减少容量。
前序Future/Past预训练当然保留，所以结论只能是**Stage3额外Future对齐的贡献**，
不是一切future预训练的贡献。

已有launcher和train.py支持该分支。本轮检查了代码条件，
运行了preflight与dry-run，相关action-only梯度/target detach测试也在本轮执行。
该配置尚未在多节点实际训练；launcher会先做真实100步、保存/评测/梯度验收，
再用新进程resume，失败会停止，不静默跳过。

卡需求：320PPU（20节点×16），新任务，不resume当前50k。

```bash
VARIANT=latent \
RUN_ID=oxe27_latent_zero2_320_50k_save5k_seed42_01 \
NPROC_PER_NODE=16 RESUME=0 \
bash /home/zhengguantian/WoG_fresh/run_stage3_perf_320_zero2.sh
```

平台负责提供WORLD_SIZE=20/RANK/MASTER_ADDR；不用用户手填成torchrun的全局320。
RUN_ID必须是新目录；不要覆盖已完成forecast。
该latent不做Future编码，冻结teacher计算量较少；这是同数据/同更新的对照，
不是声称总FLOPs完全相等。预计时间必须看实际早期吞吐，不能保证旧任务时长。

公平比较必须从原来的共同起点训练，不能拿已经得到Future监督的50k关掉loss继续训。
训练内旧Bridge/Fractal监控不作为结论：新checkpoint会再使用27域离线协议，
并交给另一服务器在固定闭环协议上比较。最终还需多训练seed，不能以3个采样seed替代。

### 为什么这个实验最值得先跑

- 若latent与forecast差不多：当前结果不能归因于额外Future监督，
  应先查未来target/动作接口，而不是在它上面继续讲双向推理故事。
- 若forecast在离线和闭环都优于latent：建立了现有方法的一块机制证据，
  再定位历史及K的独立价值。
- 若forecast离线更好但闭环更差：我们得到一个明确的学习目标—部署表现冲突，
  新方案就应针对这个冲突，而不是仅增加历史容量。

## 5. 新idea的下一道门槛

后续任何候选都必须先交付：

1. 最近邻原论文＋与它们的逐项方法差异；
2. 一个不能用“普通memory/额外监督/多算几次”解释的可检验预测；
3. 同信息/同监督/同预算的直接对照；
4. 失败后如何放弃或调整该假设的预先规则；
5. 小规模真实图像/动作验收后才给大训练指令。

“过去事件可被晚证据修订，并导致当前动作正确改变”仍可作为待研究问题，
但还不是已证明的新颖算法。当前不会为了推进感而把RB-VLA或可微平滑
换个名字当新贡献。
