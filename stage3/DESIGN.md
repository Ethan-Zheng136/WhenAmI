# Stage 3：冻结两端，学习行为约束的双向条件预测

日期：2026-10-02。**本次是可验证的pilot，不是完整WhenAmI时间线或Stage 4策略。**
保留Stage 1/2全部源码、checkpoint和评测结果，不在旧训练入口上继续叠开关。

## 最新：27域OXE与320卡入口已准备，待全数据转换及v2集群验收

`stage3_scale/oxe_inventory.py`/`oxe_data.py`接入原27域mixture，
按原倍率×pinned transition数量采样，不再只做两域。27/27真实样本检查、
76项单元/回归测试和Kuka短轨迹的真实8B反向通过。
为保留Kuka，新增episode边界重复与有效动作mask，不假装仍是interior-only协议。
旧32卡验收在optimizer.param_groups恢复失败，v2改成DCP模型+rank-local Adam/RNG，
4rank CPU真FSDP/单PPU验证通过，32卡修正版尚待提交。
主配方320卡(20×16)、micro2×accum2=global1280、50k；原全权重可训练学生，
不是6.48M缓存学生。指令与边界见 `../stage3_scale/OXE27_320_SUBMIT.md`。
现在可并行提交27域原始数据准备与2节点v2验收；不是已宣布320卡主训练验证通过。

## 主训练数据范围澄清（2026-10-02）

原Stage1/2使用27域`oxe_magic_soup_plus`。最近`stage3_scale`的raw准备和
训练采样入口只覆盖Bridge/Fractal，不应把它称为全OXE主训练。
27域文件/配置清点已完成，数据入口尚待扩展与逐域验证；
两域工程验收可用但不能证明另外25域已兼容。详见
`../stage3_scale/OXE_SCOPE_CORRECTION.md`。

## 最新工程推进：全数据联合策略（2026-10-02）

独立`stage3_scale/`已实现原始图像分片/抽样、可训练VLM+Past+R/K+完整DiT、
固定Future目标/可选完整oracle教师、FSDP+DCP保存恢复、原图开发动作评测。
学生实测80.7亿可训练参数；本机真实全梯度和raw teacher/student前向通过，
69项测试通过。真正32卡8B train/save/resume仍需集群验收，不冒充已完成。
可以并行提交完整语料CPU打包和2节点16卡验收，指令见
`../stage3_scale/SUBMIT_JOINT.md`。主训练50k入口受验收及完整数据门控。
这是对之前全冻结Stage3的明确联合适应扩展，不是只把小缓存拉长步数；
旧代码/权重不变，多时间锚点时间线仍未实现。

## 已准备待提交：20k因果学生（2026-10-02）

见 `CAUSAL_STUDENT_20K_PLAN.md` / `CAUSAL_STUDENT_20K_SUBMIT.md`。
1节点16卡，4种目标×4种子独立并行；每学生由同seed的adaptive_mean5k权重
起步，新训练20k步，学习R/K与读出adapter。Future/past教师都冻结，
学生部署只有Past输入。增加无Future教师控制，区分普通蒸馏与Future增量。
每观测4FM抽样，500warmup+cosine，2500步保存，最终3noise开发动作评测；
封存测试不动。59项测试/四组真实smoke/实际中断恢复逐值一致性验证通过。
**正式集群20k任务尚未提交/运行，没有新学生收益结论。**

## 最新完成：读出适应四格×四种子实验（2026-10-02）

正式1节点×16卡的16组5000步训练、开发动作评测、自动汇总和完成审计均已结束。
见 `READER_STUDY_RESULTS.md`。真实Future相对同设置均值条件的主MSE：
固定读出降低Fractal3.55%/Bridge2.28%；可适应读出降低3.73%/3.68%，
各比较均4/4训练种子有利。适应增强Future增量的差上差证据仍不一致，
不能把此前失败完全归咎固定读出；小pilot负结果不应过度推广。
这是实际读取真实Future的oracle收益，尚未证明预测Future学生/双向监督有效。
新train16384/dev2048/test2048，测试继续封存，不与旧211样本表横比。
原Stage1/2权重未改；`runs_01`已完成，不要重复提交。

## 最新诊断与pipeline图（2026-10-02 05:41 UTC）

`PIPELINES.md` 按实际代码区分WoG I/II、旧R互译与当前R+K。
独立oracle证据诊断见 `ORACLE_EVIDENCE_RESULTS.md`：
纯真实C_F训练K未优于普通Past适配器；同结构的完整H_F、C信息还原H、
重复当前帧H与均值控制也未找到跨数据集一致的Future增益。
因此没有将H接口替换进当前R+K，没有新增循环/蒸馏。40项测试通过，
本轮训练、静态特征提取和评测均已结束；不据此断言“未来无用”。

## 最新：R+K残差回溯试验已完成（2026-10-02 05:05 UTC）

见 `REFINEMENT_01_RESULTS.md`。新代码独立于本文下述旧R互译方案：
保留Past条件，R预测Future，K输出Past条件残差，统一用冻结Past DiT监督。
五组各1000步：普通K、等参数无Future监督latent、预测Future、加oracle、
加4区间可微采样loss。主误差降低主要也能由普通适配器获得，尚无可靠
Future/回溯增量证据；多步采样辅助本轮没有改善。36项测试通过，
全部本地训练/评测完成，原Stage1/2与旧R试验均未修改。
下文“下一版仅设计/未实现”属于之前的历史状态，以本节为准。

## 后续诊断与改进对照（2026-10-02，已完成）

详见 `IMPROVEMENT_01_RESULTS.md`：梯度诊断后只将alignment权重1降至.1，
重跑两组1000步；Fractal稍好、Bridge退步，逆向增益仍微小，未替换默认配置。
同时补齐同一211窗口的原WoG Stage II / WoG-A评测，不能再将“真实Future”
误认成WoG Stage II。下一版候选 `NEXT_DESIGN.md` 强调保留Past行为、在线回溯
校正；目前仅为设计，**未实现、未验证**。

## 最新状态：正式固定子集pilot已在本地单PPU完成

见 `PILOT_01_RESULTS.md`。两组均在同一813窗口训练集上训练1000步，
211窗口按episode隔离验证，并以最终checkpoint做实际动作采样。
R条件预测有效，但双向相对仅正向的MSE变化仅Fractal −0.738%、Bridge +0.142%；
未显示有实质性的跨数据集逆向收益，也未全面优于原Past策略。
旧的“正式pilot尚未提交”等描述属于此前实施记录，当前状态以本节为准。

## 1. 做什么

```text
                  Stage 1/2 endpoints 全部冻结

历史+当前 -> Past QFormer -> H_P [16,1024] ----┐
当前+语言 -> Past VLM     -> z_P [1,4096] ----┤
                                            v
                                    输入适配(Past)
                                            |
                                      共享R，正向
                                            |
                                      输出适配(Future)
                                            |
                                      C_hat_F [16,32]
                                        /        \
                          对齐固定 C_F             固定 Future DiT(z_F)
                           (条件监督)                -> action loss

当前+未来 -> Future QFormer -> H_F ----------┐
当前+语言 -> Future VLM     -> z_F ----------┤
                                            v
                                    输入适配(Future)
                                            |
                                      同一R，逆向
                                            |
                                      输出适配(Past)
                                            |
                                      C_hat_P [16,32]
                                        /        \
                          对齐固定 C_P             固定 Past DiT(z_P)
                           (条件监督)                -> action loss
```

- 原端点QFormer、VLM及DiT均不更新。只训练两个输入适配器、共享Transformer和两个输出读出。
- 使用两个checkpoint**各自**的z、条件、动作归一化。不同读出不能随意互换。
- R仅接收来源侧H和来源侧当前图像/语言z，**没有真实动作或加噪动作输入**。
- 源头H来自实际predict_action路径的投影前query；z来自该路径的VLM generation；
  不重新编造另一套提示词/图像预处理。缓存时只跳过DiT采样本身。
- 输入中的16个query不是16个时间节点；这个版本是同一中心时刻的条件预测，
  不宣称学会正逆物理动力学、可逆变换或完整状态时间线。

## 2. R的初始配置

来源特征LayerNorm + 1024→256线性投影；
来源z LayerNorm + 4096→256投影；
一个任务token + 16个query，加query位置和方向embedding；
3个pre-LN Transformer block（8 heads，FFN multiplier=4，dropout=0）；
方向专属256→32读出。

输出为 `training_target_mean + training_target_scale * prediction`，
每个query/channel使用Stage-3**训练集**统计，std下限0.02。
这只是固定坐标尺度，不是Lambda/不确定性，不使用验证集拟合。
末端小非零初始化，使首步即可获得上游梯度，不重演全零出口问题。

## 3. 损失与梯度

```text
L = L_F_action + lambda_F * L_F_alignment
  + L_P_action + lambda_P * L_P_alignment  # only bidirectional mode
```

- alignment：按固定训练std归一化的条件L2。目标detach。
- action：WoG相同的flow matching；
  `epsilon~N(0,I)`，`tau~Beta(1.5,1)`，`x_tau=(1-tau)*epsilon+tau*A`，
  监督固定DiT输出逼近`A-epsilon`。
- 真实动作仅作为标签和DiT的标准加噪训练输入，绝不输入R。
- 固定DiT参数 `requires_grad=False`、模式eval；
  **不把预测条件经过DiT的路径包在no_grad中**，保留dL/dC_hat。
- reverse action loss仍监督当前开始的正向动作块，不把动作倒放执行。
- 默认两类损失权重均为1，是待试验的起点，不宣称最优。
  分别记录forward/backward alignment、action FM、total、grad_norm。
- 不加cycle、Lambda或第三种重建任务。双向预测是否有效先单独证明。

## 4. 数据和缓存

新建RLDS训练子集，**不复用先前256样本的评测集**。
Fractal从`train[:95%]`读，Bridge从`train`读。沿用OXE动作standardization，
先保存raw目标，再分别使用两个端点的统计归一化。
只取完整past/current/future窗口，每episode最多4样本；
默认每数据集512窗口（两个数据集合计最多1024）。

随后在每个数据集中按完整episode随机切80/20，缓存train_indices/valid_indices。
Stage-3验证集不能含同一轨迹的相邻训练窗口；但冻结端点本身可能已训练过这些轨迹，
不把它称作“全模型预训练未见测试集”。

两个大端点顺序加载，缓存H_P/H_F、C_P/C_F、z_P/z_F，只运行一次；
Stage 3训练时不用大VLM，只加载两个约409M的固定DiT。
缓存采用fp32，不包含数据增强，属于固定子集pilot，不等同于全OXE训练。
数据、固定端点、源码、数组hash都有manifest；不完整缓存不能开始训练。

## 5. 对照必须匹配什么

`--mode forward` 与 `--mode bidirectional`：

- 相同R结构初始化、训练/验证episode split；
- 相同forward minibatch、forward随机噪声与tau；
- 相同forward更新次数、LR、batch/world size和验证样本；
- bidirectional额外做backward，**总算力、总梯度贡献不同**，不称完全同预算对照。

forward-only会冻结未使用的逆向输入/输出适配器；输出shape和共享trunk相同。
这个比较回答“额外逆向任务是否帮助正向”，不是证明唯一的收益来源。
若有收益，再加等算力正向对照、多个种子和更大数据验证。

## 6. 验证阶段看什么

训练中独立验证集：

- predicted条件的对齐误差 vs **训练均值条件**的误差；
- predicted条件在固定读出上的action FM loss；
- 相同噪声下**真实条件**和**训练均值条件**的FM参照。

训练后 `stage3.evaluate` 还做实际10步Euler动作采样，存r/R²/MSE及逐样本预测：

```text
true_future               特权参照
mean_future               固定训练均值参照
predicted_future_causal   Past H/z -> R -> Future条件 -> 固定Future读出

true_past                 固定Past策略参照
mean_past                 固定训练均值参照
predicted_past_oracle      真实Future H/z -> R -> Past条件 -> 固定Past读出
```

`predicted_future_causal`不读取真实未来，但此评价需要两个当前图像的VLM特征，
计算预算不同；它**还不是Stage 4里融合进Past的在线策略**。
逆向分支是训练辅助和oracle诊断，不部署。
只有这版正向预测有行为价值，才值得讨论Stage 4融合/修正或更显式的跨时间状态。

## 7. 入口

所有新文件在 `/home/zhengguantian/WoG_fresh/stage3/`。
大文件和日志在 `/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_pilot/`。

先单节点准备缓存（默认16卡提取，不训练）：

```bash
N=512 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_prepare.sh
```

随后两组pilot（默认单GPU，先不需要320/640卡）：

```bash
S3_MODE=forward RUN_ID=s3_forward_pilot_01 \
MAX_STEPS=1000 PER_DEVICE_BS=8 NPROC_PER_NODE=1 \
bash /home/zhengguantian/WoG_fresh/run_stage3_connector.sh

S3_MODE=bidirectional RUN_ID=s3_bidir_pilot_01 \
MAX_STEPS=1000 PER_DEVICE_BS=8 NPROC_PER_NODE=1 \
bash /home/zhengguantian/WoG_fresh/run_stage3_connector.sh
```

可以使用一节点多GPU DDP训练，但对照必须保持相同batch/world size；
首次集群DDP仍需要smoke。该阶段不需修改原Stage 2任务，也不自动提交任何任务。
checkpoint只保存connector/optimizer和元数据，不重复保存两个巨大的端点。
虽保存优化器状态，本次尚无resume CLI；不要误称支持精确断点续训。

## 8. 本次检查边界

小规模测试只能验证路径、梯度、固定目标和训练是否能进行。
是否有效、能否优于Past、逆向是否有额外价值，都必须由后续held-out对照回答。

### 已实际完成的验证（2026-10-02）

- 14项测试通过，含实际2进程CPU/Gloo DDP的单向与双向梯度同步；
  不是16卡NCCL整任务验证。
- 从新的RLDS训练子集提取16窗口（Fractal/Bridge各8），按episode切成12 train / 4 val。
  两个真实大模型顺序提取，H经原v_proj还原C的误差在约5e-6内。
- 真实单PPU、原24层DiT-L固定读出、R默认256/3层，BF16：
  forward-only和bidirectional各30个optimizer step完成；
  冻结读出参数无梯度/无版本变化，所有损失与梯度有限；
  两组第一步forward条件损失/action FM完全相同，验证对照噪声和输入匹配。
- 约5.03M的R，双向checkpoint约58MiB（含optimizer），严格重载成功。
- 训练后checkpoint经过 `stage3.evaluate` 实际动作采样跑通：
  两个数据集、两种固定读出、真实/均值/预测条件，输出12组结果及逐样本预测。
- 30步双向验证条件误差：forward 1.276→0.935、backward 1.385→1.011；
  验证action FM：forward 0.103→0.088、backward 0.307→0.182。
  **这些仅是极小样本上的可学习性检查，不能证明方法有效。**
- 同一验证集forward-only的forward FM约0.08826，bidirectional约0.08836，
  基本相同，**当前没有证据宣称额外逆向任务提高了正向能力**。
- 数据与结果目录：
  `outputs/stage3_pilot/cache_smoke_20261002/`、
  `smoke_forward_20261002/`、`smoke_bidir_20261002/`、
  `smoke_bidir_eval_20261002/`。

正式pilot尚未提交，不修改已完成Stage 1/2的任何权重。
启动变量使用 `S3_MODE`，不是 `MODE`：本机Bash启动脚本会占用/覆盖通用MODE变量。
