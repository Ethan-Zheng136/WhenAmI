# R + K 首轮实现与实验计划

2026-10-02，执行前固定。用户同意尝试，并询问是否需要让action loss多走几步。
原 Stage1/2 权重、代码和全部已有结果不变；新代码使用 refinement 独立入口。

## 什么是共同指导动作

不是两个冻结DiT投票。在线只有一个冻结Past DiT：
`Past H,z -> R -> C_hat_F -> K(Past H,z,C_hat_F) -> delta_C_P`
然后 `Past DiT(z_P,C_P+delta_C_P)` 生成同一份16步动作块。
在线action loss通过DiT、K反传到R。K出口零初始化，初始行为与Past完全一致；
首步R的纯action梯度为零是预期现象，K出口更新后必须非零并记录。
原Future encoder只供固定辅助目标/oracle证据；其DiT本轮不参与读出。

## “多走几步”的三种不同含义

1. 多个随机tau/noise上的FM loss：降低Monte Carlo估计噪声，不是顺序推理。
   实现 `--fm-draws`，取平均而非直接叠加；本轮固定1。
2. 动作采样多步展开：从纯Gaussian噪声走4个Euler区间，对最终动作计算MSE，
   全程可微、不detach中间x、不使用真值动作初始化。本轮独立消融。
3. R/K来回多轮修正：不同于以上两种，本轮**不做**。先验证一轮闭合路径，
   没有收益时继续循环不保证新增信息，也可能放大误差。

标准FM已对整个16步动作块监督，并不只监督第一个物理动作。
评测仍统一用原10点网格（9个Euler区间），不因训练辅助改评测协议。

## 模型

R沿用SharedConnector的正向结构（256宽、3层），逆向专属适配器冻结。
K：128宽、2层TransformerDecoder；Past H/z作为query，标准化Future C作为memory；
4heads、FFN×4、dropout0；残差输出乘训练集Past条件std，末层严格零初始化。
16个query仍不是16个时间节点。各组R/K随机初始化匹配；不用旧R挑选checkpoint。

## 四组固定对照

同一813训练窗口、211验证窗口，按episode隔离；seed42、batch8、LR1e-4，
各1000步，alignment权重.1；最终1000步checkpoint作主评测：

| run | 在线K条件 | 真实Future辅助 | 额外采样loss |
|---|---|---|---|
| rk_adapter_01 | 固定训练均值 | 无 | 无 |
| rk_predict_01 | R预测Future | 无 | 无 |
| rk_teacher_01 | R预测Future | 有 | 无 |
| rk_teacher_roll4_01 | R预测Future | 有 | 4区间，权重.25 |

teacher的action loss为 `(online_FM+oracle_FM)/2`，避免直接翻倍总动作梯度尺度；
`+.1*future_alignment`。本轮不额外加行为蒸馏，避免同时修改多个机制。
rollout组只给在线分支增加 `.25*terminal_action_MSE`。
匹配更新次数/样本/初值，不是等算力；不能把额外计算与额外监督混为同一因果。

### 实现审查增补：等参数的无Future监督控制

在正式最终动作结果出现前固定增补 `rk_latent_01`：
使用与predict完全相同、可训练的R+K，`mode=predict`，但alignment权重0，
也无oracle。R输出仅是普通中间latent，不称其为已学到Future。
目标坐标的全局训练均值/std保持相同，**没有逐样本Future监督**。

原因：仅比较1.21M参数的K适配器与4.91M参数的R+K，无法排除容量增加。
该补充控制与predict比较，才能进一步区分“多一个可训练模块”和
“Future辅助监督”的影响。仍是1000步、相同初值/样本/动作噪声。
这组在原四组脚本之后独立执行，不修改正在执行的脚本或训练源码。

## 检查

- 初始 online FM与Past逐值一致，delta=0。
- 步0/1/10/100/1000记录 **仅在线action FM** 对R/K的梯度，不混入alignment/oracle。
- 单元测试覆盖来源隔离、oracle不泄露、零初始化、随后R梯度、reader冻结、
  multi-tau平均、Euler链路未detach、save/load。
- 全部实际动作评测使用相同样本/噪声及Past读出；旧Past固定参照须一致。
- 区分普通适配器收益、预测Future增量、oracle回溯增量、多步采样loss增量。

本轮是一训练/采样种子的筛选，不能当作稳健性或闭环证据。
若oracle校正也不优于普通适配器，不扩大训练或立即做蒸馏。
