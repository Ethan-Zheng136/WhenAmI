# 因果学生正式20k：实验前固定方案

**命名澄清：本轮为固定预算的20k缓存学生对照，“正式”不表示全量主训练。**
数据量、冻结范围和WoG预算差异见 `TRAINING_SCOPE_CLARIFICATION.md`。

日期2026-10-02。教师已验证，用户准备集群资源，现准备一键16卡提交。
独立新文件，不改原Stage1/2、Reader Study教师或封存测试集。

## 与WoG的方法关系与预算

核对归档WoG原文§3.3及附录B.1：
Stage II联合动作监督与Future条件对齐，未来编码目标冻结，实际策略表征可学习；
论文仿真训练是Stage I OXE100k、Stage II Bridge/Fractal50k，batch1024。
本轮借鉴联合监督，不谎称照搬完整WoG：我们的VLM仍冻结，只学习R/K和DiT适配器。

本轮20,000个**新增**optimizer steps，batch32，每观测4个独立FM噪声/时间抽样，
相当于每更新128个FM实例，但仍只有32个独立观察样本，不把两者混作batch128。
新更新共640,000窗口抽样，约39.1次遍历当前16,384窗口缓存的等效量。
与WoG full-dataset/global-batch预算不同，20k不是“等于WoG”或全OXE训练。
大缓存内反复抽样可能过拟合，开发曲线必须保存；主结果固定最终20k，不挑最好。

## 统一初始化与教师

四个训练种子42/43/44/45，每个种子的所有学生都：
- 从对应 `reader_study/runs_01/adaptive_mean_seedS/checkpoints/step-005000.pt`
  加载K和DiT适配器：不含逐样本Future训练，避免teacher权重初始化混淆；
- 新建相同随机初始化R（256宽3层）；
- 初始化优化器，无缝resume的是本轮显式断点功能，不是父run的5k优化器；
- 原Past/Future编码器、VLM及DiT原参数冻结，训练R/K与24层rank32新适配器。

教师分别是同seed、相同数据/预算的最终5k：
- past教师：`adaptive_mean_seedS/.../step-005000.pt`；
- future教师：`adaptive_future_seedS/.../step-005000.pt`。
严格加载、全文小checkpoint SHA256校验，推理eval/no_grad，训练不得更新。
不是挑表现最好的teacher seed，也不是复用ReaderStudy未匹配的数据。

## 4组×4种子=16个独立单卡实验

| 组别 | 监督 | 回答 |
|---|---|---|
| latent | action FM | 等参数普通微调，不要求中间latent是Future |
| forecast | action FM + .1 Future alignment | Future预测本身是否有益 |
| distill_past | 上组 + .3 past-teacher velocity蒸馏 | 排除一般蒸馏/正则化收益 |
| distill_future | 上组 + .3 future-teacher velocity蒸馏 | 真实未来教师是否提供额外有效监督 |

`distill_future` 不冒称已经验证了数学可逆或跨时间动力学；
这里的回溯是利用真实后续证据指导“当前应怎样行动”。
四组可训练容量、初始化、主动作loss权重、batch、噪声和更新数匹配；
蒸馏组额外一次冻结teacher前向，不是完全等算力。
主要比较：forecast vs latent，distill_future vs forecast，
以及distill_future vs distill_past。

## 梯度和信息边界

```text
学生在线：
Past H/z -> R -> C_hat_F -> K(Past H,z,C_hat_F)
                                   |
                          C_P + delta_C_P
                                   |
                     Past DiT + trainable adapters -> action FM

训练教师：
Past H/z/C + true C_F -> frozen oracle K/DiT -> velocity target (stopgrad)
                                                   |
学生velocity -------------------------------------+-> distillation
```

- 学生 `forward/sample` 仅接收Past H/z/C，不接收action标签、加噪action、Future z/H/C。
- noisy action仅在DiT的标准FM输入中，R/K不接收它。
- 学生和教师使用同一x_tau、tau、z_P；teacher真实Future输入仅供target分支。
- 注意旧 `EvidenceProbe.forward`会detach证据；学生必须直接调用K保留R→K梯度，
  已专门实现，避免出现“辅助loss在训练R、action却被截断”的假象。
- 专家action FM始终保留；teacher不是完美上界，不用teacher目标替掉真值监督。

## 优化/保存/评测

- LR峰值1e-4；500步线性warmup，随后cosine至1e-5；AdamW weight_decay0，clip1。
- BF16训练，FP32动作采样；无新增多步rollout损失或循环模块。
- 每1000步开发FM；每2500步存R/K/adapters+optimizer+RNG的小checkpoint。
- 步0/1/100及每5000步检查**纯action loss**对R/K/adapters梯度。
- 最终20k按3个采样seed（42/43/44）评2048开发窗口，9个Euler区间。
- 同时保留16个训练种子/组别的全部逐样本预测及r/R²/首动作/gripper。
- `--resume`只能同参数、同预算、同teacher/hash和相同训练源码续跑；
  自动回到本run最新完整checkpoint，截去其后的训练日志并重放。
- 封存2048测试窗口本轮不提特征、不评测。

本次不需要重新prepare；复用ReaderStudy已完成的train/dev特征缓存。
教师与无Future基线的旧噪声42结果可作参考；因学生额外20k训练，
结论必须以本轮同预算对照为主，不能只比旧5k基线宣布进步。
