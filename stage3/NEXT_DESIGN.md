# Stage3 下一版候选：保留 Past，以未来证据回溯修正当下

2026-10-02。**独立R+K实验已实现，五组1000步对照及动作评测均完成。**
结果见 `REFINEMENT_01_RESULTS.md`：Past残差适配改善主动作误差，但等参数控制
尚未支持Future/回溯的增量收益；4区间采样loss本轮也没有改善。不要把接线
成立或普通适配器收益当作双向时间推理已经有效。
入口见 `train_refinement.py`、`evaluate_refinement.py`，
预先固定的实验计划见 `REFINEMENT_01_PLAN.md`。
真实Past DiT单PPU smoke与正式训练已验证：初始行为等于Past、K出口更新后
纯在线动作梯度到达R。36项单元/CPU分布式测试通过。
阶段编号不变：Future=Stage1，Past=Stage2，冻结端点的连接学习=Stage3，
在线管线接入/可选联合微调=Stage4。

## 为什么不是直接“让 R 更大”

Pilot01 的正/逆只在共享参数上相遇，逆向输出不进入在线决策。
初步梯度诊断也没有显示强烈的方向冲突，反而近乎正交。
这不能证明原因，但更值得改的是监督作用的位置，不是梯度投影技巧。

另外，Future 的特权信息不保证它的既有 DiT 比 Past 更强。
让 Past 预测 Future 后改用 Future DiT，可能丢掉已训练好的 Past 行为能力。

## 核心改动：预测未来用于重新解释当下，而不是替换 Past 策略

```text
冻结 Past:
  历史+当前 -> H_P、C_P
  当前+语言 -> z_P
                  \
                   R(Past H,z) -> C_hat_F
                        |             |
                        |             v
                        +------ K(Past H,z, C_hat_F) -> delta_C_P_online
                                                     |
                                C_P + delta_C_P_online
                                                     |
                                       冻结 Past DiT(z_P)
                                                     |
                                               在线动作 loss

训练时另有真实 Future:
  冻结 Future encoder -> C_F
                           |
                           v
             同一个 K(Past H,z, C_F) -> delta_C_P_oracle
                                       |
                          C_P + delta_C_P_oracle
                                       |
                            同一个冻结 Past DiT(z_P)
                                       |
                             特权路径的动作 loss
```

- R 做前瞻；K 做“用后续证据回看当下”的修正，不要求它是 R 的数学逆。
- Past/Future 编码器、VLM、两端现有 DiT 仍冻结；只训练 R 和小型 K。
- K 的末端残差投影零初始化，初始在线动作严格等于原 Past。
  第一更新主要进入残差头是预期现象，必须测试随后动作梯度能到达 R；
  R 自身的非零输出初始化和 future 辅助目标保留。
- `C_P` 不被替换成未训练的瓶颈，也不把真实未来输入在线分支。
- K 的 oracle 路径保留 Past 信息，而非拿较弱的 Future-only policy 当万能老师。
- 这一版仍是同一中心时刻的回溯校正，不冒充多时间节点推理。

## 损失先保持少而有明确角色

```text
L_online_action                         # 主目标：实际可部署分支
+ L_oracle_action                       # 让真实未来确实改善当下行为
+ lambda_F * align(C_hat_F, stopgrad(C_F))
+ lambda_D * behavior_distill(online, stopgrad(oracle))  # 后加的独立消融
```

行为蒸馏可比较相同加噪动作/时间上的固定 Past DiT 速度输出，不必硬匹配
所有条件坐标；真实动作 loss 始终保留。teacher 必须 stopgrad，不能把
降低 student-teacher 差异等同于降低真实动作误差。
oracle/student 共用 K 时 teacher 不是完全固定目标，这一事实必须记录；
如不稳定，应单独训练并固定 oracle K 再蒸馏，而不是暗中加EMA等机制。

已在单元测试与真实单PPU smoke中验证的梯度路径（尚非行为收益证据）：

```text
online action loss -> frozen Past DiT -> K -> predicted Future -> R
oracle action loss -> frozen Past DiT -> K
```

与旧版不同，回溯模块 K 确实参与在线动作路径，而不只是训练时的额外任务。
但“路径打通”本身仍然不是双向思想有效的证据。

## 必须防止被普通适配器收益蒙混过去

按顺序做小规模、相同 episode split 的消融，预先指定最终 checkpoint：

1. 原 Past（无可训练适配器）。
2. Past + 同容量 K，只给固定 Future 均值（普通 Past 适配器）。
3. Past + K + R 预测 Future，无真实 Future 回溯监督。
4. 第3组 + oracle K 路径；行为蒸馏再单独加，不同时改一堆因素。
5. 真实 Future 进入 K 的 oracle 评测：仅训练诊断，不是可部署得分。

**停止条件**：
- 如果第5组都不优于第2组，不应继续让 student 追这个 oracle。
- 如果第3/4组只优于原 Past、却不优于第2组，说明可能只是小适配器微调。
- 如果第4组不优于第3组，不得把普通前瞻预测的收益归于回溯监督。
- 若正向 R 可以被固定均值替代而几乎不影响行为，先诊断 K 忽略未来的捷径。

至少多个训练/采样种子；锁定设计后再准备新的 episode-disjoint 评测集。
与 WoG-A 的同集评测是性能参照，不能消除预训练预算差异。

## 如何继续走向 WhenAmI，而非长期停留在辅助任务

单中心校正成立后，再准备明确的双时间锚点 `t` 与 `t+delta`：
不同观测视角对同一轨迹位置作解释，并让该位置对应的动作窗口约束表示。
需要保存真实 delta、各位置动作窗口以及视角可见性；不能把现有16个
QFormer query 当作16个时刻，也不能把随机稀疏窗口称相邻时间点。

“想象未来”不会凭空增加观测信息。它的潜在价值是训练时借助未来证据
塑造可部署的表示，以及推理时提供结构化计算；是否有效必须以上述
普通适配器/均值/无回溯对照来证明。
