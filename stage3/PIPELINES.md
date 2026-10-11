# WoG / 旧WhenAmI Stage3 / 当前R+K：代码对应的pipeline

2026-10-02。图对应当前本地实现，而非把完整WhenAmI研究目标画成已实现。
`[冻]`=该阶段冻结，`[训]`=该阶段可训练，`--->`=训练期特权/辅助支路。
以下“旧WhenAmI”指共享R条件互译版，不是flow_v2/v3全部历史架构。
**WoG Stage II ≠ WhenAmI Past Stage2。**

## 1. WoG

### Stage I：用动作监督训练Future条件

```text
当前图像 + 指令 ──> VLM [训] ──> z ──────────────────┐
                                                    │
当前 + 真实未来 ──> 视觉基础模型 [冻]                 │
                          │                         │
                          v                         v
                    QFormer [训] ──> C_F ──> DiT-I [训] ──> action loss
                                      ^
                                      └── action loss经DiT反传，塑造Future条件
```

这里未来是真实观测，不是模型想象；直接使用这条路径是特权参照。

### Stage II：未来辅助监督当前VLM；部署只用当前观测

```text
当前图像 + 指令 ──> VLM [训]
                      │
                      ├──> cognition z ──> DiT-II [训] ──> action loss
                      │
                      └──> hidden tokens ──> oformer [训] ──> C_hat_F
                                                               │
                                                               v
真实未来 ---> 冻结的Future条件编码支路 ---> C_F -----------> 对齐loss

部署：
当前图像 + 指令 ──> VLM ──> DiT-II ──> 动作
```

按本地参考代码，`train_venc=False`关闭DiT的per-attention：
**不是把预测C_hat_F直接送进Stage I DiT来部署。**
oformer对齐与动作loss共同影响VLM；oformer不在部署动作路径中。

代码依据：
- `WoG_v2/refs/code/WoG/vla/vla.py`：`forward`的action/obs loss及`predict_action`。
- `WoG_v2/refs/code/WoG/action_model/action_model.py`：`use_per_attn`条件。

## 2. 我们的Stage1 / Stage2基础

```text
WhenAmI Stage1 / Future：
  当前+真实未来 -> Future encoder/QFormer -> H_F -> C_F -> Future DiT(z_F) -> 动作

WhenAmI Stage2 / Past：
  历史+当前     -> Past encoder/QFormer   -> H_P -> C_P -> Past DiT(z_P)   -> 动作
```

两者均用action监督训练各自条件路径，Past使用`train_venc=True`；
并不是WoG Stage II的oformer蒸馏。
到了下述Stage3，这些编码器、VLM、DiT全部冻结。
`z_P/z_F`分别来自各自checkpoint的当前图像+语言VLM，都不读取真实未来。

## 3. 旧WhenAmI Stage3：共享R的条件互译

```text
                      正向
Past H_P,z_P [冻] ──> R [训] ──> C_hat_F ──> Future DiT(z_F) [冻] ──> L_action+
                                  │
                                  └── 对齐固定 C_F

                      逆向
Future H_F,z_F [冻] -> 同一个R ──> C_hat_P ──> Past DiT(z_P) [冻] ──> L_action-
                                  │
                                  └── 对齐固定 C_P

可部署信息流：
历史+当前 -> Past H/z -> R(+) -> C_hat_F -> Future DiT(z_F) -> 动作
```

两个方向只共享R参数；逆向输出不参与上面的正向动作生成。
两个DiT不融合/投票；这里的逆向也不是把物理动作倒放。
正向接口仍需两个checkpoint的当前VLM特征z_P、z_F。
这是同一中心时刻的条件互译，不是显式时间转移或状态时间线。

代码：`stage3/{connector,objectives,train,evaluate}.py`。

## 4. 当前WhenAmI Stage3候选：保留Past，前瞻后回溯修正

### 可部署信息流（已离线评测；Stage4在线执行包装尚未实现）

```text
历史 + 当前 ──> Past encoder [冻] ──> H_P ──────────┐
                                  │               │
                                  └──> C_P ───┐   │
                                              │   │
当前 + 指令 ──> Past VLM [冻] ──> z_P ─┐       │   │
                                      │       │   │
                    H_P,z_P ──> R [训] │       │   │
                                  │   │       │   │
                              C_hat_F │       │   │
                                  │   │       │   │
                                  v   v       │   v
                                 K [训] <─────┼── H_P
                                  │           │
                               delta_C_P      │
                                  │           │
                                  └──> 相加 <─┘
                                         │
                                   C_P + delta_C_P
                                         │
                                         v
                               Past DiT(z_P) [冻] ──> 动作
                                         │
                                    action loss
                                         │
                  R <──── K <──── 冻结DiT的输入梯度
```

简写：

```text
C_hat_F = R(H_P, z_P)
delta_C_P = K(H_P, z_P, C_hat_F)
action = D_P(z_P, C_P + delta_C_P)
```

K出口零初始化，最初等于原Past；只训练R/K，端点权重不动。
最终只有一个Past DiT；Future DiT不在这版动作读出中。

### 训练期额外oracle路径：与在线路径分开的一次调用

```text
当前 + 真实未来 ---> Future encoder/QFormer [冻] ---> C_F
                                                       │
                         H_P,z_P -----------------------┤
                                                       v
                                                   同一个K [训]
                                                       │
                                                delta_C_P_oracle
                                                       │
                                    C_P + delta_C_P_oracle
                                                       │
                                                       v
                                            同一个Past DiT [冻]
                                                       │
                                                L_action_oracle

另有：R预测的C_hat_F ---> 对齐loss <--- 固定C_F
```

**真实C_F替换预测C_hat_F用于另一次oracle调用，不是同时塞进在线K。**
当前teacher实验：`.5*(L_online+L_oracle)+.1*L_alignment`；
没有行为蒸馏，没有多轮R/K循环。4区间动作采样loss是单独的对照开关，
不是默认架构，更不等于R/K推理4轮。

代码：`stage3/{refinement,train_refinement,evaluate_refinement}.py`。

## 5. 思想对比与当前证据

| | WoG Stage II | 旧Stage3共享R | 当前Stage3 R+K |
|---|---|---|---|
| 在线观测 | 当前+语言 | 历史+当前+语言 | 历史+当前+语言 |
| Future作用 | 训练时塑造当前VLM | 预测另一侧条件 | 预测后用于修正Past条件 |
| 在线动作读出 | WoG Stage II DiT | 固定Future DiT | 固定Past DiT |
| 辅助分支是否参与在线动作 | oformer不参与 | 逆向R分支不参与 | K直接参与 |
| Stage3可训练部分 | 不适用 | R/适配器 | R/K |
| 是否已建立多时间锚点时间线 | 否 | 否 | 否 |

当前证据支持Past残差适配的部分离线误差改善，但普通/等参数无Future监督控制
也能得到类似收益；没有证明Future回溯是收益来源。详见 `REFINEMENT_01_RESULTS.md`。

WhenAmI的完整研究目标仍是：
`过去怎样走到当下` 与 `后续怎样确认当下` 对同一轨迹位置给出一致解释。
当前只做一个中心时刻；**16个QFormer query不是16个时间节点**。
