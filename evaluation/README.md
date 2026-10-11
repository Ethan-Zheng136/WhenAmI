# Past Stage 2 离线评测

2026-10-02。独立入口，不修改训练代码/旧checkpoint/旧compare结果。

## 正式首轮结果（2026-10-02）

`past_s2_20k_eval_01/results.json` 已 complete=true；16卡任务完成六组评测。
Fractal为256窗口/68 episodes，Bridge为256窗口/72 episodes；noise seed=42。
逐样本NPZ中的样本ID、动作目标、seed配对一致，独立重新计算指标与JSON一致。

| 数据集 | variant | r_6 | R²_6 | normalized MSE_6 |
|---|---|---:|---:|---:|
| Fractal | Past | 0.625482 | 0.302811 | 0.063267 |
| Fractal | Past static | 0.603321 | 0.267628 | 0.066711 |
| Fractal | Future oracle | 0.691984 | 0.409901 | 0.051240 |
| Bridge | Past | 0.549192 | 0.145122 | 0.094938 |
| Bridge | Past static | 0.525112 | 0.117085 | 0.100589 |
| Bridge | Future oracle | 0.563213 | 0.169258 | 0.092126 |

完整16步动作的归一化MSE，相对static降低5.16% / 5.62%。
第一步动作MSE：Fractal降低8.74%，Bridge反而升0.41%，不能声称所有指标都改善。

补充只读分析：对相同预测按episode成组配对bootstrap（5000次，seed=20261002），
保持window-weighted MSE，95% percentile区间：

| 差值（Past减对照，负数更好） | Fractal | Bridge |
|---|---|---|
| Past - static | [-0.006713, -0.000323] | [-0.009425, -0.001705] |
| Past - Future oracle | [0.008094, 0.016265] | [-0.002055, 0.008097] |

区间只反映本次episode样本的重采样不确定性，不包含不同训练种子/噪声种子的变异。
支持的判断：本轮离线协议下Past是有效起点，真实历史相对static有小幅一致收益；
Fractal仍明显落后于Future参照，Bridge点估计接近但不是等价性检验。
不支持的判断：已超过WoG、已形成时间线、闭环有效、比旧flow分数的改善完全来自架构。
旧flow需在同一协议下重测，不能跨协议直接归因。

## 一次任务测什么

| variant | checkpoint | 输入 | 解释 |
|---|---|---|---|
| `past` | Past累计20k | 真实历史+当前+语言 | 当前可部署路径的离线动作误差 |
| `past_static` | 同一Past | 所有历史位置复制当前帧 | 历史输入干预，不是另训的单帧baseline |
| `future_oracle` | 自训WoG Stage1 20k | 真实未来+当前+语言 | 有特权信息的参照，不保证数学上的上界 |

旧 `eval_conditioned_dist.py` 不适用于这个Past版本。不要传旧npz或把
Past checkpoint误识别为recurrent。这里只通过 `past_frames` 调用Past推理接口；
真实未来和目标动作不传入Past条件模块。

## 协议

- 原始RLDS，经项目中同一份OXE standardization：RT1夹爪转绝对开合，
  Bridge动作按相邻EEF状态重标。先保留标准化后的raw动作，避免先clip再反归一化丢掉真值。
- Future的DINO图像严格为 `[t+3,t+7,t+11,t+15]`；其VAE会自行前置当前帧。
  不把当前帧再作为第五个“future”传进去。
- Past采样从checkpoint内的元数据恢复，当前默认 `[t-15,t-11,t-7,t-3]`，
  当前帧单独输入；原有Past路径负责构建按时间顺序的VAE视频。
- 所有模型比较同一中心时刻的16步专家动作 `a_t ... a_{t+15}`。
- 默认每数据集最多256个样本，每episode最多4个，完整窗口，不含padding。
  所以对短episode和边界情况没有覆盖，不能声称是全分布性能。
- Fractal：`train[95%:]`，**原模型训练过完整train，不能说这是训练未见的测试集**。
  Bridge：`val`；预训练数据是否包含相关轨迹未审计，也不声称绝对无污染。
- 先单CPU进程读取/解码样本到PFS，再torchrun分发；不会让16个rank各自重扫数据。
- `frames.npy`保留全部采样帧供不同组取用，但选择函数保证只向对应模型传允许的子集。
- 同一样本、同一seed下三组使用配对噪声；seed由dataset/episode/timestep派生，
  不依赖rank数。多个seed按**指标求平均**，不把预测平均成额外ensemble。
- 模型保持eval，调用实际 `predict_action`。记录checkpoint完整路径、大小/mtime、
  原run config、统计文件hash、关键源代码hash、样本episode/timestep、每seed指标。
- 保存完整逐样本预测，便于后续按episode做paired bootstrap或错误分析。

## 指标

- 前6维逐维Pearson r的均值；
- raw标准化动作上的前6维variance-weighted R²；
- 各维raw RMSE；
- 使用统一参考统计的Q01/Q99归一化MSE和MAE；
- 第一个动作的归一化MSE；
- 标准化为绝对0/1后的gripper准确率。

退化常数维的相关系数用JSON null，不生成非法NaN。
这些是logged expert-action预测指标，**不等于闭环任务成功率**。
`past_static`下降只能说明这次输入干预影响模型；因其可能偏离训练分布，
不能单独当作“历史必然带来泛化收益”的证据。后续应有另训的current-only对照。

## 平台命令

**1节点、16卡**，同原PPU镜像：

```bash
NPROC_PER_NODE=16 N=256 RUN_ID=past_s2_20k_eval_01 \
bash /home/zhengguantian/WoG_fresh/run_eval_past.sh
```

默认Past权重是从5k继续15k的最终checkpoint：
`stage2_16rank_probe/past_s2_16rank_20node_from5k_01--image_aug/checkpoints/step-015000-epoch-01-loss=0.2112.pt`。
这相当于累计20k，但发生过优化器重置/输入配置更换。
默认Future是 `stage1/baseline_v2--image_aug/checkpoints/step-020000-epoch-02-loss=0.1076.pt`。
Past由该Future初始化后额外训练，训练预算不相等，不能按纯架构公平对比解释。

稳健性复核可增加噪声seed（推理成本相应增加）：

```bash
NPROC_PER_NODE=16 N=256 NOISE_SEEDS="42 43 44" \
RUN_ID=past_s2_20k_eval_3seeds_01 \
bash /home/zhengguantian/WoG_fresh/run_eval_past.sh
```

也可覆写 `PAST_CKPT`、`FUTURE_CKPT`，或 `VARIANTS="past past_static"`。
输出在 `/mnt/pfs/zhengguantian/WoG_v2/outputs/past_eval/<RUN_ID>/`：

```text
prepare.log / evaluate.log
samples/manifest.json
samples/<dataset>/{frames,actions,episode,timestep}.npy
samples/<dataset>/instructions.json
results.partial.json     中途已完成组
results.json             全部完成且 complete=true
results_<dataset>_<variant>_predictions.npz
```

已有RUN_ID拒绝覆盖。`N=256`表示每数据集最多256个合格窗口，不是原始episode数；
若扫描1000个episode仍不足，会如实报告实际样本数。
第一轮先看Past本身是否有效、相对静态历史有无增益，再看与特权Future的差距。

## 实现验证（2026-10-02）

- `test_past_protocol.py`：10项通过，覆盖输入可见范围、真实时间步、
  动作归一化、指标退化处理、配对噪声和含空rank的结果排序。
- 实际三进程Gloo `all_gather_object` 测试通过，含一个没有样本的rank；
  16卡NCCL整任务尚待平台运行。
- 两个数据集各2个真实RLDS样本、完整Past/Future checkpoint、三组输入均跑通，
  输出6组有限指标、逐样本预测和 `complete=true`。
  这是**接线测试，不是性能结论**，不要引用这些极小样本分数评判模型。
- 本地验证目录：
  `/mnt/pfs/zhengguantian/WoG_v2/outputs/past_eval/smoke_20261002/`。
