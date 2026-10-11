# WhenAmI：当前结论、改进方向与 WoG 对比交接

日期：2026-10-03 UTC。适用于已经在另一台 NVIDIA 服务器准备部署的同事/agent。

## 0. 先分清两条实验线

**主 Stage3 和刚结束的 real-timeline 实验不是同一个模型。**

- 主 Stage3：OXE27 原图训练，VLM、Past QFormer、R/K、完整动作 DiT 联合更新。
- real-timeline：旧 Stage2/Past 端点完全冻结，只训练约 1.26M 的时间线与条件残差；没有加载主 Stage3 的更新权重。
- timeline 的负结果不能直接推导为主 Stage3 无效；主 Stage3 的开发集改善也不能证明 timeline 的后向修订成立。

已完成的 timeline 实际动作采样诊断：

| 对比 | macro MSE6 的相对改善，正数为好 |
|---|---:|
| Past → direct_history | -1.367% |
| Past → revision | -1.243% |
| direct_history → revision | +0.122%，配对区间跨零 |
| revision 关闭更新 → 正常 revision | +0.516% |

结论：当前冻结 reader 的条件残差方案没有净收益；分支被使用不等于方法有效。
这是 27 域、351 episode、3 个推理噪声 seed、1 个训练 seed 的内部开发诊断，
不是闭环结果，也不是三个独立训练重复。

完整结果：`WHENAMI_TIMELINE_FINAL5K_RESULTS_20261003.md`。

## 1. 当前主 Stage3：35k 可立即作为评测对象

本次查看时训练约 36.8k/50k，仍在正常更新。固定使用最新完整的 **35k**，
不要在评测开始后自动跟随目录中的最新 checkpoint。

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/
  oxe27_forecast_zero2_320_50k_save5k_01/
    checkpoints/step-035000/
    development/step-035000/
```

`COMMITTED.json` 为 complete=true、step=35000。
CPU 已读取 DCP metadata：1664 个条目，学生张量均为 FP32，
包括 `vlm`、`past_qformer`、`net`、`r`、`k`，
模型 DCP 目录约 30.095 GiB。不包含训练用的独立 Future QFormer 或冻结历史 DINO/Wan。
这里只核查了目录/metadata，**不是全量模型加载、张量完整性或导出验收**。

同一 Bridge/Fractal 开发样本、纯噪声动作采样结果：

| Stage3 step | Bridge MSE6 ↓ | Fractal MSE6 ↓ |
|---:|---:|---:|
| 100 | 0.04223858 | 0.05428590 |
| 20,000 | 0.03560167 | 0.04680466 |
| 30,000 | 0.03476056 | 0.04555271 |
| 35,000 | 0.03430008 | 0.04465748 |

35k 的 Pearson r6 分别为 0.84047 / 0.75189；
R² 分别为 0.68714 / 0.52252。这里的 r6 是逐连续动作维相关系数的均值，
不是成功率；R² 也不是相关系数的平方。

以上四个评测点本次重新从预测 NPZ 计算，核对了相同样本 ID、episode、
timestep、动作真值。这是固定开发集的描述性训练趋势，不是 WoG 对比结果。
每域 1024 窗口，BF16 模型计算、FP32 采样状态、9 个 Euler 区间，噪声 seed42。
不能与 timeline 的另一套样本/聚合方式直接比较绝对 MSE。

本次机器可读审计：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/
  compare_readiness_20261003/AUDIT.json
  compare_readiness_20261003/audit.py
```

## 2. WoG 应当选哪个 checkpoint

**可部署基线选本地 WoG-A，即该发布包标注的 Stage II；不要选真实 future oracle。**

```text
/mnt/pfs/zhengguantian/WoG_v2/ckpts/WoG-A/
  checkpoints/50k-1229.pt
  config.json
  dataset_statistics.json
```

本地参考实现：

```text
/home/zhengguantian/WoG_v2/refs/code/WoG/
```

该仓库 README 自称 unofficial implementation。论文复现需要如实标注所用
仓库 commit、资产来源和任何部署修复，不能自动等同于作者内部最终模型。

在这份**本地实现**中，Stage II 的动作路径是：

```text
当前图像 + 指令 ── VLM ── z ── DiT ── 动作
                         │
训练时额外语言隐状态 ── oformer ── 对齐冻结的真实 future 条件
```

具体而言 `train_venc=False` 时 DiT 的 `use_per_attn=False`，
推理调用 `action_model.inference(z)`，不读取 future 图像，也不把
oformer 输出作为动作头的必需条件。
“训练时使用 future 辅助监督”不等于“推理时要拿真实 future”。

我们的主 Stage3：

```text
历史 + 当前 ── Past ── H_P / C_P ───────────────┐
当前 + 指令 ── VLM ── z                       │
                    H_P,z ── R ── C_F_hat     │
                    H_P,z,C_F_hat ── K ── ΔC_P
                                              │
                     C_P + ΔC_P、z ── DiT ── 动作

训练专用：真实 future ── 冻结 encoder/QFormer ── R 的对齐目标
```

所以工程上 Stage3 checkpoint 已经是可部署策略权重，**没有必须再跑 Stage4 的要求**。
但“训练权重已经具备”与“跨机器独立推理包已经验收”是两回事。

## 3. 对比矩阵：性能和归因分开

### 3.1 立即可组织的性能对比

| 名称 | 推理允许输入 | 用途 |
|---|---|---|
| WoG-A | 当前图像、指令 | 发布基线重评 |
| Past 原端点 | 历史、当前、指令 | 我们增加 Stage3 前的基础策略 |
| WhenAmI Stage3-35k | 历史、当前、指令 | 现在部署调试、初步对比 |
| WhenAmI Stage3-50k | 历史、当前、指令 | 完成后按固定规则做最终评测 |

Past 端点：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage2_16rank_probe/
  past_s2_16rank_20node_from5k_01--image_aug/
    checkpoints/step-015000-epoch-01-loss=0.2112.pt
```

文件名 15k 对应这一训练段的更新数；该 Past 累计约 20k，不应误写成只训练15k。

这些模型输入信息量与训练历史不同，能回答“哪个实际策略表现更好”，
不能单独回答“未来预测/后向推理带来了多少收益”。
WoG 的 50k 和我们的 50k 也不自动等预算：global batch、前序训练、初始化均要披露。

### 3.2 机制诊断与下一轮公平训练

先对固定 Stage3 做诊断：

- 关闭 K 的 delta；
- 保持合法历史，替换 R 的 predicted future 为记录的均值；
- 保持其他路径，历史改为重复当前帧；
- 若做打乱，按域/任务约束并记录选择规则，不把跨机器人错配当标准反事实。

这些是**推理干预，可能分布外**，只能说明模型依赖，不能替代消融重训。
关闭 K 后的模型也不是 WoG-A，更不是原 Past：其 VLM/Past/DiT 都已经更新。

归因需要后续匹配训练：

1. Past-only joint：同初始化、同数据流、同 full-policy 优化预算。
2. 同结构 latent：保留 R/K 等容量，但不做 future 对齐。
3. 当前 forecast：同样配置，加真实 future 辅助目标。

比较 2/3 更直接检验 future 对齐目标，1/3 是整体方法消融。
正式做之前需分别检查优化器、有效参数数、初始化和梯度，不以配置名字代替验收。

## 4. 离线动作对比：先在同一批样本重评

不需要全量下载 OXE。

源开发数据：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_reader_study/data_01/train_dev/
```

只能读取该目录中明确属于 dev 的 `valid_indices.npy`。
目录还混有训练窗口和旧缓存，**不要原样整目录传输**；
应生成仅包含两域各1024评测窗口的小包，去掉 future 输入，
保留历史/当前图像、指令、动作真值、原始样本 ID/episode/timestep。
不读取封存 test，不把这些窗口宣称基础预训练从未见过。

固定协议：

1. 两域分开报告；相同样本、相同真实动作目标、相同采样噪声数组。
2. BF16 模型计算、FP32 Gaussian 初值与积分、9 个 Euler 区间、16×7 动作。
3. 先 seed42 复核既有表，再一起补 seed43/44；推理种子不是训练重复。
4. 不使用目标动作加噪来代替部署采样；不允许 future/未来专家动作进入策略。
5. 统一动作裁剪、夹爪阈值及机器人映射；保存后处理前的连续输出便于审计。
6. 各模型先按其训练统计解码到同一物理/标准化动作坐标，再用固定的评测统计评分。
7. 本次核查：Bridge/Fractal 两域 WoG-A 与 Past 的 q01/q99/mask/min/max 相同，
   mean/std 不同。当前是 BOUNDS_Q99 协议，不得擅自切换成均值方差反归一化。
8. 输出 MSE6、MAE6、首动作 MSE6、r6、R²、逐维 RMSE、gripper accuracy；
   episode 配对统计，不把同一轨迹的每帧当独立样本。

建议每个方法保存统一 NPZ：

```text
dataset, sample_id, episode, timestep, noise_seed
predictions_normalized, predictions_raw, target_raw, action_mask
```

并保存 JSON manifest：checkpoint/hash、代码版本、precision、sampler、
归一化统计 hash、原始样本 manifest hash。

## 5. SIMPLER：真正的闭环性能检验

### 5.1 顺序

1. 先跑通 WoG-A；用视频检查相机、动作方向、夹爪和场景。
2. 接入 Past/Stage3 的历史缓存，先固定一条 episode 验收。
3. WoG-A、Past、Stage3-35k 在同一任务/初始状态列表运行。
4. 35k 用于开发；50k 完成后固定为最终学生，不逐任务挑不同步数拼表。

从本地 WoG 的 `deploy/scripts/` 固定任务清单和全部命令参数，
先完成 Visual Matching，再独立完成 Variant Aggregation；两种协议不混表。
小 smoke 成功不叫主表完成。需要 downstream SFT 的其他 benchmark 单独规划，
不能因为 SIMPLER 能直接运行，就说所有论文表项都不需要适配。

### 5.2 要统一的细节

- 相同环境/子模块版本、camera、reset 初始条件、控制频率和 max steps。
- 相同动作重规划/ensemble 和机器人动作转换；如果同时跑发布默认设置，
  将“发布协议复现”和“统一协议横比”分开列，不静默削弱某一基线。
- 该本地 WoG wrapper 默认 adaptive ensemble：WidowX horizon7、Google horizon2；
  Google sticky gripper repeat10。不要只在一边打开。
- 本地参考 WidowX 脚本使用 control-freq5；所查看 Google pick 脚本使用3；
  正式实验以固定任务命令为准，不把不同机器人强行改成同一频率。
- WhenAmI 历史偏移为 `[-15,-11,-7,-3]`；缓存单位是实际控制观测步，
  不是“最近调用策略4次”。episode 重置时清空，历史不足重复最早帧。
- 记录动作块执行期间的观测，明确训练帧节奏与环境控制步的对应关系。
- 测成功率、任务分解、完整视频、推理延迟、峰值显存和错误日志。

本地 WoG 原 wrapper 引用了未在构造器定义的 `cfg_scale/use_ddim/num_ddim_steps`。
另一服务器可能已经修好；这里无法远程确认。
不要为了绕过异常随意补默认值并改变算法；核对真实 sampler，
使修复有测试和单独代码 diff。

## 6. 权重迁移：只传推理需要的内容

**本次没有生成或验收独立推理导出。** 另一服务器负责人若已有 exporter，
直接固定35k接入，不要重复重写；先回传导出 manifest 和固定输入回放结果。

Stage3 最小来源：

- 已提交 checkpoint 中的 `model_dcp/` 和 `COMMITTED.json`；
- 完整学生：VLM/Past/R/K/DiT（含 `net.blocks.*.base` 和 adapter 权重）；
- 对应结构配置、tokenizer、dataset_statistics；
- 冻结历史 DINO/Wan；
- 不依赖原服务器绝对路径的推理代码。

**不要传所有 optimizer/rank RNG，不要只传一个 DCP shard，也不要只传 R/K。**
若接收方自己导出，模型 DCP 必须整目录（含隐藏的 `.metadata`）传输，
不需要320张卡恢复训练；导出是模型权重组装，不是训练 resume。
BF16 导出会减少大小，但未经数值对齐不宣称可用。

冻结视觉权重：

```text
/mnt/pfs/zhengguantian/WoG_v2/ckpts/vision/dinov2_weights.pth
/mnt/pfs/zhengguantian/WoG_v2/ckpts/vision/Wan2.1_VAE.pth
```

必须验收：

- 独立包缺失/意外 key 检查，不用无记录的 strict=False 掩盖漏权重。
- 无 OXE/optimizer/Future teacher 仍能推理。
- 固定图像和**相同初始噪声数组**，核对 H_P/C_P/z/C_F_hat/最终条件/velocity/动作。
- 跨硬件不要求逐 bit 相同；测误差、记录容差后决定是否通过。
- NVIDIA 环境中 `past_frames` 改动确实到达模型；未来图像不会到达模型。
- 导出和传输文件 SHA256。

环境交接仍见 `WHENAMI_NVIDIA_EVAL_HANDOFF.md`；本文覆盖其20k时点的旧快照。

## 7. timeline 下一步怎么改：提案，不是已验证成果

不建议原封不动扩大旧实验、继续只看 FM loss。代码中有四个具体限制：

1. **固定 reader**：整个旧 Past/VLM/DiT 冻结，只有条件残差可变；
   action loss 能反传到残差并不等于冻结 reader 能有效利用新时间状态。
2. **innovation 没有显式校准**：现在直接用 `current-prior`，
   但 transition 的监督约束的是 `prior_condition(prior)` 对齐下一时点 `C_P`，
   不是 prior 与 current 的128维坐标一致。不能把这个差天然当测量误差。
3. **修订后没重推**：更新历史 bank 后只让 attention 读它；
   下次 transition 仍用原 `filtered[-1]`，没有从修订事件沿真实已执行动作重新传播。
4. **时间跨度很短**：3个锚点、相邻gap1–4，历史视野高度重叠；
   这不足以充分检验“迟到证据改变过去解释并改变当前决策”的长时机制。

这些是实现限制，不是已经做因果实验确定的失败原因。

建议分开验证：

### A. 先检验接口/读出问题

同一初始化和数据预算，direct_history 与 revision 同时对比：

- reader 冻结；
- reader 允许同样规模的低秩/残差适配。

先用原 Past 初始化做这个隔离实验，避免同时换 Stage3 初始化而混淆原因。
训练期间定期做固定开发集的**纯噪声动作采样**，而不只记录 FM/transition loss。
零条件残差时必须复现对应基础策略。

### B. 校准证据，再做真正的回溯传播

优先尝试在同一个已定义的条件坐标中构造：

```text
predicted_C[t] = transition 的条件预测
observed_C[t]  = 当前合法观测得到的冻结目标
innovation[t] = (observed_C[t] - predicted_C[t]) / 同一训练统计尺度
```

不要减去两个未约束为同坐标的隐向量。
随后用真实新观测的 innovation 更新历史事件，并在有限回溯窗口内，
沿已经执行的动作序列重推当前状态；所有动作预测仍只使用截至当下的证据。

```text
历史证据 + 已执行动作 ── transition ── 预测当前状态
当前真实观测 ───────────────────────┐
                                  ↓
                       校准的 observation innovation
                                  ↓
                      修订相关旧事件 b[k|t]
                                  ↓
                 沿已执行动作有限步重推到当前 b[t|t]
                                  ↓
                  可共同适配的动作 reader → 当前动作块
```

“用最新观测重新解释过去”不是用未来信息预测已经发生的旧动作；
有回溯证据/旧执行动作的状态不能拿去重建其已见过的旧动作标签。
自生成的 predicted future 也不是独立新观测。

### C. 与主 Stage3 衔接

确认上述接口收益后，所有新方法共享同一个固定 Stage3 checkpoint，
从它的**实际最终条件**上做零初始化改动，保留强基线。
direct-history 强对照必须同时获得一样的信息、reader适配和计算预算。
更长轨迹/遮挡/失败恢复的评测要先定义合法输入和标注协议，
既看整体标准性能，也看预先定义的困难子集，不能事后挑获益样本。

如果要增加 hindsight teacher，先证明老师在同预算下确有动作增益，
再蒸馏动作分布或可预测的状态部分；不硬逼因果学生复现不可预知的未来结果。

**当前不改变主 Stage3，不重启已有作业，不把这些提案写成已经实现或有效。**

## 8. 另一服务器请回传的最小结果

```text
WoG-A checkpoint/hash：
Stage3 checkpoint step/hash（建议先固定35k）：
推理导出manifest：
固定输入/固定噪声回放最大误差：
历史缓存验收：
两域离线预测NPZ/指标：
固定任务清单和部署代码diff：
WoG-A/Past/Stage3闭环成功率、episode视频：
峰值显存、单次推理延迟：
仍阻塞的具体traceback：
```

源服务器本轮工作仅为代码复核、metadata读取、开发预测复算和本交接。
没有修改训练源码、数据或权重，也没有接触封存测试集。
