# WhenAmI present-token 实现说明（panel b）

代码：`/home/zhengguantian/WoG_fresh/whenami_present/`，启动脚本 `run_whenami_present.sh`。
没有改动任何已有的包，也没有碰正在跑的 160 卡训练。

## 1. 模块（按数据流）
| 组件 | 实现 | 初始化 | 参数量 |
|---|---|---|---|
| 冻结视觉特征 → QFormer(`local_encoder`) → 每帧 h(16×1024) | 沿用原实现 | A50k | 原有 |
| VLM z → NodeCognition.seed → 节点 | 沿用 timeline_force 的实现 | identity（A50k 里没有这个模块） | 17.8M |
| 因果 timeline（revise 版 UnifiedUpdater） | 提供过去节点 S_<t（versions[c-1]） | A50k | 原有 |
| **现在 token p_t^0** | 当前帧节点 + 可学习 embed（16×1024） | embed 零初始化 | 16K |
| **U_fwd** | 原 timeline updater，query 换成 p_t^0，memory 为过去节点加当前观测 | A50k | 共享 |
| **U_bwd** | 新的 UnifiedUpdater，query 为 p_t^0，memory 为未来节点 | 拷贝 A50k timeline 权重 | 3.26M |
| **精度融合** | 每个 token 预测 log 方差 s_f、s_b；w_fwd=σ(s_b−s_f)；p_t=w·p_fwd+(1−w)·p_bwd；没有未来时 w=1 | s_b 偏置为 2（w≈0.88，起点接近 A50k） | 2×68K |
| **Revision** | 用原 timeline.update 修正过去节点，memory 为过去节点加融合后的 p_t | 共享 | 0 |
| **Position head** | p_t 平均池化 → τ∈[0,1]（sigmoid），以及 K=8 个余弦原型的 phase logits | τ 零初始化 | 0.30M |
| **动作头 DiT** | 只读 v_proj(p_t)（16×32）和 NodeCognition 读出的 z_node(p_t)；`--raw-z-bypass` 可恢复 raw z | A50k | 原有 |
| **Future Imaginer**（Stage II） | OFormer 风格：可学习 query 加未来时间戳，对过去和当前节点做 cross-attn，生成想象的未来节点 | 拷贝 A50k timeline 权重，query 零初始化 | 3.26M |
| **执行动作 GRU**（Stage II） | 沿用 ExecutedAdapter，与 timeline_force student 相同 | 小随机初始化 | 0.1M |

新增参数（相对 A50k）：Stage I 共 21.5M（其中 17.8M 是 NodeCognition），Stage II 共 25.0M，都远小于 100M。8B 骨干、QFormer、DiT 没有改动，总可训练参数 8.084B。

## 2. Loss
**Stage I（teacher，4 帧，现在位置 c∈{0,1,2} 都算，未来=真实的后续帧）**
- FM：3 个现在位置的 p_t，加上 L_gen（c=2 时修正后的过去节点 0、1，以及 hindsight 节点 0、1，都做 node→action）
- L_rev：修正后的过去节点和 sg(hindsight 节点 versions[3]) 在条件空间做 cosine
- L_prec：每个分支对 sg(hindsight 节点 c) 的异方差高斯 NLL。残差 detach，只训方差头，所以权重会跟着误差走，不会塌成均值
- L_agree：fwd 与 bwd 分歧在合并方差下的 NLL。同时记录 disagreement
- L_mono：同一 episode 内 τ_0<τ_1<τ_2 的 margin ranking（不需要标注）
- L_phase：SwAV 式交换预测，fwd 视角和 bwd 视角互相预测对方的 code；code 用 Sinkhorn 在全局 batch（all_gather）上做等分配，实现跨 episode 对齐。另加期望 phase 序号不递减的约束
- L_tau（可选，弱标签 t/(T−1)）：`--tau-label-weight`，默认 0，即完全不用标注
- 每 4 步隐藏 anchor1（`--hidden-every`）；`--future-drop 0.25` 随机去掉全部未来，让 teacher 也能只用 fwd

**Stage II（student，3 帧加执行动作，没有未来帧）**
- 想象未来时，只用 teacher 未来 anchor 的**时间戳**（不用像素）；部署和 dev 时用固定间隔 gap×{1,2,3}（`--imagine-gap 4`）
- FM（3 个现在位置），以及向冻结 teacher 蒸馏：
  - 蒸馏 p_t：条件 cos 加 z cos，权重 0.1
  - 蒸馏 τ：MSE，权重 0.1
  - 蒸馏 phase：KL，权重 0.05
- 想象未来与 teacher 真实未来节点对齐：条件空间和节点空间各一个 cos，权重 0.1
- L_rev：对齐 teacher 的 hindsight 节点
- L_prec 以 teacher 的 p_t 为目标，另有 L_agree、L_mono、L_phase

默认权重：gen 1、rev 0.1、prec 0.1、agree 0.1、mono 0.1、phase 0.05、phase-order 0.05、distill 0.1、imagine 0.1（都可以用 `--*-weight` 改）。

## 3. 消融开关
`--no-backward`（去掉 U_bwd）、`--fusion {precision,mean,forward}`、`--no-revise`、`--raw-z-bypass`、`--phases K`、`--tau-label-weight`、`--future-drop`、`--hidden-every 0`，以及上面各项 loss 权重。脚本里通过 `EXTRA_ARGS="..."` 传入。

## 4. 训练与日志
- LR：已有参数 1e-6，新参数 1e-5。Stage I 的新参数是 node_cognition 和 present.*；Stage II 的新参数是 imaginer 和 GRU。warmup 1000 步后 cosine 衰减。
- 拓扑任意：accum = global_batch / (节点数 × 每节点卡数 × micro)。64 卡、batch 320 时 accum=5。
- 多节点用 hybrid_zero2，单节点自动切到 full。
- resume 要求 world 和 recipe 都相同，preflight 不写死卡数。
- train.jsonl 每步记录：所有 loss、w_fwd 的均值和标准差、fwd/bwd disagreement、logvar、τ0/1/2、τ 顺序正确率、|τ−t/T|、phase 熵，以及 step 1 和 first-check 时各模块的梯度范数审计。
- Dev（每 5000 步）：
  - mse6：teacher 测 causal_last、causal_middle、hindsight_middle；student 测 causal_last、causal_middle（用想象的未来）
  - dev τ 顺序正确率
  - probe：48 条 episode×8 帧，测 progress R²、episode 内 Spearman、aliasing 帧对正确率（视觉 cos 前 20% 且进度差≥0.3）、phase 直方图、w_fwd
  - 注意：probe 用的是训练 index 里的 episode，因为 dev 没有 episode 长度，所以不是 held-out。

## 5. 64 卡提交命令（与 run_whenami_timeline_force_320/160.sh 同格式；平台注入 WORLD_SIZE=节点数、RANK=节点号、MASTER_ADDR/PORT，每个节点各执行一次）
```bash
# Stage I（先跑 admission：100 步，再续训到 120 步验证；通过后从 step-120 接着跑 main）
STAGE=teacher VARIANT=present PHASE=full RUN_ID=present_teacher_present_64_50k_01 NPROC_PER_NODE=16 bash /home/zhengguantian/WoG_fresh/run_whenami_present.sh
# 不跑 admission、直接开 main：PHASE=run
# Stage II
STAGE=student VARIANT=present PHASE=full RUN_ID=present_student_present_64_50k_01 NPROC_PER_NODE=16 \
TEACHER_CKPT=/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_present/training/present_teacher_present_64_50k_01/main/checkpoints/step-050000 \
bash /home/zhengguantian/WoG_fresh/run_whenami_present.sh
# 断点续训（新 RUN_ID，拓扑必须相同）：
STAGE=teacher VARIANT=present PHASE=resume RUN_ID=present_teacher_present_64_resume_01 NPROC_PER_NODE=16 \
RESUME_FROM=<旧run>/main/checkpoints/step-XXXXXX bash /home/zhengguantian/WoG_fresh/run_whenami_present.sh
# 消融：VARIANT=no_backward|mean_fusion|forward_fusion|no_revise|raw_z；只打印命令不执行：DRY_RUN=1
```
输出目录：$STORE/outputs/whenami_present/training/$RUN_ID/{admission1,admission2,main}；日志：同目录下的 main.node$RANK.log 等。

## 6. Smoke 结果
见第 7 节（单张 PPU，engineering 模式，bf16 存储）。

## 7. Smoke 结果（neolix_al_2，单张 PPU，engineering 模式，bf16 存储；时间为 UTC）
- **单元测试**：`python -m whenami_present.test_present`，5/5 通过。覆盖以下内容：
  - shape
  - fwd 分支看不到未来（改动未来节点后 p_fwd 逐位不变，p_t 会变）
  - 没有未来时 p_t 严格等于 p_fwd
  - mean、forward、no-backward、raw-z 各开关
  - imaginer 的 shape 和梯度
  - 精度头在 NLL 下离开均匀权重（准确的分支 w>0.8）
  - 参数预算
- **Stage I**（`local/smoke_03/teacher1`，21 步）：
  - 没有 NaN，峰值显存 77GiB，约 2.2 s/步
  - 第 1 步梯度审计：vision、projector、llm、qformer、U_fwd、node_cognition、dit、embed、U_bwd、precision、position 全部 >0
  - 初始 dev 正常：causal_last 0.0687，causal_middle 0.0378，hindsight_middle 0.0333（w_fwd=0.879）
  - probe 正常运行
- **Stage I resume**（teacher2 从 step-20 续训）：第 21 步的全部 loss 与不中断的那次逐位相同，RESUME_VERIFIED 通过
- **Stage II**（student1，21 步，teacher 是 smoke 里的 Stage I ckpt）：
  - 冻结 teacher 加 student 同在一张卡上，峰值 92.3GiB
  - 13 个梯度组全部 >0（含 future_imaginer、executed_gru）
  - dev causal_last 0.0583
- **Stage II resume**（student2）：第 21 步逐位相同
- **固定 batch 过拟合检查**（`local/overfit_01`，40 步，新模块 lr 1e-3）：
  - teacher：
    - total 1.339→0.048，fm 0.384→0.030，L_gen 0.598→0.079，L_rev 0.268→0.033
    - prec_nll 1.16→−0.52，L_phase 2.10→0.97
    - τ 顺序正确率 0.4→1.0
    - w_fwd 0.88→0.49：精度权重确实在学，没有停在初始值，也没有塌成常数
  - student：
    - total 0.461→0.325，fm 0.083→0.015，imagine_cond 0.292→0.040，imagine_node 0.185→0.090，L_rev 0.106→0.037
    - 继承来的 precision/position 头用的是 lr 1e-5，在 bf16 存储下更新被舍入掉，所以基本没动；正式训练用 FP32，不受影响
- **限制**：
  - 单卡 FP32 Adam 放不下 8B，所以 smoke 只能用 bf16 存储
  - 64 卡 hybrid_zero2 只在启动脚本里 dry-run 过，正式提交后需要看前 100 步（first-check 会做梯度审计、保存和评测）
