# WhenAmI：Future / Past 分阶段路线

更新：2026-10-02。此文档补充而不覆盖 DESIGN.md / DESIGN_v2.md。
`whenami_v3`、`flow_v2`、`flow_v3` 仍是历史实验名，不对应下表的阶段。

## Stage 3 试验实现（2026-10-02）

**最新：27域完整数据入口已实现，主训练选择320卡，不再沿用两域256卡配方。**
见`stage3_scale/OXE27_320_SUBMIT.md`。27域真实兼容审计与76项测试通过，
Kuka边界重复/有效动作mask避免整域被短轨迹过滤；加权采样延续原OXE mixture。
旧两域准备已完成；旧32卡验收因optimizer.param_groups恢复失败。
checkpoint-v2经4rank CPU真FSDP与单PPU验证，新的32卡集群验收仍待提交。
全27域准备也尚未执行；320卡主训练入口检查两者完成标记，不跳过验收。

**最新范围修正：原项目是27域OXE混合，不是只训练Bridge/Fractal。**
Stage1/2配置均为`oxe_magic_soup_plus`；当前联合训练数据入口仍只有两域，
不能把下述“全数据”理解为全OXE。27域源数据/统计清点通过，但扩展入口尚未实现。
见`stage3_scale/OXE_SCOPE_CORRECTION.md`；先不要把两域任务A作为OXE准备提交。

**最新：全数据联合策略版已实现工程入口，待32卡验收。**
详见`stage3_scale/JOINT_DESIGN.md`与`stage3_scale/SUBMIT_JOINT.md`。
现在训练原图输入的VLM视觉/语言、Past QFormer、R/K与整个DiT，共约80.7亿参数；
冻结Future目标/可选oracle教师。本机真实全反向和原图推理通过，69项测试通过。
全语料CPU打包与两节点32卡train/save/resume验收可以并行提交；
256卡50k主训练必须先通过验收，不声称已验证完整多机长跑或科学效果。
这是先前端点冻结方案的独立联合适应实验，不改变已有实验/封存测试集。

**最新：20k因果学生整套训练已准备好，待用户提交。**
命令 `run_stage3_causal_student_20k.sh`，1节点16卡，四组目标×四训练种子；
学生统一从adaptive_mean5k加载K/读出适配器，再新增20k，预测器R新建。
真实Future只进冻结teacher/监督目标，学生在线不读取；封存测试不打开。
59项测试及真实四组smoke通过，本机10步中断续到20步与连续训练逐值一致。
详见 `stage3/CAUSAL_STUDENT_20K_SUBMIT.md`。尚无正式20k学生结果。

**最新：Reader Study四格×四种子集群任务已全部完成并审计。**
新train16384/dev2048/封存test2048，16组各5000步，原权重不变。
真实Future相对同列均值条件：固定读出MSE降低Fractal3.55%/Bridge2.28%，
可适应读出降低3.73%/3.68%，每种对照均4/4训练种子有利。
详情见 `stage3/READER_STUDY_RESULTS.md`。读出适应的额外作用尚非唯一解释；
这仍是oracle证据，不是预测Future/双向学生成功。封存测试未编码或评测。
`runs_01`已结束，原提交指令不要重复执行到同一路径。

**最新诊断（05:41 UTC）：pipeline图与Future信息接口对照均完成。**
图见 `stage3/PIPELINES.md`。单独训练真实Future C的oracle K仍未胜普通适配器；
投影前H、C信息还原H、重复当前帧H与均值的同结构对照，也没有发现
跨两数据集一致的额外Future价值。见 `stage3/ORACLE_EVIDENCE_RESULTS.md`。
40项测试通过；正式R+K结构未改，未解冻端点，未继续堆叠推理轮数。
这不能证明Future无用，冻结读出/单中心目标/数据与训练规模等因素仍待区分。

**最新（05:05 UTC）：R+K已实现并完成五组本地1000步对照。**
保留Past，R预测Future，经K产生Past条件残差，再由一个冻结Past DiT读动作；
纯在线action梯度已经实测能到R/K。普通K适配器主误差相对原Past降低
Fractal5.99%/Bridge10.21%，但等参数控制尚未支持Future与oracle回溯增益。
4区间可微动作采样辅助本轮未改善主指标，步时约为teacher FM的2.61倍。
详见 `stage3/REFINEMENT_01_RESULTS.md`；36项测试通过，无本轮遗留任务。
这仍非多时间锚点时间线，Stage4在线管线未实现。

**后续改进对照也已完成（04:27 UTC）：**
诊断显示抽样共享正/逆梯度近乎正交；将alignment权重1降到.1，
重跑单/双向各1000步，Fractal稍好而Bridge退步，仍无实质逆向收益，
因此没有替换默认训练配置。原WoG Stage II / WoG-A已补评于同一211窗口：
Fractal MSE=.039541、Bridge=.038758；它不是“真实Future”那条Stage I参照。
详见 `stage3/IMPROVEMENT_01_RESULTS.md`。
`stage3/NEXT_DESIGN.md` 是保留Past、让未来回溯校正参与在线动作的下一版候选；
只写了设计，未实现或验证。21项测试通过；本轮本机训练/评测均已结束。

**最新：用户授权后，已在本地单PPU完成正式固定子集pilot。**
单向/双向各1000步，813训练窗口/211验证窗口；最终checkpoint动作评测均完成。
双向相对仅正向的归一化MSE：Fractal降低0.738%，Bridge升高0.142%，
增益很小且非跨数据集一致；不能据此宣称双向机制已经有效。
原Past在同一验证集上的Fractal表现仍好于预测Future条件路径。
结果与限制详见 `stage3/PILOT_01_RESULTS.md`，
完整机器可读比较位于 `outputs/stage3_pilot/comparison_pilot_01.json`。

新增独立 `stage3/`，设计与入口详见 `stage3/DESIGN.md`。
冻结两个端点QFormer/VLM/DiT，用来源侧宽特征+当前任务特征预测另一侧条件，
通过固定目标对齐与固定目的侧DiT的action flow-matching loss训练共享连接模块。
R不读取真实动作或反方向目标特征；单向/双向对照已实现。
**这是同中心的双向条件预测，不宣称已实现多节点状态时间线或Stage 4在线融合。**

14项单元/CPU分布式测试、真实特征提取、两个模式各30步单PPU训练、
小checkpoint保存重载与完整动作读出检查通过。
小样本测试仅12 train / 4 val窗口，不能据此判断双向优于单向；
这是正式pilot之前的smoke记录；正式pilot及后续对照状态以上文为准。

## 最新训练状态（2026-10-02）

Past Stage 2 已完成累计20k更新：原生产路径5k模型权重，
接 `WoG_16rank_probe/` 的20节点×16进程、每卡batch=8，继续15k。
后半段优化器重新初始化、shuffle从64000改为4096，不是无缝resume。

最终权重：
`/mnt/pfs/zhengguantian/WoG_v2/outputs/stage2_16rank_probe/past_s2_16rank_20node_from5k_01--image_aug/checkpoints/step-015000-epoch-01-loss=0.2112.pt`

2026-10-02 09:12（北京时间）完成最终checkpoint保存；20个节点退出码均为0，
320个rank均到达最后一批，OOM计数无增量。继续15k的任务耗时约11.9小时，
最后100步平均action loss=0.131803，中位步时2.7716秒。
节点cgroup峰值1021–1059GiB，低于1792GiB上限。

最终checkpoint已mmap读取，Stage-2元数据、所有模型组键名/形状与起点一致；
Past QFormer与DiT全量张量有限，抽查关键参数相对5k均更新。
**完成训练不代表性能已验证，Past因果动作评测尚待完成。**
下文“本次状态”和“本次验证”描述保留为2026-10-01实现时记录，
后续接手以本节为当前状态。

### Past评测入口（2026-10-02新增）

`run_eval_past.sh`，1节点16卡，默认每数据集256个完整窗口。
从原始RLDS统一取样，比较Past真实历史、Past静态历史输入干预、
WoG Future oracle；不是旧的`run_eval_cond.sh`协议。
细节、数据泄漏边界、实际默认checkpoint与输出位置见 `evaluation/README.md`。
已完成小样本完整模型接线测试，正式评测尚待提交。

正式首轮评测现已完成（2026-10-02），详见 `evaluation/README.md`：
Past的r/R²为Fractal 0.625/0.303、Bridge 0.549/0.145；
相对静态历史的完整动作归一化MSE降低约5.2%/5.6%。
Future oracle为0.692/0.410、0.563/0.169，Past未超过它。
支持Past基础路径继续研究，不代表Stage 3双向时间线已经成立。

## 统一命名

| 阶段 | 作用 | 本次状态 |
|---|---|---|
| Stage 1 — Future | 真实未来条件 + action loss；现有 WoG baseline | 已训练 20k，原权重保留 |
| Stage 2 — Past | 可获得的历史条件直接帮助生成当前起始的动作 | 本次实现 |
| Stage 3 — 双向连接 | 固定已学好的两个编码器，训练正逆预测/读出 | 设计中，未实现 |
| Stage 4 — 在线整合 | 让连接机制在无真实未来时帮助行动，必要时小学习率解冻 | 设计中，未实现 |

这不是旧 WoG 的两阶段编号！**Past Stage 2 使用 `train_venc=True`**，
因为 QFormer 和 DiT per-attention 都需要训练并在部署时保留。
旧代码的 `train_venc=False` 是 WoG 的 oformer 蒸馏路径，不用于本次 Past。

## Stage 2：先做最小历史视觉 baseline

```text
历史图像 [t-15,t-11,t-7,t-3] ── DINOv2 ── 4×64 patches ──┐
                                                        │
历史 + 当前 [t-15,t-11,t-7,t-3,t] ── Wan VAE ── tokens ──┤
                                                        ▼
                                        原样 WoG QFormer（6层/1024）
                                                        │
                                              C_H [B,16,32]
                                                        │
当前图像 + 指令 ── VLM ── z_t ──────────────────────────┤
                                                        ▼
                                                       DiT
                                                        │
                                              a_t ... a_{t+15}
                                                        │
                                                   action loss
```

- 不加 T、Λ、三路融合或 future distillation，不宣称已形成完整时间线。
- 本版不把历史动作喂给 QFormer（与 Future 视觉条件 baseline 一致）。
  历史动作可供后续 Stage 3 的转移使用；不能暗称本版已建成 action-conditioned state。
- QFormer 完全复用，不新增时间/action embedding；VAE 的输入顺序保留历史时序。
- VLM 只看当前图像+语言；历史通过保留到部署的 per-attention 条件进入动作头。
- 视觉编码器冻结并保持 eval；Past QFormer、DiT、VLM 默认按原 full-finetune 配置训练。
- 未来动作只出现在 loss 标签中；不进入条件编码。输入未来帧直接报错。
- 当前目标的原始列索引是 `-min(obs_offsets)`，不是历史采样帧数。
  本例动作数组覆盖 `t-15...t+15`（31行），标签准确取第15至30行。
- 使用现有 RLDS action_mask 屏蔽无效标签；其边界/中性动作定义沿用原数据管线。

## 初始化与公平比较

默认 `INIT_MODE=stage1`：从 `baseline_v2` 20k 迁移完整 VLM/DiT，
严格将 `qformer` 权重载入独立的 `past_qformer`。Future 源文件不被更改。
这是一条“Future 预训练后适配历史”的实验，**不是与 Future 同训练预算的从零对照**。

`INIT_MODE=openvla`：从相同 OpenVLA 起点初始化 VLM，Past QFormer/DiT 从头训练；
对应同起点的历史 baseline。所有初始化信息均写入 run config。
不要仅训练 Past QFormer 却永久冻结不适配新条件的 DiT。

## Stage 3 是否冻结前两者？

**建议 Stage 3 的连接学习初期冻结 Future/Past 编码器（参数和 train/eval 模式）。**
固定目标表示，避免两个头相互追逐或共同塌缩，先验证正逆推理是否成立。
但冻结不等于 `no_grad` 包住全部后续计算：
若固定动作读出要给新连接模块提供行为梯度，读出参数可冻结，
从动作 loss 到连接模块的计算图必须保留。

Stage 3 必须定义部署时新模块如何被消费，否则训练一个独立的逆向预测器、
部署时删除它，不会自动改善固定的 Past 策略。
Stage 4 可继续固定 teacher、训练在线连接/读出，或逐步解冻 Past/DiT；
这不是强制所有模块都解冻。是否共享 R、目标空间/概率建模仍待讨论，
本次不预先实现。

## 文件与兼容性

- `WoG/vla/past.py`：历史输入、时间窗口、checkpoint 合约、在线采帧辅助函数。
- `WoG/vla/vla.py`：独立 Past 模块、Stage-1 权重迁移、动作监督及推理。
- `WoG/scripts/train.py`：`whenami_stage=2` 入口；禁止复用已有 run 目录。
- `run_stage2_past.sh`：新入口；输出到 `outputs/stage2/`，默认250步 smoke。
- `WoG/scripts/check_past_stage2.py`：只读权重/环境预检。
- `WoG/tests/test_past_stage2.py`：语义、梯度、兼容性测试。

旧参数默认 stage=0，保持原 Future/递推/masked/flow 路径；
旧 checkpoint 不迁移、不重写。新 checkpoint 中保存独立的 `past_qformer`
以及 `whenami` 元数据（阶段、offsets、horizon）。加载按 checkpoint 元数据，
不依赖可能被覆盖的 config.json 中的采样配置。仍需 config.json 的 VLM 类型信息。

QFormer 增加无参数的 `encode_queries()`，可读出末端降维前 `[B,16,1024]`；
它不改变输出/权重键，也不意味着这些 query 是逐时刻状态。

## 启动

先查看命令而不训练：

```bash
DRY_RUN=1 bash /home/zhengguantian/WoG_fresh/run_stage2_past.sh
```

平台沿用原镜像，每节点8进程。先250步：

```bash
NPROC_PER_NODE=8 PER_DEVICE_BS=8 LR=3.2e-5 \
MAX_STEPS=250 SAVE_INTERVAL=250 RUN_ID=past_s2_stage1_smoke_01 \
bash /home/zhengguantian/WoG_fresh/run_stage2_past.sh
```

通过集群 smoke 后再另开 run（不覆盖 smoke）：

```bash
NPROC_PER_NODE=8 PER_DEVICE_BS=8 LR=3.2e-5 \
MAX_STEPS=20000 SAVE_INTERVAL=5000 RUN_ID=past_s2_stage1_20k_01 \
bash /home/zhengguantian/WoG_fresh/run_stage2_past.sh
```

默认实际 repeated_diffusion_steps=4，与旧 VLA 训练循环的实际默认一致。
旧脚本虽然写8，但 run_vla_training 原先没有透传，实际 forward 使用4；
新 Stage 2 显式透传，可用 REPEATED_DIFFUSION_STEPS 覆盖，不悄悄改变旧实验。

现有 FSDP saver 只存 model，不存 optimizer；故本入口禁止“精确 resume”的说法。
可用 CKPT=<Past checkpoint> 和**新 RUN_ID**显式 warm start（优化器/步数重置）。
不要用旧 `run_eval_cond.sh` 评 Past：它要求真实未来并使用旧缓存采样。

## 在线接口

```python
from vla import load_vla
from vla.past import history_frames

policy = load_vla(past_checkpoint, action_model_type="DiT-L")
# Stage-2 身份、train_venc=True、offsets 自动从 checkpoint 恢复。
policy = policy.float().cuda().eval()
episode_images = []  # 每个 episode reset；每个环境步追加一帧
episode_images.append(current_image)
past = history_frames(episode_images, policy.obs_offsets)
action, normalized = policy.predict_action(
    image=current_image, instruction=instruction,
    past_frames=past, unnorm_key=dataset_name,
)
```

采样跨度以**原始环境步**计，不是 policy 调用次数；如果一次执行多个动作，
要按环境步收集图像。episode 开头缺历史时重复首帧，与 RLDS 裁边规则一致。

## 本次验证记录（2026-10-01）

- `tests/test_past_stage2.py`：**26 passed**，含真实小型 QFormer/DiT 的梯度、
  Future→Past 权重迁移、训练/推理一致性、TF chunk→collator→模型、
  OpenVLA 随机头起点，以及单 PPU 的 FSDP 保存/身份恢复。
- `check_past_stage2.py`：对现有 baseline 20k 的168个 QFormer张量逐项核对键名/形状通过。
  节点预检会先清除自身进程中的分布式环境变量，避免在 torchrun 之前误建通信组。
- `smoke_stage2_past.py --backward`：**完整真实模型 + 一条真实 Fractal RLDS样本通过**。
  观测时间为 `[0,4,8,12,15]`，只包含历史/当前；推理动作 `(16,7)` 且有限；
  Past 输出投影与 DiT per-attention 均有非零、有限梯度。
  为控制单卡显存，该梯度测试冻结 VLM，不更新任何权重。
- 全模型诊断结果与日志：
  `/mnt/pfs/zhengguantian/WoG_v2/outputs/stage2_checks/full_smoke_20261001.{json,log}`。
- **未验证**：多节点 full-VLM 训练、250步保存、20k效果、Stage 3/4。
  单样本 loss 仅为接线诊断，不是任何性能结论。
- 旧代码改动前的小文件备份：
  `/home/zhengguantian/WoG_fresh/change_snapshots/past_stage2_20261001T062802Z/`。
  不含数据/模型副本；未改写旧 checkpoint 或已完成训练日志。
