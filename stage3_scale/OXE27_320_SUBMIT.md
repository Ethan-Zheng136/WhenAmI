# OXE27 主训练：320卡配方与当前提交顺序

2026-10-02 13:21 UTC更新。**A全量准备和B的32卡验收均已完成，正式入口的真实
preflight门控已全部通过，可以提交320卡（20节点×16）的Stage3主训练。无需重跑A/B。
目前只做了提交前检查，尚未启动320卡任务；不保证长跑性能或最终科学效果。**

A包含27域、13,935/13,935源分片，最终索引有1,034,808条去重训练轨迹、
42,575,060个有效训练中心。已核查全部分片文件存在且大小匹配、全部轨迹的train身份、
域内窗口计数及索引身份唯一性，并从每域的正式产物各抽一条轨迹读取图像/动作（27条）。
没有重新对约1.92TB产物做全量SHA扫描，也不是逐帧穷举解码。
源码、统计和初始化端点与验收版本一致，主输出目录尚不存在。
检查记录：`outputs/stage3_joint/oxe27_full_readiness_20261002.json`、
`outputs/stage3_joint/oxe27_main_real_preflight_20261002.log`。

已核查三组真实8B训练1—4步、2/4步checkpoint、恢复后梯度和开发动作评测。
这证明工程链路可运行，不是4步小样本就证明科学效果。
按用户清理要求，已删除六份2/4步验收checkpoint张量，释放约540.25 GiB；
报告、日志、recipe、开发预测及checkpoint元数据保留，主训练不从验收权重初始化。
清理清单和归档位于`outputs/stage3_joint/cleanup_admission_20261002_01/`，
原验收目录保留`ADMISSION_COMPLETE.json`，各`checkpoints/PRUNED.json`注明不可恢复。
没有修改任何已验收的Python源码，没有删除正式端点或原始数据。

## 0. 已提交旧任务的真实状态

- 旧两域`raw_full_01/train_index.json`已生成、complete=true，保留，不删除。
  它只是Bridge/Fractal，没有27域，不能用于本次OXE主训练。
- 旧32卡验收`admission_32_01`在tiny FSDP optimizer restore失败：
  `Missing key in checkpoint state_dict: optimizer.param_groups`，
  涉及rank8—31，真实8B训练未进入。
- 已改为`checkpoint_v2.py`：模型仍全员DCP，optimizer/RNG按真实local shard
  每rank独立保存，不做全局optimizer param_groups转换；相同world与参数布局恢复。
- 保留旧checkpoint和失败日志，新验收使用`oxe27_admission_32_v2_01`，
  不覆盖旧失败目录。CPU4-rank真FSDP空参数分片测试和单PPU测试通过；
  随后的32卡实际重测也已通过，三组真实模型保存/新进程恢复/评测完成。
- **后续失败已区分**：2026-10-02日志时间19:13:07的同一作业重试，
  已加载`checkpoint_v2.py`，但仍指向旧`admission_32_01`，
  在保存`tiny_fsdp/step-1`时报`FileExistsError`，不是再次发生optimizer恢复错误。
  `tiny-node0.log`为多次运行追加日志，不能把前面旧异常误当本次根因。
  NCCL abort/SIGTERM为退出清理。启动器现已在torchrun前检查旧产物并明确拒绝；
  不删除、不覆盖旧ckpt；重提须使用下面的新目录。

## 1. A/B均已完成（以下保留为运行记录，直接跳到第2节提交主训练）

### A：已完成27域正式数据准备（本次实际为1节点、16进程）

```bash
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_oxe27_prepare.sh
```

这是CPU/I/O工作，不做GPU模型计算。支持1节点或多节点，平台提供节点级
WORLD_SIZE/RANK/MASTER_ADDR；建议先2节点，32个进程。
若平台只能申请GPU节点，也无需为这个任务申请320卡。

数据源：
- 原项目`oxe_magic_soup_plus` **27域**；
- 13,935个元数据指定TFRecord文件，约5.39TB文件字节；
- 1,331,097个source train episodes（最终只输出train-role有效样本，
  不把source数量当作最终训练数量）。

使用TFDS metadata明确选择版本/shard，`dobbe`目录的另一套1016-shard布局
不会混入当前256-shard数据。域采样概率为原mixture multiplier乘原checkpoint
的`num_transitions`后归一化，不是27域平均。

输出：
`outputs/stage3_joint/oxe27_raw_full_01/oxe27_train_index.json`
要求`complete=true, data_mix=oxe_magic_soup_plus, domain_count=27`。
只有这个输出可供320卡主任务，不会误接受旧两域/27域smoke索引。

### B：已完成的联合训练验收（2节点×16卡=32卡，不需重提）

以下命令仅保留为运行记录；该目录已有完成产物，不能再次原样提交。

```bash
ADMISSION_ROOT=/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/oxe27_admission_32_v2_01 \
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_oxe27_admission.sh
```

新目录`outputs/stage3_joint/oxe27_admission_32_v2_01/`。
两节点使用相同命令；不加`RESUME=1`。若此新目录之后也已有产物，
先检查失败阶段，再另选新目录；不要反复自动重启整个验收入口。
自动tiny多设备FSDP恢复测试，然后latent/forecast/distill_future三组真实8B模型
分别train2步→保存→新进程resume到4步→开发动作采样。
使用27域真实原图smoke（54条train轨迹、13,374可抽样中心），不是再筛选效果。
micro-batch2、FM draws4，与主训练每卡张量形状一致。
不占用全320卡做最初恢复兼容性检查。

成功标志：
`oxe27_admission_32_v2_01/ADMISSION_COMPLETE.json`

## 2. 现在可以提交320卡主训练

申请 **20节点，每节点16卡**，命令：

```bash
RESUME=0 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_oxe27_train_320.sh
```

默认forecast；旧VLM视觉/语言、Past QFormer、R/K、完整DiT都参与训练，
约8.07B学生参数。冻结Future编码目标；并非再做6.48M缓存学生实验。

固定配方：
- 50,000 optimizer steps；
- 每卡micro-batch2，累积2 → global batch **1280**；
- 每观察4次FM随机时间/噪声抽样，独立观察仍1280/更新；
- 原学生参数LR2e-5，R/K LR1e-4，warmup1000后cosine到峰值10%；
- 保守沿用LR，不仅因卡数256→320就武断线性放大学习率；
- 每2500步分布式保存、开发动作评测，封存测试不打开；
- 50k×1280=64M观察抽样，不与WoG50k×1024宣称等预算。

同预算无Future监督对照，另行提交、不同输出目录：

```bash
VARIANT=latent RUN_ID=oxe27_latent_320_50k_01 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_oxe27_train_320.sh
```

可选完整冻结Future oracle蒸馏：

```bash
VARIANT=distill_future RUN_ID=oxe27_distill_future_320_50k_01 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_oxe27_train_320.sh
```

三者是不同任务，不会在同一个320卡作业中自动全部训练。
默认先推进forecast与latent，检验Future辅助监督对大模型的增量；
distill需额外冻结7B教师前向、算力/通信更大，不能误称等算力。
ReaderStudy oracle来自两域，其跨27域监督有效性未验证，不默认替换所有动作真值。

## 3. 27域接入不是简单修改名字

逐域真实样本审计发现Kuka大量轨迹只有9—23帧，
旧两域“必须完整[-15,+15]窗口”会将其整域过滤掉，已明确改变边界规则：
- 不跨episode，历史/未来缺帧重复边界图像；
- 至少有一个真实后续观测；
- 不足16步动作位置有有效性mask，不作为真实动作监督；
- 每个观察的有效动作loss单独归一化，不让短轨迹被零填充动作主导；
- 保留原域动作标准化与夹爪语义，pad仅服务张量形状；
- 记录history/future重复窗口数量。

这改变了两域interior-only诊断的采样协议，不把二者数字直接横比。
重复的Future帧是边界处理，不是新的真实未来信息，模型能否利用短轨迹仍待训练。
DROID原随机exterior-camera选择用serialized episode hash固定随机seed，
从而保留原随机选相机类型但确保重跑/分区ID不漂移；不声称增强分布完全不变。

## 4. 已完成验证

- 27/27域，54个真实train episode样本：
  原standardization、primary camera、语言、7D动作、图像解码、
  action normalization、重复解码身份一致性全部通过。
- 27域raw fixture逐域加载与weighted minibatch确定性重放通过；
- Kuka完整首shard（580源episode）准备得到6682 train windows；
  DROID完整首shard（40源episode）得到7408 train windows；
  不是只生成了假的smoke元数据。
- 27域weighted sampler实际抽到Kuka样本，真实8B学生反向通过：
  VLM视觉/projector/LLM、Past、R、K、DiT都获得非零有限梯度；
  单PPU约32.8GiB峰值，BF16 master且没有Adam，不能代表多机完整训练验证。
- 76项单元/回归测试通过；包含4-rank CPU真FSDP含空local shard恢复，
  下一更新参数逐值一致，以及单PPUtiny FSDP恢复。
- launcher语法/dry-run通过；主入口检查27域、真实审计、v2恢复schema、
  microbatch/FM形状、源码与基础依赖hash。

当前未完成：320卡长跑。13,935个source完整转换和32卡验收均已完成，
主训练入口的真实提交前门控已通过。
主训练入口没有绕过门控的开关。以前的完整端点权重/开发/测试分区不修改。
