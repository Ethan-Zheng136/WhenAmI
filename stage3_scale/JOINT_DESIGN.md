# 全数据联合策略版：实现设计与实验边界

**最新（2026-10-02）：已扩展为27域`oxe_magic_soup_plus`，选择320卡、
global batch1280；窗口边界有效动作mask、原mixture采样及checkpoint-v2恢复
见`OXE27_320_SUBMIT.md`。旧32卡验收失败后，修正需在新目录重测。
下面两域/256卡段落是此前设计，不能作为最新训练配方。**

> **范围更正：本文“全数据”只指Bridge/Fractal训练分区。**
> 原项目使用27域`oxe_magic_soup_plus`；用户要求的通用OXE主训练还需要
> 扩展并验证数据入口，不能用下面两域配方代替。见`OXE_SCOPE_CORRECTION.md`。

2026-10-02。**训练与验收入口已实现；真正多节点8B训练尚待集群验收。**
这不是之前的6.48M缓存学生；也不宣称小测试已经证明双向方法有效。

## 1. 实际结构

```text
原始历史图像 ----> 冻结DINO/Wan ----> 可训练Past QFormer ----> H_P,C_P
                                                                  |
原始当前图像+语言 ----> 可训练VLM(vision/projector/LLM) ----> z_P ----+
                                                                  |
                                                        可训练R -> C_hat_F
                                                                  |
                                                 可训练K(H_P,z_P,C_hat_F)
                                                                  |
                                                              delta_C_P
                                                                  |
                                                        C_P + delta_C_P
                                                                  |
                                            全部可训练Past DiT(含既有adapter)
                                                                  |
                                                          action flow loss

训练辅助：
当前+真实未来 -> 冻结DINO/Wan/Future QFormer -> C_F -> 对齐C_hat_F

可选distill_future：
冻结的完整oracle VLM/Past QFormer/K/DiT
  使用真实C_F和自己的固定z -> teacher velocity -> 停止梯度的行为监督
```

- 学生 `JointPolicy.forward` 没有真实Future图像/条件输入。
- action/noisy action只进入动作DiT，R/K不读标签。
- **梯度不detach**：action及Future辅助loss可改变VLM、Past QFormer、R/K和动作读出。
- DINO/Wan时序基础视觉特征仍冻结；VLM自身的DINO/SigLIP视觉骨干为独立可训练模块。
- `latent`、`forecast`、`distill_future`三组同一学生结构；
  latent没有逐样本Future监督，作为同预算普通联合微调对照。
- 默认forecast；distill_future另作为可选教师监督对照，不与forecast混称。
- 第一轮只是同一中心时刻的前瞻/回溯校正；没有实现多时间锚点递推、
  物理可逆性或机器人在线执行包装，不能说完整WhenAmI研究目标已达成。

## 2. 初始化与可训练范围

- 原Past：累计20k的Stage2 checkpoint；
- K/DiT adapter：Reader Study同seed adaptive_mean最终5k；
- R新建，其他核心Past/VLM/DiT延用已训练权重；
- Future QFormer：原Stage1 final20k，冻结；
- 可选oracle：Reader Study adaptive_future同seed final5k + 固定Past大模型。
- 实测学生可训练 **8,065,609,895** 参数，约80.7亿，不是旧6.48M设置。

为了控制目标坐标，R/K条件均值/尺度固定为之前训练集统计，
不因新数据动态改变。这个选择需要在科学比较中明确。

## 3. 全数据，不是更大的固定特征缓存

全源扫描覆盖：
- Fractal 0.1.0：87212 source train episodes，1024文件；
- Bridge_orig 1.0.0：53192 source train episodes，1024文件。

每episode沿用原OXE动作标准化与Reader Study内容hash划分。
只输出train-role轨迹，保留过去已经固定的dev/test身份。
去掉原来“每数据集8192窗口、每episode最多8窗口”的样本上限；
使用所有具备完整[-15,+15]观测与16动作标签的合法中心位置。
跨文件重复episode在最终索引中只保留一个来源。

分片保存的是**压缩原始图像、动作与语言**，不是冻结的H/z/C。
每个训练batch重新解码并执行学生VLM/Past表征。
同一窗口所有图像共享随机裁剪、brightness/contrast/color增强，
不翻转，不改动作坐标。固定随机规则可以按step/rank重放。

两域各50%采样，域内按合法窗口数均匀；是随机有放回抽样，
不声称精确无重复epoch。所有抽样由(seed,step,microstep,rank)确定；
没有不可恢复的TF长shuffle缓存，也不让320个训练rank分别完整扫数据。

## 4. 真分布式训练与恢复

- FSDP FULL_SHARD，FP32 master参数，BF16计算，FP32梯度归约；
- 对LLM/视觉/BERT/DiT/R/K块非重入activation checkpointing；
- 真global batch + 梯度累积；每微步同步，不用no_sync堆积完整7B梯度；
- DCP全员并行写分片模型+optimizer，每rank独立RNG；
  完整写入后才生成COMMITTED并原子发布目录。
- 恢复必须相同world、recipe、data-index/hash及源码；
  无完成标记的中断目录不作为可恢复checkpoint。
- `stop-after2 -> 新进程resume到4`为集群验收固定流程。

## 5. 训练预算候选及开发评测

主训练默认候选：
- 50k更新，global batch1024，4个FM draws；
- 原学生参数LR2e-5，R/K LR1e-4；
- warmup1000，cosine到各自峰值的10%，clip1；
- 每2500步保存并进行开发集实际动作采样。

建议首个主训练拓扑：**16节点×16卡=256卡**，micro2、accum2 → global1024。
若必须用20节点320卡，global1024不能被320×micro2整除，
需要明确选择新global batch并重新确认训练预算，而非静默改变。

开发评测从现有2048个dev原始图像重新编码，只读历史+当前；
FSDP各rank调用次数相同，末尾不足补算后丢弃，避免集合通信挂起。
BF16模型计算、FP32Euler积分，与旧FP32缓存评测不混为完全相同协议。
最终封存test仍不打开。

## 6. 当前验证事实

- 两个真实TFRecord分片读取/标准化/分区/打包通过：
  62条train轨迹、968个完整窗口；不是全语料已转换完成。
- 单PPU完整真实学生反向通过：VLM视觉、projector、LLM、Past QFormer、
  R、K、完整DiT都有非零有限梯度；BF16 master、无optimizer，
  峰值32.8GiB。这个检查不是FP32-master Adam完整单卡训练。
- 实际冻结大oracle raw-input前向、学生raw-input动作采样通过；
  本地BF16，仅接线检查，不将2个样本的指标作为效果证据。
- CPU双进程真实DCP/DDP保存恢复后的下一次更新逐值一致；
- 单PPU tiny FSDP+DCP保存恢复通过，但world1会退化为NO_SHARD，
  **不能冒充多机FULL_SHARD验收**。
- 曾尝试两个rank共享单PPU，HGGC拒绝`Duplicate PPU detected`；
  已明确要求每个NCCL rank独占实际设备，未绕过检查。
- 69项单元/回归测试通过。

## 7. 仍需集群确认

原始语料完整打包可以与32卡验收并行提交。
32卡验收：tiny FSDP/DCP跨设备测试 → 三种真实8B目标各训练2步保存
→ 新进程恢复到4步 → 开发样本动作评测 → 发布验收标记。
只有同一代码通过验收且全语料索引complete，主训练入口才允许执行。

尚未确认：32卡8B FSDP峰值内存/collective行为、真实8B DCP优化器恢复、
256卡长跑吞吐/稳定性、科学效果。完整多时间WhenAmI仍非本次交付范围。
