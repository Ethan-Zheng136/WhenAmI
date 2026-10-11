# Timeline-A：训练就绪范围和完整pipeline对比

2026-10-04。本文件区分原始研究愿景、已经完成的Stage3和本轮真正实现的A，
不把单PPU smoke写成真实多节点通过。

## 1. 现在可以开始什么训练？

本轮新增集成已实测：8项入口/开发集检查通过；真实单PPU训练、保存、27域开发评测、
新进程恢复通过。连续训练与恢复训练的最终566个模型state条目、563组Adam状态逐项相同，
恢复点和最终点的27域评测结果也相同。此前模型的26项测试仍保留有效。
本地集成验收为每域1条、共27条开发episode；正式脚本使用完整410条，
410条的身份/形状与零训练重叠均已审计。

可以提交**2节点×16PPU的训练验收**：

1. 在10k总配方下训练到100步，保存model/Adam/RNG，评测27域开发集；
2. 退出后用新进程恢复到120步，再保存和评测；
3. 验收通过后，用同配置从120步继续到10000步。

这不是重新从0开始三遍，也不是从之前Stage3-50k继续训练。
三个研究组都从共同的Future Stage1-20k端点初始化QFormer/DiT，
冻结该端点的VLM和基础DINO/Wan。

还没有通过的是**真实32-PPU多节点通信/显存/保存恢复/吞吐验收**，
所以第一条命令必须停在120步，不能无条件自动开启长期大训练。

## 2. 数据与配方

### 已有训练数据直接复用

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/
  oxe27_raw_full_01/oxe27_train_index.json
```

流式重新选择真实episode与三个不同时间锚点，不重复训练之前27条smoke样本。
读取原图重新运行冻结视觉/VLM；不缓存可训练QFormer的输出。
真实动作只作训练标签，不作为当前A条件编码器输入。

### 新增独立开发集

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_a/dev27_16ep_01/
```

- 27域、410条独立episode、每条3个实际时间锚点；
- 从原来已登记的DEV episode重新取多锚点片段；
- 逐条核对原始TFRecord内容键，训练索引重叠为0；
- 基础端点预训练可能见过，不宣称基础预训练全程未见；
- 不是封存test；
- 只给冻结基础编码器缓存开发特征，训练中的可训练表示每次重新计算。

不需要下载OXE，不需要重新运行全量预处理任务。

### 首轮正式教师实验的固定配置

| 项目 | 配置 |
|---|---|
| 组别 | pooled / independent / timeline |
| 每组硬件 | 2节点×16PPU=32卡 |
| 单卡micro batch | 2条轨迹 |
| global batch | 64条轨迹 |
| 每轨迹锚点 | 3个真实时刻，变化间隔 |
| 每节点FM draws | 2 |
| 总更新 | 10000，包含100/120验收段 |
| 学习率 | QFormer/原DiT 1e-5；时间模块1e-4，恒定 |
| 保存 | 每1000步，另在100和120保存 |
| 开发评测 | 初始、每500步、段末与最终 |
| 评测内容 | 完整观测、隐藏中间节点；FM、实际Gaussian采样动作MSE/首动作/夹爪 |
| 输出 | 所有27域＋域等权汇总；tqdm、step、预计剩余时间 |

这是教师结构可行性实验，不是最终论文训练预算。
单训练seed42的结果不能替代多seed稳定性验证。
尚不能预报真实32卡速度；观察首次验收计时再决定是否扩大batch或卡数。

## 3. 提交命令

脚本：

```text
/home/zhengguantian/WoG_fresh/run_timeline_a_32.sh
```

集群资源申请2节点，每节点16PPU。平台提供WORLD_SIZE=2、RANK、MASTER_ADDR；
不要把平台的WORLD_SIZE手工设置成32。

### 第一个任务：timeline，多节点入场验收

```bash
TL_MODE=timeline PHASE=admit NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_a_32.sh
```

输出：

```text
outputs/whenami_timeline_a/training/timeline_a_timeline_32_10k_01/
  admission1/    # 1–100
  admission2/    # 新进程101–120
```

同样可分别提交另两组（每条是独立2节点任务）：

```bash
TL_MODE=pooled PHASE=admit NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_a_32.sh

TL_MODE=independent PHASE=admit NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_a_32.sh
```

三组同时跑共6节点96PPU；没有必要一开始320卡。
不是节省科学实验，而是避免将新分布式入口和训练规模两个变量混在一起。

### 对应组验收通过后，继续到10k

```bash
TL_MODE=timeline PHASE=full NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_a_32.sh
```

pooled/independent同理替换TL_MODE。脚本必须找到该组100/120步的有效checkpoint、
新进程恢复记录及27域评测，才允许执行full。它从120恢复，不重新初始化。
若已存在main目录，脚本拒绝覆盖；中断续跑需要根据实际checkpoint另行恢复。

## 4. WoG：先学习动作相关Future条件，再撤除未来

以下按本地固定WoG实现及已归档论文的方法部分。

### WoG Stage I

```text
真实未来图像 ── 冻结视觉 ── 可训练Future Encoder ── C_F ──┐
                                                            │
当前图像＋指令 ── VLM ── z ────────────────────────────────┤
                                                            ↓
                                                       动作DiT
                                                            ↓
                                                        动作loss
                         动作梯度训练Future条件编码器和策略
```

### WoG Stage II与部署

```text
训练：
真实未来 ── 冻结Future Encoder ── C_F target
                                         ↑ 对齐
当前＋指令 ── VLM ──┬── Future预测头 ─────┘
                     └── 动作DiT ── 动作loss

部署：
当前＋指令 ── VLM ── 动作DiT ── 动作
```

Stage II不是把预测Future再强制送回动作头。
它把“真实未来输入”转换为“表示学习监督”。
WoG的条件token不被定义为一一对应的事件时刻。

## 5. 我们最初的timeline：原始愿景，不等于已经实现成功

```text
观测：    o₋₂          o₋₁           o₀           o₊₁          o₊₂
           ↓            ↓            ↓             ↓            ↓
状态：    S₋₂ ──T──→   S₋₁ ──T──→   S₀ ──T──→    S₊₁ ──T──→   S₊₂
           ←─────────── 后向证据推断/消息 ────────────────────────
           │            │            │             │            │
           D            D            D             D            D
           ↓            ↓            ↓             ↓            ↓
        各时刻动作   各时刻动作    当前动作      各时刻动作   各时刻动作
```

其核心愿景：同一个空间里的多个任务状态，由动作/时间关系连接，每个节点可被动作读出。
旧版将其与可逆算子、精度融合、复杂递推等绑定，实际效果未成功。
不能把后向证据推断等同于物理动作的函数逆。
也不能把看过未来动作标签的状态预测同一动作，当作策略能力。

## 6. 已完成Stage3-50k：单中心条件修正，不是多节点timeline

```text
历史＋当前 ── 冻结DINO/Wan ── Past QFormer ── H_P、C_P ─────┐
                                                            │
当前＋指令 ── VLM ── z                                      │
                     │                                      │
               H_P、z ── R ── 预测C_F                       │
                     │                                      │
               H_P、z、预测C_F ── K ── ΔC_P ────────────────┤
                                                            ↓
                                                     C_P＋ΔC_P、z
                                                            ↓
                                                      动作DiT
                                                            ↓
                                                        当前动作

训练辅助：真实Future ── 冻结教师 ── R的对齐目标
```

联合训练VLM/Past/R/K/完整DiT。已经完成训练和评测。
没有显式时间节点、已执行动作条件转移或真实后见节点更新。
离线动作误差有改善，所报告SIMPLER闭环尚未整体超越WoG-A。
当前A不是从它的50k继续训练，二者是独立实验线。

## 7. 新Timeline-A：本次真正实现的版本

```text
三个真实锚点：      o₁                 o₂                 o₃
                    ↓                  ↓                  ↓
独立冻结编码：  DINO/Wan,VLM       DINO/Wan,VLM       DINO/Wan,VLM
                    ↓                  ↓                  ↓
共享局部编码：      Q                  Q                  Q
                    ↓                  ↓                  ↓
局部状态：         H₁                 H₂                 H₃
                    └──────────────────┼──────────────────┘
                                       ↓
                         明确可见性＋相对时间的关联模块
                                       ↓
节点读出：         S₁                 S₂                 S₃
                    ↓                  ↓                  ↓
条件投影：         C₁                 C₂                 C₃
                    ↓                  ↓                  ↓
共享动作头：       DiT                DiT                DiT
                    ↓                  ↓                  ↓
监督：          a₁:₁₊₁₅           a₂:₂₊₁₅           a₃:₃₊₁₅
```

相同QFormer/DiT权重被多个节点共用，不是三份独立网络。
VLM提供各节点的图像/语言特征，冻结；Q、关联模块和整个DiT参与训练。
每节点16个query、内部状态1024维，时间关联hidden256，最后映射32维动作条件。

**这里没有在节点间输入真实动作、没有物理T、没有cycle或正逆递推。**
这是“动作监督塑造条件时间线”的第一步，不是旧愿景全部实现。
节点之间的跨时间读取是学习的attention关系，不能先验叫作物理因果关系。

### 遮挡训练

```text
可见o₁              隐藏o₂              可见o₃
   ↓                   ×                   ↓
  H₁              缺失节点查询             H₃
   └───────────────────┬───────────────────┘
                       ↓
                  推断时刻2条件
                       ↓
                  同一个DiT → a₂
```

隐藏时同时移除该节点图像产生的局部特征和VLM cognition，不能从动作头旁路看答案。
时刻3相对时刻2属于真实未来：本阶段因此是**特权教师**训练。
不能直接将A的masked/full动作成绩写成可部署策略表现。

## 8. 三个控制组的关系

```text
pooled：
H₁,H₂,H₃ ── 聚合 ── 同一条件bank ── 各时间点读出

independent：
H₁ → S₁；H₂ → S₂；H₃ → S₃
每节点只能读取自己；隐藏节点没有其他视觉证据

timeline：
H₁,H₂,H₃ ── 保留独立时间地址 ── 每个目标读取可见节点 ── S₁,S₂,S₃
```

参数/数据/动作监督预算匹配，但路由和可利用信息不同：
independent是“禁止跨节点信息”的结构消融，不是能读取等量上下文的强memory。
pooled只是压缩对照，不是官方WoG；当前pooled的共同条件不能替代未来需要的
强时间寻址memory对照。因此三组首轮结果用于可行性，不能单独完成新颖性归因。
所有组同权重初始化，pooled因聚合基态不同而初始输出不同，必须报告初始指标。

## 9. A完成以后还需要B

```text
训练完成的Timeline-A（看到真实未来）
                   ↓ 冻结，提供逐节点监督目标
                   │
历史＋当前＋指令 ── 在线学生 ── 预测/学习timeline表示
                         └── 动作头 ── 当前动作

部署：只保留在线学生，不读取真实未来
```

B尚未实现；计划优先比较WoG式“表示监督、动作解耦”，再比较显式读取预测timeline。
这与已经完成的Stage3阶段编号无关；不把“A完成”叫作最终可部署WhenAmI完成。

本轮训练优先回答：**显式跨时间条件关联是否比聚合或独立节点更有效？**
B回答：**教师的优势能否转化为不看未来的策略能力？**
