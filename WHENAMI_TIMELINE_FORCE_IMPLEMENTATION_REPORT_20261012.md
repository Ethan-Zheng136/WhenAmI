# WhenAmI 时间线强读出：实现、真实 smoke 与320卡交付报告

2026-10-12 UTC。本次已实现并完成下列**本机工程验收**。
**320rank正式FP32-state HSDP验收尚未执行**，会由提交脚本在集群先执行；
未训练出新的正式A+或U，不声称性能/论文新颖性已经验证。

提交文档：`WHENAMI_TIMELINE_FORCE_320_SUBMIT_20261012.md`。
源码：`whenami_timeline_force/`，脚本：`run_whenami_timeline_force_320.sh`。
证明：`whenami_timeline_force/LOCAL_VERIFICATION.json`；
结果目录：
`/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/local/`。

## 1. 最终实现与timeline的联系

```text
逐帧视觉/语言编码 h_i,z_i
          │
宽节点seed：保留4096维cognition的分组信息，融入16×1024节点feature
          │
同一事件 i，不同证据截止点 e → 节点版本 S_i^e
          │
每次新证据到达，先更新旧节点，再形成当前节点
          │
        S_t^t
          ├──→ 16×32 condition C_t
          └──→ 4096 node cognition z_node,t
                         │
                   一个原DiT
                         │
                     当前action
```

没有raw z_t→DiT的直接通路、独立B路径、Dmemory/Bridge或动作ensemble。
统一节点仍可保留current视觉/语言内容，不人为要求所有决策都依赖历史。
动作loss有路到达旧节点并不等于策略已经学会有效修订。

`S_i^e`是动作条件表示，不是已标注的事件语义或概率校准belief。
第一版是最近≤3帧有限窗口重建；不称长期持续时间线或物理逆动力学。

新增node cognition约18M参数；全教师可训练参数8,080,213,223。
保留原VLM/QFormer/DiT及可兼容temporal参数，修改前向语义后不承诺旧A数值等价。

## 2. 两阶段训练已经有独立入口

Stage1：
- 旧A50k warmstart，不从零训练VLM；
- OXE27完整TRAIN源索引，四个实际视觉锚点；
- 三个causal当前节点 + 三个hindsight回溯节点等权FM；
- 每4个updates遮蔽中间节点1所有视觉/cognition shortcut；
- teacher不读取执行命令，避免抄早期action标签。

Stage2：
- 必须加载完成的新Stage1 teacher；正式入口拒绝旧A/engineering teacher；
- 三个合法prefix当前节点FM，teacher对C/z的hindsight监督；
- 同一旧节点在cutoff1/2的reader-semantic revision loss；
- reader-semantic度量同时读取该版本的C和node cognition，
  不重新加入raw cognition旁路；
- student读取完整已执行区间；不读取current待预测命令/未执行tail；
- teacher参数冻结，student condition经过teacher reader的输入梯度保留。

正式配方每阶段320卡/50k，主体LR1e-6，新模块1e-5，warmup1000，
globalbatch320，micro1，draws2，常规save/eval每5k。
配方是有证据的温和适配起点，不称最优或保证收敛。

## 3. 本机PASS清单

### 3.1 单元与launcher：9项

日志：`local/unit_and_launcher_final.log`。

- 所有reader输入来自同一节点；
- 稳定时间ID／全局origin平移不改变相对时间结果；
- future不影响此前prefix版本；
- 隐藏视觉输入不可走own-node shortcut；
- 学生不允许回溯历史action作为主FM；
- teacher不允许动作输入；
- direct/revise参数匹配，同信息组织对照；
- 小FP32完整Adam/RNG恢复下一步一致；
- 320/16/50k/save5k/明确resume/freshRUN_ID的launcher约束。

### 3.2 数据：27域81条原TRAIN样本

`local/DATA_AUDIT.json`：
四个不同实际时间锚点、完整已执行区间、合法action mask，
训练索引与已有held-outDEV隔离，train_overlap=0。
这是跨域抽查，不冒充扫描全部OXE轨迹或验证每个机器人命令语义。

### 3.3 真实8B权重与强读出

`local/smoke_real_02/RESULT.json`：
- 实际A50k模型1548项tensor加载，新增node cognition3项明确列出；
- teacher prefix/future遮蔽、隐藏视觉、时间origin检查；
- raw z_t不进入reader：实际reader hook与节点C/z逐项一致；
- causal/hindsight FM分别对VLM/Q/U/node readout/DiT有有效梯度；
- 当前FM→本次更新后的旧节点梯度L1 **0.12439365685**；
- 同信息替换为旧版本，实际velocity最大差 **0.09375**；
- isolated revision loss通过冻结teacher reader回传student输入梯度；
- teacher不alias，不更新，不产生参数grad；
- 一步完整模型AdamW，1512项parameter states；
- 独立decoder与normalforward一致，future/current-label/padding无泄漏检查；
- 单PPU峰值allocated **76.59GiB**。

重要：放入的是A50k权重，**不是已经训练好的新A+**；
上述敏感性是工程路径证明，不是动作改善或event理解证明。

### 3.4 实际Stage1训练入口、保存与新进程恢复

本机仅短engineering：
`train_teacher_05` updates1–2，
`train_teacher_resume_02`从step1恢复并执行step2。
实际Adam/RNG/DCP完整保存恢复；复算的step2 loss一致，
6项选定tensor共16,423元素逐位一致。

`local/ACTUAL_RESTART.json`：
**不称整个8B所有tensor逐位一致**，也不称320卡恢复已过。
CPU小模型FP32与实际单卡BF16状态的检查分别记录，不能混为正式精度。

### 3.5 实际Stage2训练入口

`train_student_01`：
从新格式engineering teacher加载1551项公共tensor，
执行命令adapter/P为明确新参数；teacher完整加载无遗漏。
真实数据联合FM/currentC/currentz/revision，所有主干/新模块梯度有效，
一次Adam更新、完整保存、DEV sampler通过。
这是工程teacher，不作为正式训练模型对外评测。

### 3.6 保存学生的独立raw-image推理

`local/INFERENCE_VERIFIED.json`：
- 新student DCP1559项严格加载，无随机遗漏；
- 原始RGB部署接口与训练数据接口，在同Euler noise下采样逐位一致；
- current动作标签改变不影响模型输入；
- 输出16×7，normalized/denormalized有限；
- 不加载teacher、不用future、不ensemble；
- 本机推理峰值allocated **17.08GiB**。

未验收远端NVIDIA环境和机器人SIMPLERwrapper，
其控制时序/命令正向归一化/视觉/动作处理仍须独立对齐。

## 4. 调试中发现并处理的问题，不隐藏失败

- `smoke_real_01`：activation-checkpointwrapper改变named_parameter前缀，
  alias检查KeyError；改为规范化名字，`smoke_real_02`通过。
- `train_teacher_01`：gripper指标对Long求mean；
  转FP32指标统计，后续实际DEV通过。
- `train_teacher_02/03`：本机Torch的NO_SHARD嵌套FSDP，
  未被forward执行的ViT末层state_dict缺失；summonfull尝试也失败。
  单卡工程测试改为rootFSDP，实际完整保存成功。
- `train_teacher_resume_01`：单卡全Adam状态+rootFP32reduce临时buffer超过96GiB；
  单卡engineering用BF16reduce，后续完整恢复下一步成功。

**正式320卡没有采用这些单卡精度妥协**：
仍是已验证过的nested16rank HSDP方式、FP32原参数与Adam，
BF16计算、FP32梯度reduce。
因此必须在真实集群重新验收；不能把rootNO_SHARD PASS当nestedHSDP PASS。

## 5. 已有工作、novelty与代码借鉴

直接复用的是本项目此前真实训练过的：
WoG reference backbone/DiT、A的局部Q/temporal参数接口、
stage3_perf topology/CheckpointIO、原OXE准备索引。
没有clone无关大仓库，也没有把新依赖塞进原训练环境。
旧源码208项hash核对未变。

独立联网材料/归档：
`outputs/whenami_unified_20261012/research/FETCH.json`；
WoG/PFD/ChainVLA/RB-VLA/SimpleMemVLA都有很近的部分。
HSD本轮原站受限，不能冒称全文近邻审计已完成。

不能以“动作塑形latent”“futureteacher”“可修订memory”
“速度场差值蒸馏”单独主张首创。
候选贡献是固定事件身份与证据cutoff的节点版本修订，
当前节点读取本次更新后的历史，所有动作条件出自这个节点，
并在同信息directhistory/匹配预算与闭环上证明作用。
目前是可执行研究候选，不是保证新颖或CVPR录用的方法。

## 6. 尚待集群与效果验收

|项目|状态|
|---|---|
|真实模型、本机梯度/因果性/Adam|PASS|
|实际教师保存、新进程恢复、学生保存|PASS，单卡engineering|
|保存student独立rawimage推理|PASS，单PPU|
|320卡/节点16卡通信与分片保存恢复|待提交脚本实测|
|正式FP32-state HSDP精度/训练稳定性|待集群验收|
|A+是否真的提供更有用hindsight/revision|待50k训练与配对DEV|
|U是否超越B/D/WoG|待单checkpoint SIMPLER|
|long-term timeline/event语义|当前未实现/未验证|

提交脚本默认：
320rank topology probe → 真实模型1–100 → 新进程101–120 →
核查后main从120连续到50k。
失败自动停止，不热改源码、重置optimizer或跳过验收。

## 7. 如何开始

平台20节点，每节点16卡：

```bash
STAGE=teacher VARIANT=revise \
RUN_ID=timeline_force_Aplus_320_50k_01 \
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_whenami_timeline_force_320.sh
```

当前先提交这个Stage1。Stage2以及同信息direct控制的完整命令见提交文档，
都为单任务320卡，不要求同时占640卡。
A+完成前不能提交依赖它的正式student。

常规checkpoint每5k，10k/20k/30k/40k/50k可评测。
本机没有启动任何正式长训练。
