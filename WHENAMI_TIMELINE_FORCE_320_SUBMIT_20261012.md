# WhenAmI 时间线强读出：320卡两阶段提交与验收

2026-10-12 UTC。独立实现目录 `whenami_timeline_force/`；
旧 A/B/C/D、旧 U 和 WoG 参考源码均不修改。
实际本机/集群验收状态以同目录最终报告和 `LOCAL_VERIFICATION.json` 为准，
本文的提交指令不能代替实验有效性与多节点PASS。

## 1. 这次到底改变了什么

```text
一张帧 h_i / z_i → 一次逐帧编码
          │
      宽节点 seed
          │
  同事件 ID + 证据 cutoff
          │
 新观测 → 更新旧节点 S_i^e
          │
 当前节点读取本次更新后的旧节点
          │
       S_t^t
          ├→ condition projection → C_t
          └→ node cognition readout → z_node,t
                          │
               一个 DiT(C_t,z_node,t)
                          │
                       action
```

raw VLM cognition 不直接进入 DiT。
保留宽节点，不把全部视觉/语言信息挤进16×32低维条件；
同一节点先由时间更新塑形，再产生两种原 DiT 接口输入。
该设计是“所有控制条件来自节点”，不是独立 B/D 分支或 action ensemble。
有结构瓶颈不保证模型真的使用 future/history；真实控制收益必须验证。

节点 `S_i^e` 指截至证据 e 对事件 i 的解释，不是概率校准belief。
first release 是最近≤3个实际观测的窗口重建，
不是跨episode/长程持久状态、不生成未来视频、不做候选动作规划。

## 2. 两阶段

### Stage1：A+ visual-only teacher

初始化：现有审计的 A50k 完整模块，新增节点 cognition lift/readout。
参数覆盖严格检查；改变 forward 后不承诺旧 A 数值等价。

训练：已有 OXE27 **全量 TRAIN 源索引**，不是小型缓存评测集。
每条采样轨迹四个实际视觉锚点 p0/p1/t/f。
每个batch六个等权FM请求：

```text
因果：S0^0 → a0    S1^1 → a1    S2^2 → a2
回溯：S0^3 → a0    S1^3 → a1    S2^3 → a2
```

每4个updates的一组采样遮蔽中间节点1视觉/cognition，继续相同六项FM。
遮蔽比例固定，不能事后只为某组调整。
teacher不输入任何执行命令/动作label；labels只进入DiT noisy-action训练和FM target。
每个版本独立遵守prefix，不能fullhidden切成早期版本。

### Stage2：causal U student

初始化：**新Stage1的最终50k teacher**；
复制编码器、时间U、条件投影、node cognition、DiT，
新增执行命令adapter/辅助P。
不能默默把旧A50k叫成新A+ teacher。

```text
student ≤3 observed nodes + complete executed intervals
    → shared timeline
    → current state C_t,z_node,t
    → original DiT
```

训练目标：
- 所有三种合法prefix最后当前节点的action FM；
- 当前节点条件与node cognition的hindsight目标；
- 同一旧event1在cutoff1/2下的reader-semantic after/变化监督。

teacher整个模型冻结。student主体、time/node readout/DiT/adapter/P训练。
旧节点执行命令可能简化辅助旧动作度量，
不能以旧节点distill小或MSE低证明物理语义；
主结果是合法当前控制和单一checkpoint闭环。

## 3. 固定正式配方

每个任务 **20物理节点×16PPU=320**：
- 50,000 optimizer updates，从阶段step0完整日程。
- global batch320轨迹窗口；micro1，draws2。
- 已训练主体峰值LR1e-6；新lift/readout/adapter/P峰值1e-5；
  warmup1000，后cosine到峰值的10%。这是温和适配配方，不声明最优。
- FP32原参数/Adam，FSDP BF16计算、FP32 reduce；LLM activation checkpoint。
- hybrid_zero2，每节点16rank shard；不回退到8卡。
- 每5k保存/DEV；相同三采样seed42/43/44。
- tqdm、永久PROGRESS/ETA_train/ETA_total、保存/评测PHASE日志。
- 单次batch的六/三节点监督有额外成本；
  不能把窗口数与原WoG单节点observations/s直接混比。

100/120checkpoint用于一次性**实际集群保存恢复验收**，
通过后继续同一日程到50k，不重新warmup。
常规main不每100步保存，只每5k。

## 4. 现在提交 Stage1

在平台选择20节点、每节点16卡。平台 `WORLD_SIZE=20` 是节点数，
`RANK=0..19` 是节点rank；torchrun会生成训练rank0..319。
不要手工把平台WORLD_SIZE设成320，也不要把RANK都设为0。

```bash
STAGE=teacher VARIANT=revise \
RUN_ID=timeline_force_Aplus_320_50k_01 \
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_whenami_timeline_force_320.sh
```

默认PHASE=run依次执行：

```text
320rank tiny topology / Adam-RNG nextupdate equality probe
    → real model updates1–100 / save100
    → newprocess resume / updates101–120 / save120
    → verifysource/topology/gradients/DEV/Adam-RNG
    → main从120继续到50000
```

不需要用户先单独提交另一个小任务，
集群验收失败就停止，不拿320卡悄悄跳过错误继续正式训练。
不声称本机单卡已经验证320rank。

输出：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/training/
  timeline_force_Aplus_320_50k_01/
    topology_probe/
    admission1/
    admission2/
    main/
      train.jsonl / progress.jsonl / phase_events.jsonl
      checkpoints/step-005000 ... step-050000
      development/step-000120 ... step-050000
```

## 5. Stage1完成，再提交Stage2

首先确认Stage1 `main/COMPLETE.json` 和最后checkpoint COMMITTED完整，
源码/metadata一致；新Stage2 preflight拒绝未完成/engineering teacher。

```bash
STAGE=student VARIANT=revise \
TEACHER_CKPT=/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/training/timeline_force_Aplus_320_50k_01/main/checkpoints/step-050000 \
RUN_ID=timeline_force_U_revise_320_50k_01 \
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_whenami_timeline_force_320.sh
```

这也单任务320卡，不是和Stage1一起占640。
Stage2结束就是可部署策略，对应WoG StageII结束。
可选下游SFT是Stage3，不是强制需要修补模块。

### 推荐的同信息机制对照（卡空出来再跑）

同新A+ teacher、同初始化/帧/命令/loss/日程，仅旧节点query来源不同：

```bash
STAGE=student VARIANT=direct \
TEACHER_CKPT=/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/training/timeline_force_Aplus_320_50k_01/main/checkpoints/step-050000 \
RUN_ID=timeline_force_U_direct_320_50k_01 \
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_whenami_timeline_force_320.sh
```

direct每次从原始历史证据重建旧节点，
revise接续同事件上个prefix版本，两者都读取相同完整合法证据，
当前节点都读本次形成的旧节点；两者不是两个候选动作融合。
graph/FLOPs差别如实量测，不能仅参数一样就称完全等计算。

## 6. 中断恢复

如果已有main中断，使用**新RUN_ID**，显式给最后完整checkpoint；
训练入口校验同源、同teacher、同topology、同recipe及Adam/RNG。

```bash
STAGE=teacher VARIANT=revise PHASE=resume \
RUN_ID=timeline_force_Aplus_320_resume_01 \
RESUME_FROM=/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/training/timeline_force_Aplus_320_50k_01/main/checkpoints/step-005000 \
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_whenami_timeline_force_320.sh
```

student恢复同时保留同一个TEACHER_CKPT。
不改globalbatch/topology、不重新初始化optimizer、不删除原父checkpoint。

## 7. 评测与部署交接

独立 `whenami_timeline_force/inference.py`：
- 只接受新student checkpoint；
- 读取≤3帧与完整已执行normalized区间；
- 只一个DiT，9Euler intervals、FP32noise/updates、BF16policy；
- 输出normalized action，denormalize恢复OXE标准动作。

机器人wrapper需核查：
actual action→标准7D→normalized命令的正向变换，
与策略action→实际机器人command的逆向变换；
视角、控制频率/时间ID、重规划prefix、gripper处理一致。
原agent已交付的B/D wrapper不热改，新架构单独交接。

已有DEV三节点可诊断middle因果/hindsight和last因果；
它不等于新四锚点teacher全量特权效果，
更不等于SIMPLER success。
未来正式评测应使用同checkpoint、同协议、配对episode并记录失败阶段/时延。

## 8. 证据边界

“timeline相关”来自代码里的同事件ID/证据cutoff、版本更新、当前动作读出，
不是画图中多写一个timeline字样。
但初版仍是有限窗口；不能直接声称长程persistent timeline已经完成。

没有性能保证或首创保证：
WoG/PFD/ChainVLA/RB-VLA是实际近邻，
固定事件版本的修订监督与合法当前控制需要强baseline/闭环验证。
新A+ teacher50k与学生50k是新增预算，比较旧方法必须披露完整训练谱系。
