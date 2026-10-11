# B/C 六项续训：各自20k → 累计50k

日期：2026-10-07 UTC。用户明确要求旧B四组（旧/新起点各B0/B1）及C两组续训。
这里“旧/新B”分别表示A10k/A50k初始化，不是本轮续训的step。

**每条为独立20节点×16 PPU=320卡任务。六项同时运行共1920卡，可以分批提交。**
本机仅做有限smoke、恢复测试和原checkpoint审计，不在测试机执行正式续训。

**2026-10-07 15:20 UTC更新：六项均已完成本地验收。**
B的22项测试、C的16项测试通过；六份真实父模型及对应教师的前向/反向检查完成，
两份LOCAL_VERIFICATION均已生成，六case本地preflight通过。
C实际检查确认1549项状态完整加载、32个condition tokens、read_scale分别为0/1。
本机smoke已退出，没有在测试机启动正式续训。
实际320rank完整Adam恢复及同后端预测回放仍由提交后的自动关卡执行。

## 0. 开始前的状态检查

真正完成的本地验收由以下文件判定，不凭本文存在就声称已经通过：

```text
/home/zhengguantian/WoG_fresh/whenami_b_continue/LOCAL_VERIFICATION.json
/home/zhengguantian/WoG_fresh/whenami_c_continue/LOCAL_VERIFICATION.json
```

须有`local_passed=true`，且本地preflight通过源码/父checkpoint检查。
未完成时launcher会拒绝运行，不绕过验收。
实际320卡完整模型+Adam/RNG恢复及同后端DEV回放，在提交后的自动验收段完成。

## 1. 六条命令

### 旧B0：A10k初始化的action-only

```bash
B_CONTINUE_CASE=old_b0 PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_b_continue_320.sh
```

### 旧B1：A10k初始化的condition-align

```bash
B_CONTINUE_CASE=old_b1 PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_b_continue_320.sh
```

### 新B0：A50k初始化的action-only

```bash
B_CONTINUE_CASE=new_b0 PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_b_continue_320.sh
```

### 新B1：A50k初始化的condition-align

```bash
B_CONTINUE_CASE=new_b1 PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_b_continue_320.sh
```

### C-aux：future仅辅助监督

```bash
C_CONTINUE_CASE=c_aux PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_c_continue_320.sh
```

### C-read：future直接参与动作

```bash
C_CONTINUE_CASE=c_read PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_c_continue_320.sh
```

每份平台任务申请20个节点，入口`WORLD_SIZE=20`表示节点数；
不要手工写320/1920，不要把六条命令塞进同一个120节点任务。
平台注入MASTER_ADDR/RANK，容器与挂载沿用原任务。任务之间无需同步开始。

## 2. 恢复和学习率

|任务|父学生checkpoint|教师保持不变|续训主体LR|第二组参数LR|
|---|---|---|---:|---:|
|old_b0|旧B0-20k|A10k|1e-6|forecast_head：1e-5|
|old_b1|旧B1-20k|A10k|1e-6|forecast_head：1e-5|
|new_b0|新B0-20k|A50k|1e-6|forecast_head：1e-5|
|new_b1|新B1-20k|A50k|1e-6|forecast_head：1e-5|
|c_aux|C-aux-20k|A50k|1e-6|timeline：1e-6|
|c_read|C-read-20k|A50k|1e-6|timeline：1e-6|

第二组参数不同，不能把B的aux-LR直接照搬到C。
B0的forecast_head不参与其action-only损失，仍保留原优化器组结构。

- 六组各自恢复完整模型、16份Adam moment/step及320份rank RNG。
- 累计step从20,001继续，到50,000，**新增30,000步**。
- 恒定保持父20k末端低学习率，不重新warmup，不重新解释原20k cosine历史。
- global batch320、micro1、两次FM噪声draw，动作目标与数据采样公式不变。
- B保留原两个时间点因果学生及当前condition目标；C保留固定future_gap=4、
  current/future双节点监督、32个动作condition tokens。
- **C-aux的read_scale=0，C-read的read_scale=1**，从父权重恢复；
  续训用全局step调用原调度，不从0重启read-in，不把C-read退回C-aux。

此实验是“20k原训练日程 +30k末端低LR续训”，不是“从头50k cosine”。
单纯增加训练预算不能包装成架构创新，也不保证一定涨DEV或闭环。

## 3. C的父checkpoint

```text
C-aux:
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_c/training/
timeline_c_future_aux_A50k_h4_320_20k_01/main/checkpoints/step-020000

COMMITTED SHA256:
78fe858652809457835cc7fb5087daf13d69bcc1376c0d7aa307f7469e5dc4a7

C-read:
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_c/training/
timeline_c_future_read_A50k_h4_320_20k_01/main/checkpoints/step-020000

COMMITTED SHA256:
e00707e3d3331ecf9accfde8494a3a74af31f4eedfd3508f6bf0c5a9f503f071
```

四组B的准确父路径、哈希及详细恢复说明见`B_OLD_NEW_CONTINUE_50K_20261007.md`
和`whenami_b_continue/contracts.py`。
六组都不从A重新warmstart学生，不自动跟随更新的教师。

## 4. 自动验收和正式训练

每个任务都会执行：

```text
本地证明/原父模型/固定teacher/数据分割校验
→ 320-rank小模型拓扑恢复probe
→ 用原recipe恢复各自20k模型+Adam+RNG（C含read_scale）
→ 重放原20k DEV，对保存的逐样本三seed预测做同FSDP后端比较
→ 训练20,001–20,100，保存并评测
→ 新进程恢复20,100，回放校验，训练到20,120，保存并评测
→ 校验所有rank恢复记录、优化器step、固定低LR、梯度
→ 新进程恢复20,120，继续至50,000
```

首次加载用不可变的父recipe；后续保存和恢复用包含父完整recipe的续训recipe。
不修改原checkpoint的COMMITTED，不放松共享CheckpointIO的校验。
同FSDP预测回放最大绝对误差容差1e-6；失败不自动放宽或绕过。
100/120个新增验收更新计入新增30k，不额外跑一段20k。

每2500步存模型/Adam/RNG；每1000步及每个保存点做DEV。
25k/30k/35k/40k/45k/50k用采样seed42/43/44，其余用42；
中间曲线比较只取共同seed，避免假下降。
tqdm显示累计step，progress另记录新增更新数、速度、ETA。

## 5. 输出与保护

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_b_continue/training/
  b_continue_old_b0_20k_to50k_01/
  b_continue_old_b1_20k_to50k_01/
  b_continue_new_b0_20k_to50k_01/
  b_continue_new_b1_20k_to50k_01/

/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_c_continue/training/
  c_continue_c_aux_20k_to50k_01/
  c_continue_c_read_20k_to50k_01/
```

各自包含`topology_probe/`、`admission1/`、`admission2/`、`main/`，
最终模型为`main/checkpoints/step-050000`。

原20k模型和评测不覆盖；D、pooled-B及另一服务器的SIMPLER不热改。
需要中断恢复时先核对已保存的新checkpoint；不盲目重复fresh命令。
`PHASE=full`只适合20,120已验收、`main`还没有创建的情况。

## 6. 本地证据范围

- B：22项CPU/launcher/嵌套FSDP恢复测试；4份实际父模型前向反传；
  全部64份Adam和1280份RNG元数据/计数校验。
- C：16项对应测试，包含read_scale的恢复/续训不重置；
  2份实际父模型前向反传；32份Adam、640份RNG和实际DCP read_scale校验。
- 单卡smoke是每个父checkpoint一次固定样例前向/反向，无长训练；
  不加载单卡8B完整FP32 Adam，也不宣称单卡检查替代真实320rank验收。
- 独立小模型恢复测试比对下一更新参数与Adam moments逐值相同，
  并核对Python/NumPy/Torch RNG；小模型人工设置20k计数用于恢复夹具，
  不是声称小模型训练了20k。

本节是验收要求，完成状态以两份LOCAL_VERIFICATION和各自真实RESULT为准；
若文件尚未生成，不得称全部smoke已通过。
