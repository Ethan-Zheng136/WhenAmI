# Reader Study 01：历史提交指令（任务已完成）

2026-10-02：用户提交后，16组正式训练/开发评测已全部完成，结果见
`READER_STUDY_RESULTS.md`。**不要重复提交下面的指令到同一RUNS目录。**
以下保留当时的资源和验证记录，不代表当前仍待提交。

## 提交资源与命令

平台申请 **1节点、每节点16卡**，命令：

```bash
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_reader_study.sh
```

不用另外提交prepare。脚本自动：
1. 提取已经准备好的18,432个train/dev窗口的两个端点特征；
2. 同时执行4个对照×4个训练种子，每卡一个独立实验；
3. 每实验5000步，batch32、LR1e-4；保存最终checkpoint；
4. 在2048个dev窗口上做动作采样，并汇总种子结果和配对比较。

不是16卡DDP一个模型；不要改节点数或把global batch乘16。
这次用16卡是为了同时推进16项必要对照，而不是资源保守导致顺序等待。

四组：冻结读出/可适应读出 × 固定均值/真实Future。
适应通过每层rank32残差模块实现，只训练K和新适配器，**不写入原端点权重**。
实际K参数1,207,712，读出适配器参数1,572,864。
真实Future分支是oracle诊断，不是部署策略。

## 已完成的前置工作

- 新raw数据：train16384窗口/2500 episodes，dev2048/305，test2048/315。
- 跳过原RLDS流前2048 episodes，超出旧pilot扫描范围；
  以episode内容hash固定分区，检查无跨分区同episode。
- train/dev放`data_01/train_dev/`；test物理隔离在`data_01/sealed_test/`。
- 测试集没有编码或评测，不会被本次launcher打开。
- 48项测试通过；四格各20步真实Past DiT、batch32 smoke均通过。
- 新数据16窗口端到端特征提取、适应训练、严格重载、动作评测通过。
- 原reader权重保持不变；固定组adapter不更新，适应组adapter确实更新。
- launcher `bash -n`、dry-run、16个worker任务映射检查通过。
- 新16卡整任务尚未实跑；单卡smoke不冒充整节点验证。

## 产物与监控

根目录：
`/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_reader_study/`

```text
submission_ready_01.json
data_01/train_dev/manifest.json          # 集群特征提取完成后生成
reader_study_encode_01.log
reader_study_sweep_01.log
runs_01/
  frozen_mean_seed42.log                # 每个实验独立log；seed42/43/44/45
  frozen_mean_seed42/
    train.jsonl
    development.jsonl
    checkpoints/step-005000.pt
    development_eval/results.json
    complete.json
  ... 其余15组 ...
  development_comparison.json
```

不要重复提交到同一个RUNS，脚本拒绝覆盖。无精确resume功能；
如失败，先定位编码/训练哪个阶段，保留现场，不删除原checkpoint。
本地测试结果不作为收益证据；等正式多种子结果回答冻结接口是否限制Future增益。
