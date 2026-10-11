# A50k → 原版 B0 / B1：独立 640 卡实验

日期：2026-10-06。新增两份 **20 节点 × 16 PPU = 320 卡**的任务，合计 640 卡。
这是原版 B 的更强初始化对照，不是 C，也不是把一个任务的 batch 翻倍。

## 1. 这次究竟改了什么

| 项目 | 原 B | 本次 B50 |
|---|---|---|
| 学生模型初始化 | Timeline-A10k | **Timeline-A50k** |
| 冻结教师 | Timeline-A10k | **Timeline-A50k** |
| 学生输入 | past + current | 不变，真实 future 不进入学生 |
| B0 | action FM | 不变 |
| B1 | action FM + 0.1 × condition cosine loss | 不变 |
| 学习率、采样、优化器、推理器 | 原 B 配方 | 不变 |

固定初始化/教师：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_joint/training/
timeline_joint_timeline_320_50k_01/main/checkpoints/step-050000

COMMITTED.json SHA256:
f785848d08a55bc5aee4e821b3feac61b3543665e6117982b7eeb0cbbb301714
```

**不是从旧 B20k 接着训练，也不恢复 A 的 optimizer。**
第一次从 A50k 只加载模型，B 的 AdamW 与学习率调度重新开始。
随后本任务 100 → 120 → 20k 的衔接，才会恢复本任务自己的模型、Adam、逐 rank RNG。

```text
                         A50k 模型参数
                          /        \
                         v          v
            学生 B：可训练          教师 A：冻结
            past + current         past + current + 真 future
                  |                         |
            当前时刻 condition      利用 future 的【当前】condition
              /           \                 |
             v             v                | stop-gradient
       原版 DiT action   辅助 forecast_head <─┘
             |             |
         action FM     cosine alignment
             |             |
        B0、B1 都用      仅 B1 加入总 loss
```

注意：教师目标是**被真实 future 信息改善的当前节点条件**，不是 future 节点条件。
辅助头输出不直接输入 DiT。没有混入 C 的固定 t+4 查询、32-token future read-in 等改动。
B0 仍计算教师前向用于匹配计算和诊断，但不使用教师损失更新。

## 2. 提交指令

平台为每个任务配置 **20 节点，每节点 16 卡**。分别提交以下两份命令。
平台注入的 `WORLD_SIZE` 在启动器入口表示节点数 20，`RANK` 表示节点 rank；
torchrun 内部才是 320 个训练 rank。不要手动把入口 `WORLD_SIZE` 写成 320/640。

### 任务一：B0，只有 action loss

```bash
B50_VARIANT=action_only PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_b50_320.sh
```

### 任务二：B1，action + 教师 condition 对齐

```bash
B50_VARIANT=condition_align PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_b50_320.sh
```

无需重新准备 OXE，复用原 B 的 OXE27 TRAIN 索引与 DEV 分割。
不要把这两条命令放进同一个 40 节点任务顺序执行。

## 3. 固定训练配方

- 每组 20,000 个更新，global batch 320 条样本，micro batch 1/卡。
- 每条样本三个真实时间锚点；原版可变间距 1–12；学生物理截断到前两个。
- 每个当前动作标签 2 次 flow-matching 噪声采样。
- 主干峰值 LR `1e-5`，辅助头峰值 LR `1e-4`，warmup 1,000，cosine schedule。
- 原版 hybrid_zero2、LLM activation checkpointing、gradient clipping 1.0。
- 初始 0 step 做因果 DEV 基线；每 1,000 step 做 DEV，最终 20k 使用 3 个采样 seed。
- 常规每 2,500 step 存模型 + Adam + RNG；另存 100 / 120 step 验收点。
- 保留 tqdm、秒/step、ETA、显存峰值；指标不是 SIMPLER 成功率。

`PHASE=run` 自动执行：

```text
本地源码/教师/数据预检
  → 实际 320-rank 小模型拓扑与断点恢复测试
  → 完整学生+教师训练到 100，保存并评测
  → 新进程恢复到 120，保存并评测
  → 检查梯度、损失、冻结教师、恢复和 27 域因果 DEV
  → 从 120 继续至 20,000
```

前 120 步属于 20k 总更新，不是额外再跑 20k。
本地单卡 smoke 不替代实际 320 卡验收；任一关失败不进入后续大训练。
100/120 步会有保存、评测、重新加载停顿，不能只看 tqdm 暂停判断卡死。

## 4. 输出与保护

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b50/training/
├── timeline_b50_action_only_A50k_320_20k_01/
└── timeline_b50_condition_align_A50k_320_20k_01/
    ├── topology_probe/
    ├── admission1/
    │   └── development/step-000000/RESULT.json  # 本次 A50k-causal 基线
    ├── admission2/
    └── main/
        ├── recipe.json
        ├── LOAD_VERIFIED.json
        ├── RESUME_VERIFIED.json
        ├── progress.json
        ├── train.jsonl
        ├── development/
        ├── checkpoints/step-020000/
        └── COMPLETE.json
```

启动器拒绝覆盖已存在的任务段。重复提交前先检查结果；不要无脑换 RUN_ID 重跑。
`PHASE=full` 仅用于已成功完成 120 步验收、且 `main` 还没创建的情况，
不是任意中断恢复开关。

新增代码位于 `whenami_timeline_b50/`，启动器为 `run_timeline_b50_320.sh`。
原 B 的模型、数据、损失、评测与共享 runtime 仅复用，不修改。
原 A/B/C checkpoint 不覆盖，不删除。

## 5. 如何判断是否改善

1. 同一训练/评测后端比较本次 0-step A50k-causal、B0、B1；
   先确认适配没有损伤较好的 A50k 起点。
2. 对照旧 A10k→B0/B1，回答升级起点是否有帮助；
   A50k 的额外预训练预算必须披露，不能称等总算力比较。
3. 同一评测协议跑 SIMPLER，分别报告 Google VM/VA 与 WidowX，
   不能用离线 MSE 或夹爪准确率替代闭环成功率。
4. 不预先保证 20k 优于 10k，也不把“更强初始化”称为“最终最好的 B”。
   这是验证该假设所缺的直接实验。

本地验证证据由 `whenami_timeline_b50/LOCAL_VERIFICATION.json` 记录；
实际集群验收则以各任务 `admission2` 和后续 `main` 产物为准。
