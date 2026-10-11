# 因果学生20k：可提交

**规模澄清：这是20k缓存学生机制对照，不是全数据／大模型联合主训练。**
每实验只训练6.48M新增参数，VLM特征已缓存；16卡是16个独立实验。
详见 `TRAINING_SCOPE_CLARIFICATION.md`。不要仅增加本脚本卡数来冒充主训练。

2026-10-02。代码、teacher检查、真实模型smoke与resume验证已完成；
**正式集群任务尚未提交，由用户提交以下命令。**

## 资源与指令

**1节点 × 16卡**，每卡一个独立实验：

```bash
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_causal_student_20k.sh
```

无需再跑prepare。自动完成16组训练、三种采样seed动作评测和跨训练种子比较。

每训练种子42/43/44/45各四组：
1. `latent`：等参数普通学生；
2. `forecast`：增加Future条件预测监督；
3. `distill_past`：forecast加无Future教师的行为监督；
4. `distill_future`：forecast加真实Future教师的行为监督。

四组学生完全相同的可训练容量6,484,672参数，部署都只接受Past H/z/C。
真实Future只用于训练目标/冻结教师，不能进入学生forward/sample。
不是16卡DDP；不要把每组batch32乘16当作学生global batch。

## 初始化、步数与优化

- 各学生从上一轮**同seed的adaptive_mean最终5k**加载K/DiT适配器，
  新建相同随机R。原Stage1/2参数冻结。
- 两类教师为同seed的adaptive_mean/adaptive_future最终5k；都冻结。
- **20,000个新增学生更新**，不含父模型的5k，不是“从5k续到总20k”。
- 父模型只加载权重，本轮优化器重新初始化；中断后的本轮resume则恢复优化器。
- batch32，每观察4份独立FM时间/噪声；并不是4步真实时间rollout。
- LR：500步warmup至1e-4，然后cosine降至1e-5。
- 每2500步保存；每1000步开发FM；主动作评测用固定最终20k。
- 最终开发集2048窗口，推理noise seed42/43/44；测试集2048窗口继续封存。

参照WoG的是action+future辅助监督及可学习决策表示的原则，
不是宣称训练预算等同：WoG附录B.1的50k/global batch1024与当前固定缓存不同。
当前20k×32是640k窗口抽样，约39.1次训练集等效遍历，要警惕过拟合；
开发曲线与全部checkpoint保留，不看到结果后挑best作为主成绩。

## 断点恢复

同一任务中断且已有保存点时，可用：

```bash
RESUME=1 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_causal_student_20k.sh
```

- 只能同数据、超参数、总预算、教师hash、训练源码，不能借resume改实验；
- 每个未完成实验从其最新完整checkpoint继续，已完成的跳过；
- 恢复模型、optimizer、绝对步号调度和RNG，截去checkpoint之后的旧日志；
- 若某实验在第一个保存点前失败、无checkpoint，会明确报错，不静默重开覆盖。
- 本地20步实测：第10步停下后重启，最终模型/optimizer的617个张量、
  两个动作采样seed的预测数组与连续运行**逐值一致**。
  这是当前本机软件/硬件下的恢复验证，不保证跨平台改变环境仍bitwise一致。

## 提交前验证

- 59项测试通过，覆盖因果输入隔离、teacher冻结/stopgrad、action-only到R/K/
  reader adapter梯度、同噪声蒸馏、同容量四组、调度及断点恢复。
- 四组真实Past DiT各20步、batch32×4 FM，保存/严格重载/两noise评测通过。
- 所有四组首步action loss和纯action梯度完全一致。
- 8个父模型/teacher checkpoint已逐个检查同cache/seed/身份和有限权重，
  SHA256记录在 `submission_ready_20k_01.json`。
- launcher语法、dry-run及16组任务映射通过。
- 正式20k/16卡学生整任务尚未执行；已有16卡ReaderStudy成功不能替代这次验证。

## 输出

根目录：
`/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_causal_student/`

```text
runs_20k_01.launcher.log
runs_20k_01/
  latent_seed42.log
  latent_seed42/
    config.json
    train.jsonl
    development.jsonl
    gradient_audits.jsonl
    checkpoints/step-020000.pt
    development_eval_noise42/results.json
    development_eval_noise43/results.json
    development_eval_noise44/results.json
    complete.json
  ... 其余15组 ...
  development_comparison.json
```

正式任务未结束前不要重复提交相同RUNS；已完成ReaderStudy目录不修改。
