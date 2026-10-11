# 每节点16进程故障排查（2026-10-01）

## 最新进展：20节点满16卡、15k长训已完成（2026-10-02）

`past_s2_16rank_20node_from5k_01` 从正式Past 5k权重继续15k，
20×16=320个训练进程、per-device batch=8、global batch=2560、
shuffle=4096。全部20个节点退出码0、无安全终止、OOM计数保持0；
320个rank的最后批次索引均为14999，日志无重启/NaN/Inf。
节点cgroup高水位1021.32–1058.76GiB，节点上限1792GiB。

5k/10k/15k（本次任务计数）的三份模型checkpoint全部落盘；
最终32.2GB权重已读取并检查Past/DiT有限性。
已验证范围因此扩展到**这套配方的20节点15k模型训练与模型保存**，
仍不包括恢复大shuffle、40节点640进程或优化器状态保存。

## 双节点通过与测试产物清理（2026-10-01 18:12 北京时间）

`past_s2_16rank_2node_probe_01` 已完成250步，32个rank全部到达最后一批；
两个节点均正常返回0、OOM计数保持0，采样内存峰值约682/644GiB。
最后100步中位步时约1.61秒，32.2GB模型checkpoint保存并mmap读取验证通过。

经用户要求，**单节点、双节点两份250步测试checkpoint已删除**，并清理本次创建的
临时pytest目录，合计删除约64.56GB文件。测试源码、配置、训练日志、节点监控及
正式训练产物全部保留。下面的checkpoint路径仅为历史验证记录，不再是可加载权重。
审计记录：
`outputs/runtime16/cleanup_completed_tests_20261001T101157Z.json`。

下一个20节点速度测试从**原Future Stage-1 baseline**初始化，不再引用已删除的测试权重；
20×16×8=2560 global batch，与正式40×8×8一致，但shuffle/管线配方不同。
这是新250步速度测试，不是正式Past训练接续。每卡batch=8的16rank密度尚待该测试验证。

## 最新结论：单节点16卡已通过（2026-10-01 16:44 北京时间）

`past_s2_16rank_probe_01` 已完成250步，全16个rank到达最后一批，
节点外层监控返回码为0，无安全终止，cgroup OOM计数从0到0。
采样到的节点内存峰值约676.4GiB，节点上限1792GiB；
最后100步中位步时约1.59秒。

已保存并只读检查：
`outputs/stage2_16rank_probe/past_s2_16rank_probe_01--image_aug/checkpoints/step-000250-epoch-00-loss=0.1690.pt`。
文件32,237,050,769字节，torch mmap加载成功，含正确Stage-2元数据、
168个Past QFormer权重张量和DiT per-attention。

**有效范围**：独立实验副本、单节点16进程、per-device batch=4、
shuffle=4096、250步及模型checkpoint保存。说明已有一套可用的满卡配置；
尚未验证多节点、长训或恢复较大shuffle，也未验证优化器状态保存。
下面的“未在集群执行”描述保留为此前排查经过，当前状态以本节为准。

## 正式训练不受影响

生产代码仍在 `WoG/`，正式入口仍是 `run_stage2_past.sh`，每节点默认8进程。
本轮所有实验性改动在 **`WoG_16rank_probe/` 独立源码快照**。
没有修改公共 `pylibs-train`，没有替换 production launcher，也没有提交集群任务。

40节点完整 Stage 2：

```bash
NPROC_PER_NODE=8 PER_DEVICE_BS=8 LR=3.2e-5 \
MAX_STEPS=20000 SAVE_INTERVAL=5000 RUN_ID=past_s2_stage1_20k_01 \
bash /home/zhengguantian/WoG_fresh/run_stage2_past.sh
```

平台640卡、实际320进程、global batch=2560。重复提交须换 RUN_ID，避免混入旧日志。

## 已核实的证据

1. 旧用户粘贴日志中，多次16进程任务的**第一个失败是 SIGKILL / exitcode=-9**，
   出现在数据统计/输入初始化阶段；随后 torchrun 向其他rank发送SIGTERM。
   SIGKILL 本身不证明内核OOM，旧日志没有相应的节点cgroup事件，不能声称已完全定位。
2. 同期的优化器保存失败是另一个事件：NCCL ALLGATHER 超时、SIGABRT。
   不应以“增大NCCL超时”作为解释/修复所有16进程问题的方案。
3. 旧 `mmap+assign` 优化降低了模型加载内存，但旧对话明确记录：
   **优化后16进程仍失败，8进程才成功**。不能再重复宣称该补丁已解决。
4. 每个rank独立构建27个数据集的管线，独立shuffle 64,000个观测窗口。
   `DataLoader(num_workers=0)` 不代表“没有数据线程”；dlimp/TF内部仍有异步映射与reader。
5. dlimp 默认 `autotune.enabled=True`、`experimental_warm_start=True`。
   最末尾 `with_ram_budget(1)` 只设置 TF autotune 的预算，不是进程/cgroup硬内存上限，
   不会把显式64,000个shuffle元素、模型权重、解码大轨迹都限制到1GiB。
6. shuffle 位于图像解码/resize**之前**；不能简单用
   `64000×5×224×224×3` 算实际shuffle内存——输入主要仍是编码图像字节，大小依数据集而变。
7. `allocate_threads` 给27个数据集各至少一个reader/trajectory并行槽，
   不能直接把其总数设成8（会触发断言）；本实验限制共享执行线程池，不删除数据集。
8. 当统计缓存hash缺失时，各rank会独立扫描统计。
   早期多个 `0/8611` 进度条与 FMB 的8611条trajectory吻合；
   但不能证明所有后续SIGKILL也发生在同一阶段。
9. 本地测试机只有1个PPU、11 CPU配额、100GiB内存cgroup。
   **不能在这里声称验证了16个物理GPU的完整训练。**

## 实验副本的改动

- 每进程 TF private threadpool=8、intra-op=1、inter-op=2，frame map并行=2。
- 关闭 autotune 和 warm_start；不注入自动prefetch。
- **在TFDS数据读取源头**设 `interleave_block_length=1`、解码并行=1、
  文件并行=1、reader buffer=1MiB、关闭auto-cache。不是仅给最外层dataset设options。
  Installed dlimp 原先只设了cycle_length，TFDS的block_length仍默认16；
  对RLDS而言缓冲元素是完整episode，不是单张图像。
- `MALLOC_ARENA_MAX=2`，OMP/MKL/OpenBLAS线程数=1。
- 明确加载现有 baseline 对应的27数据集统计值，缺数据集直接报错，
  不在640个进程中重新跑统计。使用者须保证数据/standardization recipe一致。
- 不改变数据混合比例、原始动作归一化、时间采样和loss定义。
- 16进程每卡batch=4，对照8进程每卡batch=8；40节点时两者global batch都为2560。
- 16进程诊断每rank shuffle默认4,096，先验证满卡进程能启动、反向、保存。
  通过后再尝试 `SHUFFLE_BUFFER=32000`，使每节点总buffer槽数与8×64,000一致。
  **减小buffer改变shuffle相关性；诊断配方不是最终论文对照配方。**
- 增加阶段内存日志：导入、模型加载前后、建数据前后、FSDP后、首批/反向、每50批。
- 节点外层监控记录cgroup current/limit/peak/OOM计数、子进程RSS与线程数；
  即使某rank先SIGKILL，仍尽力留下证据。RSS相加会重复计入共享mmap页，
  判定节点内存应优先看cgroup计数，rank日志另含PSS。

## 推荐验证顺序（尚未在集群执行）

先**1节点×16进程、250步**（全27数据集、包含一次保存）：

```bash
NPROC_PER_NODE=16 PER_DEVICE_BS=4 MAX_STEPS=250 SAVE_INTERVAL=250 \
RUN_ID=past_s2_16rank_probe_01 \
bash /home/zhengguantian/WoG_fresh/run_stage2_past_16rank_probe.sh
```

平台节点数设1。不要与生产任务使用同一个输出目录。
通过后再2节点检查跨节点通信，最后才40节点。
如果仍失败，读取：

```text
outputs/runtime16/<RUN_ID>/node-<node_rank>.jsonl
outputs/runtime16/<RUN_ID>/ranks/rank-<rank>.jsonl
```

- OOM计数是否相对于开始时增加？最后内存是否接近cgroup上限？
- rank停在模型加载、统计、shuffle首批、反向还是checkpoint？
- 若RSS/cgroup正常、OOM无增量，转查平台驱逐、线程资源、PPU/NCCL日志；
  不继续默认“又是内存”。

这是一份**候选缓解方案＋可定位问题的测试入口**，不是已经验证的16卡修复。

## 本地单进程实测

所有实验都不加载VLM、不创建优化器，只读取真实OXE的27数据集混合。
用独立父进程监控，超过安全阈值主动终止，**安全终止不算内核OOM**。
统计值均显式固定为 baseline 的同一份统计，避免混入统计重算。

| 设置 | shuffle | 结果 |
|---|---:|---|
| 原dlimp读取设置 | 128 | 首批返回前RSS已达20.57GiB，18GiB安全阈值主动终止 |
| 仅末端限制TF options | 128 | 首批返回前RSS达21.16GiB，同样主动终止 |
| 连源头读取buffer一起限制 | 128 | 成功读出32样本，监控到峰值RSS约13.38GiB |
| 源头限制 + 完整训练图像增强 | 32,000 | 首批已返回，但RSS达29.82GiB，触发28GiB本地安全阈值 |
| 源头限制 + 完整训练图像增强 | 4,096 | 成功读出128样本，监控峰值RSS约20.46GiB，正常退出 |

这些是单次小型诊断，不代表相同的样本到达顺序，不代表16卡训练已验证。
private threadpool=8 **不等于进程只有8个线程**：TF reader/interleave还有独立后台线程，
本地实测线程数依然在数百量级，节点监控保留这个指标，后续可继续优化。

实验文件在：
`/mnt/pfs/zhengguantian/WoG_v2/outputs/runtime16/local_profile_20261001/`。
`bounded*` 与 `source_bounded*` 分别对应仅末端限制、源头也限制的实验。

最终回归：`tests/test_runtime_limits.py tests/test_past_stage2.py`，
**34 passed**（包括单PPU小模型FSDP存取、监控器故障/安全退出、source reader options、
混合比例与统计路径、Stage-2语义测试）。未做16物理卡完整训练。

五次本地监控中cgroup的 `oom_kill` 计数均未增加；三次超过阈值由父进程主动SIGTERM。
TF在正常退出时有既有的 AtomicFunction析构告警；成功试验以完整结果JSON、
正常返回码和样本断言共同判定，不把这些退出告警当成训练失败。

修复候选目前支持的结论是：
**输入端存在可复现的内存放大，源头缓冲限制与保守shuffle能缓解它。**
它是否足以解决历史16进程失败，必须由上述集群测试和OOM计数验证，
不能仅根据RSS×16外推（mmap共享页、节点cgroup限额、模型/优化器状态都不同）。
