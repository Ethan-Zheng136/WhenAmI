# WhenAmI：四个 A50k 起点、20k 训练完成模型的 SIMPLER 评测交接

更新时间：**2026-10-07，UTC**。
接收对象：已经跑通旧 Timeline-B / WoG 的 NVIDIA 服务器及负责 agent。

> **可以直接转发的任务说明**
>
> 请测试本文的四个固定 20k checkpoint：`B0_A50k_20k`、`B1_A50k_20k`、
> `C_aux_A50k_h4_20k`、`C_read_A50k_h4_20k`。
> 四组训练和最终 DEV 均已完成，不需要再训练。
> 沿用你已经验通的 SIMPLER 任务、seed、机器人后处理和历史缓存协议。
> **前两个沿用原 B 架构；后两个必须接入 C 架构，不能只给旧 B wrapper 换权重。**
> 先核验身份、完整加载和数值回放，再启动四组正式评测；不要只挑离线最好的模型。
> 最后回传逐 episode 结果和完整配置，不只发平均成功率截图。

## 0. 交接状态：哪些完成，哪些还没有

**已在源服务器核实：**

- 四组连续完成 1–20,000 更新，训练 loss/梯度范数均为有限值。
- 每组均有 `main/COMPLETE.json` 和 `step-020000/COMMITTED.json`。
- 每份最终 checkpoint：16 个模型 DCP 分片、16 个 optimizer 分片、320 个 RNG 文件。
- 最终因果 DEV：27 域、410 个不同 episode、3 个采样 seed `[42,43,44]`，全部落盘。
- 四份 checkpoint 的原始源码契约与当前源代码一致。
- 已从 C 的实际 DCP 权重读取 `read_scale`：**C-aux=0，C-read=1**。

**本次文档更新没有执行：**

- 四模型的新 safetensors 便携包导出及往返回放；
- 四模型在 NVIDIA 上的新接口/数值验收；
- 四模型的 SIMPLER 运行。

因此，**“训练完成”不是“四个新便携包已经交付并验收”**。
本机当前没有核验通过的这四模型新便携包；接收 agent 需要完成下文的导出/加载适配。
已有 `B20k_20261006_01` 是 **A10k→B20k**，不能把那个包的权重当成本轮新模型。

## 1. 唯一模型身份与机器可读清单

请同时拿走这两个文件：

```text
/home/zhengguantian/WoG_fresh/WHENAMI_FOUR_20K_SIMPLER_HANDOFF_20261007.md
/home/zhengguantian/WoG_fresh/WHENAMI_FOUR_20K_EVAL_MODELS_20261007.json
```

JSON 记录四个模型的完整路径、`COMMITTED.json` SHA256、DCP `.metadata` SHA256、
状态项数、模型类、variant、teacher 身份、C 的 horizon/read_scale、
共享配置/统计文件哈希、关键源码哈希和最终 DEV 结果。

**正式结果目录请使用下列 `eval_id`，不要笼统命名为 `B20k`：**

| eval_id | variant | 模型类 | 状态项数 | DiT condition tokens |
|---|---|---|---:|---:|
| `B0_A50k_20k` | `action_only` | `whenami_timeline_b.model.Student` | 1550 | 16 |
| `B1_A50k_20k` | `condition_align` | 同上 | 1550 | 16 |
| `C_aux_A50k_h4_20k` | `future_aux` | `whenami_timeline_c.model.TimelineStudent` | 1549 | 32 |
| `C_read_A50k_h4_20k` | `future_read` | 同上 | 1549 | 32 |

四组学生初始化和固定训练教师均来自同一 **Timeline-A50k**：

```text
teacher COMMITTED.json SHA256:
f785848d08a55bc5aee4e821b3feac61b3543665e6117982b7eeb0cbbb301714
```

这是训练溯源信息，**推理不加载教师，也不需要 A50k 权重**。

### 四个原始 checkpoint

```text
B0_A50k_20k:
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b50/training/timeline_b50_action_only_A50k_320_20k_01/main/checkpoints/step-020000

B1_A50k_20k:
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b50/training/timeline_b50_condition_align_A50k_320_20k_01/main/checkpoints/step-020000

C_aux_A50k_h4_20k:
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_c/training/timeline_c_future_aux_A50k_h4_320_20k_01/main/checkpoints/step-020000

C_read_A50k_h4_20k:
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_c/training/timeline_c_future_read_A50k_h4_320_20k_01/main/checkpoints/step-020000
```

它们是源服务器路径；目标服务器需要通过已有传输方式接收，并记录本地映射。
**不要使用 admission 的 100/120 步 checkpoint，不要自动选“目录里最新文件”。**

## 2. 四者的推理区别——最容易接错的地方

### B0/B1：原版 B，升级初始化而非升级推理结构

```text
一张真实历史 + 当前 RGB + 指令
             │
     每帧独立 VLM / DINO / Wan
             │
      local QFormer + Timeline
             │
       当前 condition [16,32] ───────→ 原 DiT → 16×7 action
             │
       forecast_head
             └── 仅训练辅助，不接入 DiT
```

B1 在训练中额外对齐“教师看过真实未来之后形成的**当前节点 condition**”。
B0 不加这个损失。两者推理接口完全一样，均不看真实未来。
可以沿用已经验通的 B20k 推理构建、输入和动作采样；**替换为本文的新 B 权重**。

### C-aux/C-read：因果输入，显式查询当前与未来时刻

```text
一张真实历史 + 当前 RGB + 指令
             │
     每帧独立 VLM / DINO / Wan
             │
       observed source memory
             │
     Timeline 时间查询：[t, t+4]
     两个查询均只能读取 ≤t 的观测
             │
       ┌─────┴─────┐
    current      predicted future
    [16,32]         [16,32]
       │               │
       ├─ C-aux: cat(current, current) ───────→ DiT
       └─ C-read: cat(current, future) ───────→ DiT
                      [32,32]                   │
                                            16×7 action
```

实现依据：`whenami_timeline_c/model.py::TimelineStudent.forward`。

- **两组都没有真实 future 输入**，未来节点是由已有观测预测的条件，不是未来图片。
- 两组训练都对齐教师的 current 和 future condition。
- `future_gap=4` 是模型查询的时间偏移，**不是要求固定取 4 步历史，也不是 execute_horizon=4**。
- 最终 `read_scale`：C-aux 为 0，C-read 为 1；buffer 必须从 checkpoint 恢复。
- 不要加载完成后又调用 `set_step(0)`，否则 C-read 会被悄悄变成不读预测未来。
- `variant` 和 `future_gap` 是构建参数，不能指望单靠 state_dict 恢复；需写入导出配置。
- 两组都保留 32 个 action condition tokens。不要“优化”C-aux 为 16 个。
- DiT 使用的 `reader_z` 仍为**当前节点**，不要替换为未来 cognition。
- C 没有 B 的 `forecast_head`。不能用 `strict=False` 忽略差异后称为加载成功。

## 3. 已有工具如何复用，以及不能直接运行的命令

源工程：

```text
/home/zhengguantian/WoG_fresh/
  whenami_timeline_b_handoff/runtime.py
  whenami_timeline_b_handoff/prepare_release.py
  whenami_timeline_b_handoff/replay.py
  whenami_timeline_b/model.py
  whenami_timeline_c/model.py
  whenami_timeline_joint/model.py
  whenami_timeline_transfer/contracts.py
```

### B50 两组

- `runtime.build(config, device)`：可以复用，构建的就是原 B。
- `runtime.load_dcp(student, checkpoint)`：可读取本文 B50 checkpoint；
  调用前自行按本交接 JSON 核对 step、teacher 和 manifest hash。
- `prefix_inputs` / `sample`：沿用原合同，不要在换权重时改采样器。
- **旧 `prepare_release --step 20000` 不会导出新 B50！**
  该脚本的源路径写死为 `whenami_timeline_b/...A10k...`。
  请在独立副本中改成本文显式路径，或新增参数化导出入口；不要覆盖旧包。

### C 两组

- 旧 `runtime.build()` 构建的是 B，不是 C。
- 旧 `load_dcp()` 只接受 B 的两个 variant，旧 `replay.py` 也只构建 B。
  **三处都需要 C-aware 的独立适配，不能只放宽 variant 检查。**
- `prefix_inputs()` 和 `sample()` 的计算接口可复用：它们只读模型返回的
  `conditions` 和 `reader_z`，并未把 token 数硬编码成 16；复用后仍需回放验收。
- 源码快照必须新增 `whenami_timeline_c/__init__.py`、`model.py`。
  旧 `prepare_release.snapshot()` 的复制列表不包含 C，需显式补入。
- 导出配置/manifest 必须记录 C 架构、variant、`future_gap=4`、
  32 tokens 和 checkpoint 中的 `read_scale`；不要冒充旧 B 包。

下面是接收方可参考的 **C 冷构建方式**，不是声称已经在 NVIDIA 跑过的新入口：

```python
from whenami_timeline_b_handoff.runtime import build
from whenami_timeline_c.model import TimelineStudent
import torch

# build 不读取 A/Stage1/base-weight checkpoint；下方也不能先跑推理再加载。
b, visual, transform, tokenizer, _ = build(config, device)
student = TimelineStudent(
    b.vlm, b.local_encoder, b.reader,
    variant=variant, future_gap=4,
)
del b
for parameter in student.parameters():
    parameter.data = parameter.data.to(dtype=torch.bfloat16)
student = student.to(device).requires_grad_(False).eval()
student.vlm.vision_backbone_requires_grad = False
prompt = student.vlm.get_prompt_builder
# 下一步：严格加载对应 C 的全部 1549 项；含 FP32 read_scale。
```

`TimelineStudent` 构造会重新创建 Timeline；所以必须加载**完整 C 权重**，
不能只加载 VLM/DiT，也不能认为复用 b 的组件就已经有训练好的 C。

### 完整加载验收

无论 DCP 还是 safetensors：

1. 构建正确的模型类，检查 variant、20k、A50k teacher 身份。
2. 比较完整 key 集合和每个 tensor 的 shape；不允许遗漏/多余项。
3. 目标必须是独立内存，不向 mmap 训练 checkpoint 写入。
4. B 应加载 1550 项，C 应加载 1549 项；加载值全部 finite。
5. C 检查 `read_scale` 实际值为 0/1，并检查 `future_gap==4`。
6. 不把“非严格加载没报错”或“输出 finite”当成加载/数值验收通过。

原 `runtime.load_release` 虽然按分片调用 `strict=False`，
但最后会检查完整 key 覆盖；这种完整覆盖检查必须保留。

## 4. 需要传什么，不需要传什么

**不需要 OXE 全量数据、训练 teacher、Stage1 权重、optimizer 或 RNG。**
需要：

- 四组完整学生权重；
- 模型构建配置和 `dataset_statistics.json`；
- 原推理 Python 依赖源码快照，额外加 C 的模型定义/适配；
- 已有 tokenizer、冻结 DINO/Wan 权重，可复用已验通 B 包中的相同资产；
- 两个域的因果数值 fixtures，以及新模型对应的预期输出；
- 本文与机器可读 JSON。

共享配置/动作统计源路径：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage1/baseline_v2--image_aug/config.json
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage1/baseline_v2--image_aug/dataset_statistics.json
```

这里只取两个 JSON，不取同目录下 Stage1 的模型权重。对应哈希已写入清单。

### 推荐路线：源端导出新 safetensors 包

在独立导出目录中复用原 B 的导出/fixture/往返检查逻辑，按第 3 节补齐 B50 路径和 C 适配。
每个包保留 `MODEL.json`、配置、统计、源码与 `SHA256SUMS`。
同机先执行“直接 DCP → 导出 → 重新加载 → 同输入同噪声回放”，再发送 NVIDIA。
不要覆盖旧 `B10k_20261005_02` 或 `B20k_20261006_01`。

### 备选路线：接收方直接读取原始模型 DCP

仅在接收方 DCP 读取已验通时使用。不需要启动 320 rank，也不调用训练 resume。
每个 checkpoint 的 **整个 `model_dcp/` 加 `COMMITTED.json`** 都要传，
不能只拿 `__0_0.distcp`。

四份原始模型各约 **30.07 GiB**，合计约 **120.28 GiB**，不含共享视觉资产。
这只是未导出的模型分片体积，不是必须搬四份 optimizer。

下面命令在**源服务器**运行，先替换目标地址，并在目标端建好 `.../raw/`：

```bash
python - \
  /home/zhengguantian/WoG_fresh/WHENAMI_FOUR_20K_EVAL_MODELS_20261007.json \
  'your_user@your_host:/your/path/whenami_four20k/raw/' <<'PY'
import json, subprocess, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
destination = sys.argv[2].rstrip("/")
for m in manifest["models"]:
    subprocess.run([
        "rsync", "-a", "--info=progress2",
        "--include=COMMITTED.json", "--include=model_dcp/***", "--exclude=*",
        m["source_checkpoint"].rstrip("/") + "/",
        destination + "/" + m["eval_id"] + "/",
    ], check=True)
PY
```

这是**仅权重原件**的传输模板，不会自动传源码、共享资产、JSON 或生成 C adapter。
传输前生成整包的逐文件 `SHA256SUMS`，接收端校验；本清单的 manifest/metadata
哈希不能代替模型 payload 的完整校验。
若目标 PyTorch 无法读取源 DCP，改走源端 safetensors 导出，不要破坏已工作的仿真环境。
不要复制 PPU 的 Torch 二进制、驱动、通信库和编译缓存到 NVIDIA。

## 5. 四组必须统一的输入、采样与动作协议

- 输入：一张真实历史 + 当前 RGB + 指令；每帧独立 VLM / DINO / Wan。
  不拼接成旧 Stage3 的五帧视频，不加载真实未来。
- 时间填实际观测索引，严格递增；runtime 会减去第一帧时间。
  不把推理调用次数冒充控制步，不把 C 的 future_gap 当 history_gap。
- `env.reset()` 清空历史、动作队列、ensemble、sticky gripper。
  冷启动只有一张真实图时用 K=1；不要伪造历史。四组采用同一冷启动协议。
- 都使用当前 `reader_z`；DiT 必须接模型返回的完整 `conditions`。
- 原生推理：BF16 参数计算，保留 FP32 buffer；
  外部初始噪声/累积状态/时间网格为 FP32；
  `linspace(0,1,10)`，**9 次 Euler 更新**，输出 `[16,7]`。
- 不对整个模型调用 `.bfloat16()` 而误改 FP32 buffer。
- Google 统计域：`fractal20220817_data`；WidowX：`bridge_orig`。
  沿用已经验通的 q01/q99/mask、旋转、动作尺度和机器人夹爪后处理。
- 控制频率、execute_horizon、ensemble、相机、history_gap、
  episode/task 清单、噪声 seed 规则与旧 WoG/B 对照固定一致。
- 不在看到成功率后为不同模型分别挑选最有利的 history_gap 或动作尺度。

**精度提醒：** 本轮训练内 DEV 是 FSDP 推理，原便携包是 native 推理。
之前已观察到 FSDP 显式输入 BF16 转换与 native FP32 外部状态带来的数值差别。
不要为了“凑到下表 MSE”只修改某一个模型的采样精度；
需要新精度协议时，四组和对照都应一致重跑并单独命名。

## 6. 正式 SIMPLER 前的验收清单

1. 按本 JSON 校验四个 checkpoint 身份；完整 key/shape/finite 检查。
2. Bridge/Fractal 各一条**已观测两帧** fixture，同输入同初始噪声回放。
   输入 fixture 可复用旧包；**旧包的预期输出不能作为新模型的 expected 输出**。
3. 记录 `conditions`、`reader_z`、首个 velocity、最终未裁剪动作的 max_abs/RMSE。
   C 额外检查 current/future condition、query_times、read_scale。
4. B 的 `conditions` 形状 `[1,16,32]`；C 为 `[1,32,32]`；
   `reader_z` 为 `[1,1,4096]`，动作 `[1,16,7]`。
5. C-aux 的后 16 tokens 应为 current 的复制；
   C-read 的后 16 tokens 应取预测 future。不得在 wrapper 中重新拼错。
6. 同设备导出/重新加载应按原严格往返协议验收；
   PPU→NVIDIA 不假定 bitwise 相等，但必须解释数值差异。
   回放退出码 0 只表示程序完成，不能自动代表数值验收。
7. 每组在 Google/WidowX 各少量 episode 检查 reset、历史、动作与夹爪，
   再跑既定完整任务清单。
8. 每卡模型/渲染峰值显存实测；本次没有新的 NVIDIA 显存测量。
   复用已验通的多进程或模型切分策略，不把多卡显存当成自动共享。
   WanVAE 的内部 model/scale 不是普通注册子模块，不能依赖外层 `.to()` 自动跨卡搬完。

四组全部需要测。B0/B1 可以先沿用成熟接口推进，C 完成适配/回放后跟上；
不要因为 C 的适配工作未完成，就把旧 B 模型误标为 C 跑出一张表。

## 7. 已有离线结果——仅供溯源，不是 SIMPLER 分数

以下均为最终 **FSDP DEV、27 域、410 episode、3 个采样 seed**：

| 模型 | MSE6 ↓ | 首动作 MSE6 ↓ | Gripper accuracy ↑ |
|---|---:|---:|---:|
| B0_A50k_20k | 0.04454655 | 0.03548272 | 0.94757680 |
| B1_A50k_20k | 0.04510611 | 0.03613531 | 0.94531040 |
| C_aux_A50k_h4_20k | 0.04444165 | 0.03404954 | 0.94241223 |
| C_read_A50k_h4_20k | 0.04443274 | 0.03429826 | 0.94320389 |

- 这不是三次独立训练；不能把三个采样 seed 写成三个训练 seed。
- C-aux/C-read 的整体 MSE 很接近；B1 的离线点估计没有超过 B0。
- 不能据此省略某组 SIMPLER，也不能声称 C-read 已经闭环更好。
- C 与 B 不只是“future 是否读入”一个变量：
  C 改了时间查询、双节点对齐、训练未来间距等。
  **隔离 future read-in 的直接对照是 C-aux vs C-read。**
- 本轮 B vs 旧 A10k→B 反映升级初始化/教师的结果，但增加了 A 预训练预算。
- 原 A50k-causal 的某些记录只有一个采样 seed，另一些来自 native backend；
  不与这张三 seed FSDP 表直接混算增益。

下一步核心问题是：更强起点是否改善实际成功率，以及读入预测 future 相比仅辅助监督
是否有闭环价值。不要仅用离线 MSE 回答。

## 8. 接收 agent 回传格式

建议每个 `eval_id` 独立输出：

```text
<eval_id>/
  model_identity.json       # eval_id/variant/step/teacher_step/源哈希/本地权重哈希
  eval_config.json          # 模拟器commit、任务、seed、历史、采样、动作后处理
  replay/RESULT.json
  episodes.jsonl            # task/robot/seed/episode/success/error/视频路径
  summary.json
  logs/
  videos/                   # 至少保留失败和异常样例
```

汇总分别报告：

- Google VM、Google VA、Google 汇总；不要把 Google 汇总叫作“所有机器人 overall”。
- WidowX grasp 与最终 success，二者不能混同。
- 任务级 episode 数、成功数/率、执行异常数；不能丢掉异常后只报正常样本。
- 同协议 WoG、旧 A10k→B20k 结果（若已具备），以及四模型的配对 episode 明细。
- GPU/Torch/依赖和模拟器 commit、精度/切分/量化、峰值显存、推理耗时。
- `history_gap`、时间单位、cold_start、control_freq、execute_horizon、ensemble、
  统计域、旋转/夹爪转换、全部 adapter 修改与源码哈希。

这轮主任务就是四个 20k。额外 A50k-causal 对照可以后补，不阻塞这四个的交付。

## 9. 与旧交接文档的关系

- `WHENAMI_TIMELINE_B_SIMPLER_HANDOFF_20261005.md`：
  保留详细环境、历史缓存和动作后处理说明，但其 B10k 身份/路径不适用于本轮。
- `WHENAMI_B20K_EVAL_UPDATE_20261006.md`：
  描述旧 **A10k→B20k** 已有便携包，可复用已验通代码/共享资产，不能复用其权重冒充本轮。
- `TIMELINE_B50_A50K_SUBMIT_20261006.md`、`TIMELINE_C_A50K_SUBMIT_20261006.md`：
  训练设计与提交记录；本轮不需要重新提交训练。

本文与附带模型清单定义本次四模型身份；若接收端已有不同文件、不同 step 或新导出包，
先核对源 `COMMITTED.json` 哈希再继续，不凭文件夹名猜测。
