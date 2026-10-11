# WhenAmI Timeline-B：NVIDIA / SIMPLER 评测交接

日期：2026-10-05。接收对象：已经部署 WoG / 旧 Stage3 的 NVIDIA 评测服务器与负责 agent。

> **2026-10-07 最新评测交接：** 请阅读
> `WHENAMI_FOUR_20K_SIMPLER_HANDOFF_20261007.md` 和配套
> `WHENAMI_FOUR_20K_EVAL_MODELS_20261007.json`：
> 本轮是 A50k→B0/B1、C-aux/C-read 四个完成20k的模型，C需要独立推理适配。
> 下文保留历史 B10k 包记录，不代表本轮四模型已经导出或完成 NVIDIA 验收。

> **2026-10-06 状态更新：** 两组A已完成50k，两组B已完成20k，最终checkpoint和评测均已提交。
> 结果见同目录 `WHENAMI_TIMELINE_AB_FINAL_RESULTS_20261006.md`。
> **下文便携包仍是B10k，未自动升级或替换为B20k。**
> 最终B20k需要另行导出独立包；本次检查没有修改已交付10k包或其checksum。

**这是新 Timeline-B 的交接，不是旧 Stage3 的接口更新说明。**
旧文档 `WHENAMI_NVIDIA_EVAL_HANDOFF.md` 可作环境历史参考，但其四帧历史、R/K、
checkpoint 名称不能用于本次模型。

## 0. 先说结论与验收边界

1. B0/B1 已有完整的 **B-step-010000** checkpoint，可以开始中期闭环评测；
   不必等待 A50k 或 B20k，没有必须补训一个新 Stage 才能推理的要求。
2. 此模型身份为 **Timeline-A10k → B10k**，不是 A50k→B，也不是旧 Stage3-50k。
3. 本次提供独立模型构建/加载、纯因果前向、动作采样和数值回放工具。
   **SIMPLER 的历史缓存、动作后处理和 policy 注册仍由接收方接入。**
4. 原服务器只有 PPU。是否完成本机导出/回放，以包内
   `VERIFICATION.json` 的 `complete=true` 为准；
   **本机检查不等于已经在 NVIDIA 或 SIMPLER 上通过。**
5. 不下载 OXE、不加载训练老师、不搬 optimizer/RNG，也不用另下 LLaMA/OpenVLA 基础权重。
6. 不修改原服务器正在训练的源码、checkpoint、配置及 teacher；只使用本次独立快照。

### 本机已经完成的检查

- B0/B1 的10k权重各1550个状态项完整加载，所有加载值有限。
- 7项CPU序列化/缺失键/精度/校验和等测试通过。
- 两组各在Bridge、Fractal真实已观测前缀上检查，新输入构造与原评测输入逐项一致。
- 两组“DCP直接加载→导出→重新加载”的条件、cognition、首步velocity和最终动作逐项一致。
- 两组K=1冷启动都能产生有限输出；这只是接口测试，不是冷启动成功率证明。
- 本机PPU程序峰值allocated约17.41GiB，**不是3090上的显存测量或容量保证**。
- 额外从便携包目录、全新进程冷构建B1并加载导出权重，Bridge/Fractal全部回放量
  逐项相等；该独立回放峰值allocated约17.05GiB，不依赖Stage1/A老师权重或原项目代码目录。

独立进程结果见 `fresh_replay_condition_align_clean/RESULT.json` 和最终 `VERIFICATION.json`；
NVIDIA与SIMPLER的验收不由以上结果替代。

## 1. 本次交付路径

交接文档：

```text
/home/zhengguantian/WoG_fresh/WHENAMI_TIMELINE_B_SIMPLER_HANDOFF_20261005.md
```

便携推理包：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b/nvidia_handoff/B10k_20261005_02/
```

结构：

```text
B10k_20261005_02/
  VERIFICATION.json
  WORK_STATUS.json
  SHA256SUMS
  code/
    whenami_timeline_b_handoff/     # 新的构建/加载/采样/replay，不是SIMPLER wrapper
    whenami_timeline_b/            # 原学生定义
    whenami_timeline_joint/        # 原Timeline定义
    ...                           # 小型Python依赖快照
    WoG_16rank_probe/
      pretrained/vision/
        dinov2_weights.pth
        Wan2.1_VAE.pth
      prismatic/models/backbones/llm/llama2_config/
  action_only/
    MODEL.json
    config.json
    dataset_statistics.json
    model-*.safetensors
    fixtures/{bridge_orig,fractal20220817_data}.{json,npz}
  condition_align/
    ... 同上 ...
```

两个学生共享一份源码、tokenizer 和冻结视觉权重。
导出使用 BF16 参数、保留原 FP32 buffer，不是 INT8/INT4 量化。
体积以包内 `MODEL.json` 分片字节数和 `du -sh` 为准；无需复制原完整训练目录。

原始 checkpoint 仅用于溯源：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b/training/
  timeline_b_action_only_A10k_320_20k_01/main/checkpoints/step-010000/
  timeline_b_condition_align_A10k_320_20k_01/main/checkpoints/step-010000/
```

每个原 checkpoint 是16份 FP32 DCP 模型分片，加训练恢复状态。
不要把单个 `__0_0.distcp` 当完整模型，不要在 NVIDIA 推理时调用320rank训练恢复入口。
便携包已经去掉 optimizer/RNG，原始 DCP 无需再搬一份。

## 2. 两个模型到底是什么

| 名称 | 学生初始化 | B阶段训练 | 推理输入 |
|---|---|---|---|
| B0 action_only | 同一个固定 Timeline-A10k | 当前动作 FM | 一帧历史＋当前＋指令 |
| B1 condition_align | 同上 | 当前动作 FM＋0.1条件余弦对齐 | 完全相同 |

老师的目标是看过过去、当前、真实未来后形成的**当前中间节点条件**，
不是未来节点/未来图像。老师只出现在训练中。
学生的 `forecast_head` 只是辅助头；其输出不应接到动作 DiT。

```text
实际已观测历史 + 当前RGB + 指令
              │
    每帧独立的VLM、DINO/Wan
              │
      local QFormer + Timeline
              │
       选择最后已观测节点
              │
       conditions, reader_z
              │
  DiT(x_tau, tau, reader_z, per_token=conditions)
              │
        16×7 normalized actions
```

B0/B1 都继承 A 的预训练；二者之差不能被解释为“有未来训练 vs 从未用未来训练”。

## 3. 搬运与校验

在原服务器执行前，替换目标地址；这里没有自动发起跨服务器传输：

```bash
SRC=/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b/nvidia_handoff/B10k_20261005_02
NVIDIA_DEST='your_user@your_nvidia_host:/your/path/whenami_eval/B10k_20261005_02/'

test -f "$SRC/VERIFICATION.json"
test -f "$SRC/SHA256SUMS"
python - "$SRC" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert json.loads((p / "VERIFICATION.json").read_text())["complete"]
assert json.loads((p / "WORK_STATUS.json").read_text())["complete"]
PY
rsync -a --info=progress2 "$SRC/" "$NVIDIA_DEST"
```

在 NVIDIA 服务器：

```bash
export RELEASE=/your/path/whenami_eval/B10k_20261005_02
cd "$RELEASE"
sha256sum -c SHA256SUMS
```

checksum失败先重新传输，不要用不完整文件继续测。包内源码是独立快照，可在接收方
另建 adapter；如果修改快照代码，必须保留原始文件和记录修改，不能继续宣称未修改的原包。

## 4. 环境：优先复用已跑通的 NVIDIA 环境

不要复制 PPU site-packages、驱动、通信库、编译缓存。
不要为本次接入直接运行原 WoG 的完整 `pip install -e .`，
它会强制旧 Torch/TF 训练依赖，可能破坏已工作的仿真环境。

本机导出命令实际生效的核心Python版本（带训练纯Python依赖路径）：

```text
torch             2.8.0  [PPU构建，不是要求复制该二进制]
torchvision       0.23.0
transformers      4.40.1
tokenizers        0.19.1
timm              0.9.10
numpy             1.26.4
safetensors       0.8.0
draccus           0.8.0
torchdiffeq       0.2.5
einops            0.8.2
sentencepiece     0.1.99
```

这是来源环境记录，不是已经在3090验证过的安装锁文件。优先在接收方已有 WoG
推理环境中运行数值回放；有依赖冲突时建独立推理进程，不随意替换仿真依赖。
导出是 safetensors，接收方无需与 PPU 的 DCP 反序列化格式兼容。

原服务器参考仿真版本：

```text
SimplerEnv              06accaca93535902d408da4855f21cece12bceb7
ManiSkill2_real2sim      ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
```

如果接收方已经使用另一固定版本，应回传 commit 并保持 B0/B1/WoG/旧Stage3一致，
不要为了符合这个历史快照而破坏已经验通的部署，也不要混合不同版本结果。

## 5. 第一步：先回放模型，不启动仿真

包内 fixture 只含两个真实已观测图像、指令、时间索引、固定初始噪声与预期中间/最终输出。
不含真实未来图像、不含专家动作输入。Bridge/Fractal样例只用于接口数值验收，不计成功率。

在目标推理环境执行：

```bash
export RELEASE=/your/path/whenami_eval/B10k_20261005_02
cd "$RELEASE"
export PYTHONPATH="$RELEASE/code:$RELEASE/code/WoG_16rank_probe${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1

CUDA_VISIBLE_DEVICES=0 python -m whenami_timeline_b_handoff.replay \
  --release "$RELEASE/action_only" \
  --output /your/path/results/timeline_b0_10k_replay_01

CUDA_VISIBLE_DEVICES=0 python -m whenami_timeline_b_handoff.replay \
  --release "$RELEASE/condition_align" \
  --output /your/path/results/timeline_b1_10k_replay_01
```

每个 output 必须是新目录。
从干净的release根目录启动，不要从含有 `av.py`、`inspect.py` 等同名文件的共享临时目录
启动，否则Python可能把普通脚本误当依赖包导入。本机曾遇到 `/tmp/av.py` 遮蔽PyAV，
已保留失败记录并从干净目录重试，未删除其他项目文件。
回传 `RESULT.json`，其中包含：

- 当前条件、VLM cognition、第一步 velocity、最终归一化动作的 max_abs/RMSE；
- 模型variant/step、Torch版本、GPU型号、峰值显存；
- 输出形状/有限性检查。

PPU与NVIDIA不能假设逐bit相等。`--require-bitwise` 仅用于同平台往返测试，
不要在跨平台首次接入时强制使用。**退出码0仅表示完成有限输出和比较，不等于数值已经验收。**
先检查差异位于输入/条件、VLM还是采样器，尤其注意阈值附近的gripper输出。
数值尚未解释清楚时，不启动成千上万条正式评测。

本机 roundtrip 的含义：直接读B DCP → 推理 → safetensors导出/重新加载 → 同噪声再推理。
它不是NVIDIA cross-device验收，也不是仿真验收。

### 硬件

先在一张空闲卡跑一次replay并测峰值。不要把8×24GB当成自动共享的192GB。
单3090放不下时可复用接收方已有模型切分/推理服务方案；
本文提供的简单runtime默认单设备，**没有宣称实现多卡自动切分或量化**。
保持B0/B1精度和切分策略一致，并记录渲染显存是否与模型同卡。
新helper在构建时显式选择当前CUDA设备。若接收方做多卡切分，需要额外注意：
原 `WanVAE` 是普通Python对象，其内部model/scale并非外层Module的注册子模块，
不能假设一次 `visual.to(other_device)` 已将它全部迁移。

## 6. 接收方需要接入的原生API

```python
from whenami_timeline_b_handoff.runtime import (
    build, load_release, prefix_inputs, sample,
)

# 构建只用配置和本地tokenizer，不读取Stage1/A老师权重。
student, visual, transform, tokenizer, prompt = build(config, device="cuda:0")
load_release(student, model_release_dir)

prefix = prefix_inputs(
    images=observed_rgb_frames,           # uint8 [2,H,W,3]，过去在前、当前在后
    instructions=observed_instructions,   # 每帧对应指令，通常两帧相同
    times=actual_observation_indices,     # [t_past,t_now]，严格递增
    visual=visual, transform=transform, tokenizer=tokenizer,
    prompt=prompt, device="cuda:0",
)
result = sample(student, prefix, initial_noise)  # FP32 [1,16,7]固定/独立Gaussian噪声
normalized_actions = result["normalized_action_unclipped"][0]  # [16,7]
```

本API**不负责**：

- action unnormalization；
- 坐标系、旋转、Google sticky gripper / WidowX绝对夹爪转换；
- 每次实际执行几个动作、temporal ensemble、历史缓存；
- SIMPLER任务/相机选择、policy registry和结果保存。

这些部分应接入已验通的 WoG/旧Stage3 评测工程，而不是在这里另写一套不同协议。

## 7. 最容易接错的输入与采样合同

### 7.1 一帧历史＋当前，不是旧的五帧视频

- 常态输入恰好两个实际观测节点；不使用 `[-15,-11,-7,-3,0]` 旧接口。
- 两张图分别经过独立的 VLM 和冻结 DINO/Wan；**不能拼成两帧/五帧Wan视频**。
- 训练历史间隔按原数据帧索引随机取1–12（短轨迹更短），并非固定gap=4。
- runtime将时间减去第一个观测索引。例如 `[100,104] → [0,4]`。
  不要无论真实间隔多少都传 `[0,1]`。
- 不要用 `WoG.predict_action().generate()` 替代本次 `cognition`：
  当前训练使用直接forward，并按每条有效token长度取最后隐藏状态。
- 输入预处理是RGB→PIL resize(224,224)→checkpoint对应图像transform，
  不额外加入未记录的crop、翻转、BGR转换或训练增强。

### 7.2 历史时间与控制步

缓存应随真实环境观测更新，不只是随“大动作块推理次数”更新。
如果每次推理执行若干控制步，历史gap不能误当成相同数量的推理调用。
需在结果配置中保存 `control_freq`、`execute_horizon`、`history_gap`、
观测时间单位及选择到的实际两帧索引。

接入smoke可先用“每次执行一个动作，历史取上一张真实观测”简化排错；
这只是验收方案，不冒充已经确定的主表最优协议。
正式评测时沿用接收方统一动作执行协议，并在看正式结果前固定history-gap映射。
不得按任务挑最有利gap后只报告赢家。

### 7.3 Episode开始和指令切换

- `env.reset()` 清空历史、动作队列、ensemble和sticky gripper状态。
- 只有当前图时，本runtime允许 `K=1, times=[0]` 的明确冷启动；
  它通过本机有限输出smoke不等于在训练中覆盖了该分布，需单独观察开始几步。
- 不要复制当前图并伪造一张过去图/负时间索引，声称拥有真实历史。
- 有真实过去图后使用K=2；不足目标gap时取最早已观测图并填写实际gap。
- 训练片段内指令相同。若环境改变任务指令，清空跨任务历史并记录该行为。

### 7.4 动作读出和采样

- 必须传 `per_token=current_conditions`；不能让DiT回退到uncondition。
- 不把 `forecast_head` 输出接入DiT，也不加载A老师或未来图。
- 输出为16×7动作块；是否执行1步、若干步或ensemble是外部评测协议。
- 本次固定：FP32初始噪声和累积状态、BF16模型计算；
  `linspace(0,1,10)` 对应 **9次Euler更新**，不是10次更新。
- 两组使用同样的外部噪声seed方案，按task/episode/control-step定位，
  不因worker数或任务运行顺序变化而换seed。
- 不对整个模型直接 `.bfloat16()` 而把原本FP32的buffer也改掉。

## 8. 动作归一化与机器人后处理

使用每个模型包的 `dataset_statistics.json`：

```text
Google Robot: fractal20220817_data
WidowX:       bridge_orig
```

接收方必须显式设置，不默认取字典第一个域。
使用q01/q99和mask，不能换成mean/std或另一个checkpoint统计。

参考已有WoG非CALVIN输出合同：

```python
normalized = np.clip(normalized_actions, -1, 1)
normalized[:, 6] = np.where(normalized[:, 6] < 0.5, 0, 1)
actions = np.where(
    stats["mask"],
    0.5 * (normalized + 1) * (np.asarray(stats["q99"]) - np.asarray(stats["q01"]))
        + np.asarray(stats["q01"]),
    normalized,
)
```

继续使用已验通、同机器人类型的坐标/旋转/gripper适配层。
尤其不能把0/1 gripper直接当作Google相对夹爪控制，也不能重复反归一化或重复反转开闭。
若接收方已有不同的经过验收的动作后处理，请明确回传差异，不静默替换。

## 9. 评测顺序与公平对照

1. checksum、完整模型加载、Bridge/Fractal数值回放；
2. reset和历史索引检查，Google/WidowX各少量episode闭环试跑；
3. 检查相机RGB、控制频率、动作幅度、旋转及夹爪；
4. 再按已有固定任务清单跑 B0/B1 同step；
5. 与同协议 WoG、旧Stage3比较，分别保存原始episode结果、视频与错误。

不要因为B1在某个中间checkpoint离线较好就只测B1。
本次10k两组都是中期checkpoint，可作为接入和探索；
最终20k应按预先固定的协议再评。不能把中期试过的任务挑选为最终测试。

当前B0/B1只能隔离额外teacher-condition监督。
两组都来自同一个Timeline-A10k，不能替代完整的pooled-A→B / timeline-A→B对照。
A未来达到50k不自动更新本轮学生；模型身份始终写清楚。

## 10. 接收方请回传

```text
GPU / driver / torch / torchvision / transformers / timm
SimplerEnv与子模块commit、任务配置hash
MODEL.json中的variant、B step、A teacher step与source hash
replay RESULT.json、主要数值差异与峰值显存
history_gap、时间单位、cold_start、control_freq、execute_horizon、ensemble
归一化域、动作尺度、旋转和gripper转换
任务/seed/episode的成功标记、错误原因、失败视频
推理延迟（区分编码与动作采样）、是否使用量化或多卡切分
```

不要只回传一张平均成功率截图。保留配对episode明细，区分“任务失败”和“执行异常”。

## 11. 尚未完成，不能混称已验收

- NVIDIA上的本次冷构建/加载/数值回放；
- 接收方的新Timeline-B SIMPLER wrapper及历史时间接口；
- 本次B10k的闭环成功率；
- 最终B20k，以及未来A50k→B的训练与评测；
- 多独立训练seed及完整结构因果对照。

本次交接不需要为这些未完成项暂停正在运行的A/B训练。

## 12. 技术依据

原模型：`whenami_timeline_b/model.py`、`whenami_timeline_joint/model.py`。
训练输入/采样：`whenami_timeline_b/data.py`、`evaluate.py`；
token/cognition：`stage3_scale/model.py`。
推理辅助代码：`whenami_timeline_b_handoff/`（独立新增，不在训练依赖中）。

公开接口参考（原始链接，仅作技术来源；具体运行以包内固定源码和目标机验收为准）：

```text
https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
https://github.com/simpler-env/SimplerEnv
```
