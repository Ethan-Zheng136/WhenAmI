# WhenAmI：NVIDIA服务器评测环境交接

> **2026-10-05 新模型提醒：** 本文主体是旧 Stage3 的部署协议。
> 新的 Timeline-A→B 学生请使用同目录
> `WHENAMI_TIMELINE_B_SIMPLER_HANDOFF_20261005.md`。
> 新 B 使用“一帧历史＋当前”，没有旧 R/K 推理路径；不得套用本文的四帧历史接口。

更新日期：**2026-10-03**  
接收对象：负责新服务器环境、模型推理部署、SIMPLER闭环评测的工程人员/助手。  
硬件计划：**8 × RTX 3090，每卡24GB**。  
本文区分“已完成”“可先准备”“尚需实现”，不能把准备清单当成已经跑通。

**2026-10-04更新：** 主训练已完成50k，最终checkpoint已提交。
昨日35k的27域离线对比已完成并复核，见`WHENAMI_OXE27_RESULTS_20261004.md`；
50k同协议复测已另行启动，尚未完成，不能混用两个步数的结果。

**2026-10-03历史更新：** 主训练已约36.8k，完整35k checkpoint可用于接入。
请同时阅读 `WHENAMI_STAGE3_WOG_COMPARE_20261003.md`，其中更新了checkpoint定位、
WoG-A/Past/Stage3对照矩阵、实际采样指标及迁移验收。下文20k是历史快照；
独立推理导出是否已由接收方完成仍需其回传验收，不能由本文推断。

---

## 0. 请先读这几条

1. **不需要重新下载全量OXE，也不需要复制训练预处理产物。**
2. 新服务器首先用于**推理和仿真评测**，不是重新训练Stage3。
3. Stage3训练结束即得到可部署策略；工程流程上对应WoG Stage II之后的评测节点。
   **没有必须再训练一个Stage4才能评测的要求。**
4. 现有20k checkpoint也能作为接入和诊断对象，不必等待50k才开发评测。
5. **WhenAmI独立推理导出、跨服务器加载及SIMPLER历史策略wrapper尚未完成。**
   目前不是“复制模型后现成脚本就能得到主表”。
6. 原PPU训练继续保持不动。不要修改共享的训练源码、数据、checkpoint和配方。
7. 不复制PPU的Python site-packages、CUDA替代运行库或整个环境到NVIDIA机器。
8. 新机器先验收“仿真能渲染”，再验收“模型能推理”，最后连接完整闭环。
9. 本文中的环境安装是按本地固定仓库给出的准备流程；**尚未在目标3090机器验证**。

## 1. 当前项目是什么

### 1.1 阶段命名

| 阶段 | 作用 | 当前状态 |
|---|---|---|
| Stage1 / Future | 用真实未来观测建立动作相关的Future条件 | 已有训练权重 |
| Stage2 / Past | 用历史和当前观测建立可部署的Past策略 | 已有累计20k权重 |
| Stage3 | 预测Future、修正Past条件，联合训练动作策略 | OXE27全量训练中 |
| 部署/评测 | 导出模型、历史缓存、环境执行、成功率统计 | 本次交接要完成 |
| 下游SFT | 按目标任务/机器人协议适配 | 不是所有评测的前置条件 |

### 1.2 部署时的信息流

```text
历史图像[-15,-11,-7,-3] + 当前图像
    → 冻结DINO/Wan历史编码
    → Past QFormer → H_P、C_P

当前图像 + 指令 → VLM → z

H_P、z → R → 预测Future条件
H_P、z、预测Future条件 → K → Past条件修正量
C_P + 修正量、z → 动作DiT → 16步动作块
```

训练才使用真实Future作为辅助目标；**正式推理不读取未来图像或专家未来动作**。
只有一个动作读出头，不是两个动作头投票。
当前没有独立backward/cycle损失，不能把它写成完整双向时间线模型。

### 1.3 最近已核查的结果

截至2026-10-03 05:50 UTC附近，主训练约21k步，最新完整可用checkpoint为20k。
20k开发动作评测：Bridge/Fractal各1024窗口，归一化MSE分别约
`0.03560 / 0.04680`。这是离线动作误差，**不是仿真成功率**。

这些只是交接时快照，正式执行时检查checkpoint的`COMMITTED.json`，
不要假设文档中的步数永远是最新步数。

## 2. 原服务器路径：仅用于定位和打包

以下路径属于原服务器。**不要假设目标机器具有相同用户名或挂载路径。**

```text
项目代码：
/home/zhengguantian/WoG_fresh/

原始模块参考：
/home/zhengguantian/WoG_fresh/WoG_16rank_probe/

当前训练器：
/home/zhengguantian/WoG_fresh/stage3_perf/

当前模型及数据协议：
/home/zhengguantian/WoG_fresh/stage3_scale/

当前运行目录：
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/
  oxe27_forecast_zero2_320_50k_save5k_01/

20k checkpoint：
上述运行目录/checkpoints/step-020000/

动作开发评测：
上述运行目录/development/step-020000/results.json

仿真仓库：
/home/zhengguantian/WoG_v2/refs/code/SimplerEnv/

WoG基线参考：
/home/zhengguantian/WoG_v2/refs/code/WoG/

冻结历史视觉权重：
/mnt/pfs/zhengguantian/WoG_v2/ckpts/vision/dinov2_weights.pth
/mnt/pfs/zhengguantian/WoG_v2/ckpts/vision/Wan2.1_VAE.pth

WoG-A基线：
/mnt/pfs/zhengguantian/WoG_v2/ckpts/WoG-A/

LLaMA配置和tokenizer：
/home/zhengguantian/WoG_fresh/WoG_16rank_probe/
  prismatic/models/backbones/llm/llama2_config/
```

主训练数据是OXE的`oxe_magic_soup_plus` **27域混合**，不是所有公开OXE数据。
当前主训练周期为5000步保存、2500步动作评测，另有一次100步恢复验收。
训练使用的DCP checkpoint含Adam/RNG，**不应直接用训练恢复接口在3090上加载**。

## 3. 新服务器究竟需要哪些文件

### 3.1 闭环评测必需

| 文件/资源 | 来源 | 说明 |
|---|---|---|
| SimplerEnv及ManiSkill2_real2sim子模块 | 按第5节固定版本获取 | 含现有任务、机器人、场景和物体资源 |
| WhenAmI独立推理代码 | 原项目整理后传输 | 尚需实现，不能直接使用旧WoG wrapper |
| 一个指定Stage3推理checkpoint | 从原服务器完整checkpoint导出 | 只保留推理状态，不带Adam |
| 模型结构配置和tokenizer | 随推理包 | 不要额外下载整套LLaMA/OpenVLA基础权重 |
| 动作归一化统计及输入协议 | 随推理包 | Bridge、Fractal等域不能误用统计 |
| 冻结历史DINO/Wan权重 | 从原服务器传输 | 当前两个文件合计约1.7GB |
| WoG-A及其推理代码/配置 | 基线需要时传输 | 不是下载全部基线模型库 |

VLM内可训练视觉权重、Past、R/K、动作DiT应在学生导出包内。
仅部署forecast学生时，不需要独立加载训练用Future QFormer或完整oracle教师。
历史DINO/Wan仍需要，不要误以为“删掉Future分支”就能删掉历史视觉编码。

### 3.2 不需要

- 原始全量OXE TFRecord；
- A生成的约1.92TB训练分片及313MB全训练索引；
- 全部5k/10k/15k/20k等checkpoint；
- optimizer、scheduler、每rank RNG等训练恢复状态；
- 旧缓存实验的大量H/z/C特征；
- RT-1、Octo、CALVIN等非本轮指定基线/任务的全量资源；
- PPU运行时、PPU编译缓存或原服务器整个Python环境。

### 3.3 如果还需要复现离线动作MSE

另外提供固定开发子集即可：
图像、语言、动作标签、episode/timestep/样本ID、评测清单。
原来的`train_dev/`约27GB，混有训练窗口与旧特征缓存；
**应先裁成只含评测窗口的小包，不要直接整目录传输。**

闭环SIMPLER评测本身不需要这份专家动作数据。
只有之后专门做某任务SFT，才另外准备该任务的演示数据。

## 4. 目标服务器建议布局和硬件检查

建议使用本机自己的目录，例如：

```text
~/whenami_eval/
  repos/SimplerEnv/
  release/whenami/        # 之后传入的独立推理包
  release/wog_a/          # 可选基线
  data/offline_dev/       # 可选离线子集
  results/
  diagnostics/
```

预留空间要覆盖模型、依赖缓存、评测视频和日志。可先规划约200GB工作空间，
这是工程余量建议，不是下载包大小承诺；视频大规模跑时另设容量上限。

收集以下信息，不要为准备环境直接升级/覆盖服务器驱动：

```bash
mkdir -p ~/whenami_eval/diagnostics
nvidia-smi > ~/whenami_eval/diagnostics/nvidia-smi.txt
nvidia-smi -L
uname -a
cat /etc/os-release
df -h
free -h
```

需要同时验证：
- GPU计算可见；
- SAPIEN所需Vulkan图形驱动和headless渲染可用。

`nvidia-smi`成功不等于渲染已成功。
若系统已经提供`vulkaninfo`，可记录：

```bash
vulkaninfo --summary > ~/whenami_eval/diagnostics/vulkan.txt 2>&1
```

没有该命令时联系机器管理员安装Vulkan诊断工具，不要随意替换驱动。
容器中需正确暴露GPU图形能力，不能只挂载计算库。

### 4.1 8张3090如何用

优先目标是先跑通一个策略—环境实例，再并行多个任务/种子。
不能把8×24GB视为无需代码支持的单块192GB显存。

学生约8.07B参数，仅BF16参数估算约15.0GiB，此外还有冻结历史视觉编码、
激活和渲染占用。单24GB卡完整实例是否可用，必须测峰值。

如果单卡不足：
- 将历史视觉编码与VLM/动作策略分放不同卡；
- 或将仿真渲染与策略推理分进程/分卡；
- 必要时采用受控模型切分。

这些部署方式尚需实现。不要一上来量化后就将结果与BF16基线混为同协议。
先得到可复现的原精度基准，再单独评估量化/卸载。

## 5. 可先独立完成：安装并验收仿真

### 5.1 固定现有参考版本

原服务器本地版本：

```text
SimplerEnv:
06accaca93535902d408da4855f21cece12bceb7

ManiSkill2_real2sim子模块:
ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682
```

不升级到ManiSkill3，不混用另一个分支的任务实现；需要升级时必须作为独立协议变更。

在新服务器运行：

```bash
conda create -n whenami_eval python=3.10 -y
conda activate whenami_eval

mkdir -p ~/whenami_eval/repos
cd ~/whenami_eval/repos
git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv.git
cd SimplerEnv
git checkout 06accaca93535902d408da4855f21cece12bceb7
git submodule update --init --recursive

git rev-parse HEAD
git submodule status
```

完整保留子模块中的`data/`和`mani_skill2_real2sim/assets/`。
发现具体资产缺失时按错误定位补齐，不要因此下载整个OXE。

### 5.2 仿真最小安装

固定参考仓库要求SAPIEN `2.2.2`，README建议NumPy `1.24.4`。
下面是准备流程，不是目标服务器已验证的完整锁定环境：

```bash
cd ~/whenami_eval/repos/SimplerEnv
printf 'numpy==1.24.4\n' > ~/whenami_eval/sim-constraints.txt

python -m pip install -c ~/whenami_eval/sim-constraints.txt numpy==1.24.4
python -m pip install -c ~/whenami_eval/sim-constraints.txt -e ./ManiSkill2_real2sim
python -m pip install -c ~/whenami_eval/sim-constraints.txt -e .
python -m pip check
```

某依赖解析失败时保留完整报错，给依赖增加明确的兼容版本后再安装，
不要简单去掉NumPy约束让它悄悄升级至2.x。

**暂不执行旧WoG项目的整个`pip install -e .`依赖集合。**
原依赖表绑定旧Torch并混入TensorFlow训练依赖，评测推理环境需要单独整理。
Torch/torchvision/transformers/timm版本由推理包的兼容性测试确定，
不要求复刻PPU专用Torch构建。

### 5.3 先跑随机动作环境检查：不需要模型权重

下面只证明环境初始化、相机图像和执行接口可用，不测策略效果。
先运行一个Google Robot任务，再运行一个WidowX任务。

```bash
cd ~/whenami_eval/repos/SimplerEnv
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import numpy as np
from pathlib import Path
from PIL import Image
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

output = Path.home() / "whenami_eval/diagnostics"
output.mkdir(parents=True, exist_ok=True)

for task in ("google_robot_pick_coke_can", "widowx_spoon_on_towel"):
    env = simpler_env.make(task)
    try:
        obs, info = env.reset(seed=0)
        image = get_image_from_maniskill2_obs_dict(env, obs)
        assert image.ndim == 3 and image.shape[-1] == 3
        assert np.isfinite(image).all()
        Image.fromarray(image).save(output / f"{task}.png")
        print(task, image.shape, env.get_language_instruction())
        env.action_space.seed(0)
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        print("ENV_SMOKE_OK", task)
    finally:
        env.close()
PY
```

请确认实际GPU进程和图像内容正常；`CUDA_VISIBLE_DEVICES`不能代替对Vulkan设备选择的验证。
失败时保存完整traceback和截图，区分资产缺失、Vulkan、包版本、环境API问题。
不要为了让脚本跑完就改任务参数或忽略异常。

成功后记录：

```bash
python -m pip freeze > ~/whenami_eval/diagnostics/sim-pip-freeze.txt
git rev-parse HEAD > ~/whenami_eval/diagnostics/simpler-commit.txt
git submodule status > ~/whenami_eval/diagnostics/submodules.txt
```

## 6. 原服务器侧待交付：真正可迁移的推理包

**这部分尚未完成，是当前工程关键路径。**

### 6.1 导出要求

从一个已提交的完整Stage3 DCP中导出学生推理状态，不能把某个rank的分片
直接重命名为`model.pt`，也不能只导出R/K而漏掉已更新的VLM/Past/DiT。

推理包应包含：

```text
whenami/
  manifest.json             # 来源run/step、hash、精度、结构、输入/输出协议
  config.json
  model/                    # 完整学生权重，必要时分文件
  tokenizer/
  dataset_statistics.json
  frozen_visual/            # 或清晰指定的独立DINO/Wan文件
  inference/                # 不依赖原机器绝对路径
  requirements-nvidia.txt   # 经目标环境实测后的锁定依赖
  tests/
```

需明确冻结视觉分支的哪些状态不在学生checkpoint内，随包补齐。
包大小在导出后实测，不提前承诺“整个推理包只需15GB”。
不用另下LLaMA/OpenVLA模型全量参数补洞；学生checkpoint已包含它们训练后的状态。
tokenizer及配置必须匹配训练时的词表扩展。

### 6.2 迁移校验

- 对一小批固定观测，用相同噪声种子比对原加载路径与导出加载路径。
- 比较中间H_P/C_P/z/C_F_hat、修正条件、velocity和最终动作。
- 数值精度或硬件不同可有误差，先报告误差再约定容差，不能无条件要求跨硬件逐bit相同。
- 验证推理只需要历史/当前/语言，不依赖OXE目录、训练索引、optimizer或teacher输入。
- 所有文件附SHA256，避免传输不完整。

## 7. SIMPLER策略接口待实现

参考`WoG_16rank_probe/deploy/wog_policy_sim.py`可了解接口形状，
但不能原样复制就称为WhenAmI部署完成。旧wrapper有模型专用配置和接口假设，
必须逐项核对，不能通过随意补属性默认值掩盖不兼容。

需要实现：

```python
reset(task_description)
step(current_image, task_description) -> raw_action, environment_action
```

核心合同：

1. 每个episode重置历史缓存、夹爪状态、动作ensemble状态。
2. 维护真实观测时间序列；用训练对应的`[-15,-11,-7,-3]`历史偏移。
   不能缓存4张图后把它们随意当成这些时间点。
3. 数据帧间隔与环境控制频率需明确映射，尤其动作块执行了多步时如何采样历史。
   所有baseline使用同一执行/观测协议，不能暗改频率提高单个方法成绩。
4. 初始历史不足时按训练合同重复最早可用帧。
5. 使用正确camera、resize、normalization、语言prompt。
6. 模型使用`eval`和受控推理模式；禁止真实Future和专家未来动作进入策略。
7. 16步动作块如何执行：执行几步、何时重规划、是否temporal ensemble，都要固定。
8. Bridge与Google Robot的动作归一化、旋转表达、夹爪语义不能混用。
9. 在模型外单独测试动作转换，排除模型正确但控制接口错位。
10. 记录端到端延迟、峰值显存、seed、失败原因和环境termination信息。

当前动作评测使用9个Euler积分区间；闭环首先沿用并明确记录，不静默改变采样次数。
不保证可以直接把其他论文表中的数字当成我们的同协议baseline。

## 8. 正式评测验收顺序

### P0：环境通

- 两种机器人环境可reset/渲染/step；
- 图像正确，不黑屏，不CPU假渲染；
- 驱动、依赖、仓库版本有记录。

### P1：模型通

- 推理包离线加载；
- 固定输入回放与原实现数值对齐；
- 实测单卡/分卡的显存及延迟。

### P2：闭环通

- 先一个task、一个seed完成完整episode；
- 保存视频和每步动作，人工检查相机、动作方向、夹爪、历史时间轴；
- 到此才叫WhenAmI接入SIMPLER，而不是此前的随机动作smoke。

### P3：主表协议

- 固定任务清单、Visual Matching/Variant Aggregation设置、种子/初始状态、trial数量；
- 同时运行可部署WoG-A、Past和WhenAmI；额外机制对照单独规划；
- 统一控制频率、重规划和动作后处理，报告推理成本；
- 成功率来自实际episode结果，不用离线MSE替代；
- 最终checkpoint的选择规则预先固定，不能逐任务从多个checkpoint挑最好拼表；
- 部署调试、开发调参集和最终测试分开；
- 记录跨训练种子/任务/试验不确定性，避免把同一轨迹每帧当独立样本。

现有封存离线test只保证这轮Stage3划分隔离，不自动保证基础预训练从未见过这些轨迹；
论文需要披露这一限制。不得在部署调试期间随意打开封存test。

## 9. 交接责任和当前阻塞项

### 新服务器负责人现在可做

- 提供系统/驱动/GPU拓扑/磁盘信息；
- 固定版本安装仿真，完成P0；
- 回传环境日志、两张截图、pip freeze；
- 预留模型目录和结果目录，确认与原服务器的文件传输方式。

### 原项目负责人需要完成

- DCP → 独立推理包导出；
- 迁移后数值校验；
- 无绝对路径的WhenAmI推理加载器；
- 历史策略SIMPLER wrapper；
- NVIDIA依赖锁定；
- 正式任务/seed/初始状态/采样/动作协议清单；
- 原WoG/Past同协议baseline接入。

**当前没有任何理由要求新服务器先下载几TB的OXE来解这个阻塞。**

## 10. 请接收方回传这份状态模板

```text
服务器OS：
GPU及驱动：
空闲磁盘/内存：
是否容器：
SimplerEnv commit：
ManiSkill2_real2sim commit：
Vulkan检查：
Google Robot随机动作smoke：
WidowX随机动作smoke：
截图/日志位置：
模型包是否已收到：
模型固定输入推理校验：
单实例峰值显存/推理延迟：
第一个完整闭环episode：
当前阻塞及完整报错：
```

## 11. 参考资料

上述仿真准备命令依据原服务器固定commit的README/setup/requirements，
不是未锁定的“最新版”。可查阅的项目地址：

```text
https://github.com/simpler-env/SimplerEnv
https://github.com/simpler-env/ManiSkill2_real2sim
https://github.com/Selen-Suyue/WoG
```

相关本地记录：

```text
WoG_fresh/STAGE3_EFFECTS_20261003.md
WoG_fresh/STAGE3_LIVE_AUDIT_20261002.md
WoG_fresh/STAGE3_ZERO2_320_SUBMIT.md
WoG_fresh/stage3/READER_STUDY_RESULTS.md
WoG_fresh/evaluation/README.md
```

旧design文件的“尚未训练/尚未提交”等段落是历史记录，不能覆盖本文已注明日期的状态。
正式数值以对应run的recipe、checkpoint和results文件为准。
