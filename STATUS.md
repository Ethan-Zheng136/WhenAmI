# WoG_fresh:全新 clone 跑通记录

**日期**:2026-09-20 · 目录 `/home/zhengguantian/WoG_fresh/`

## 缘起

怀疑 `WoG_v2/refs/code/WoG` 被改动过,重新 clone 一份对照。

## 结论 1:官方代码从未被改动

```
$ cd WoG_v2/refs/code/WoG && git status --porcelain | wc -l
0
$ git log -1 --format="%h %ci"
872d1df 2026-04-28 14:14:41 +0800
```

全新 clone 的 HEAD 同样是 `872d1df`(上游自 2026-04-28 起无更新),
且与现有 repo **逐字节 diff = 0 条**(排除 `.git`/`__pycache__`/`pretrained` symlink):

```
diff -r --brief WoG_fresh/WoG WoG_v2/refs/code/WoG   →  0 条差异
```

其余 7 个参考仓库(DensePolicy / SimplerEnv / Wan2.1 / dinov2 / openpi /
openvla / prismatic-vlms)也全部 `0 changed`。

**"被改得面目全非"的印象来源**:`WoG_v2/` 本身不是 clone,是手工搭的工作区
(`pylibs/`、`pylibs-extra/` 72 个包、`exp/` 19 个脚本、`docs/` 23 份文档、
2 个 venv)。官方代码原封不动在 `refs/code/WoG/` 子目录里。

## 结论 2:全新 clone 可独立跑通推理

```
参数量      9.01 B
推理耗时    3.88 s   (DiT-L, flow)
action      (16, 7)  ✓ 与 Ta=16 一致
range       [-0.075, 1.000]
gripper 维  {0, 1}   ✓ 正确二值化
峰值显存    34.8 GB  (fp32, 96GB 卡余量充足)
```

复现步骤(仅 3 步,**不改任何官方代码**):

```bash
git clone https://github.com/Selen-Suyue/WoG.git
# 1) symlink 权重(官方代码硬编码相对路径 ../../../../pretrained/)
mkdir -p WoG/pretrained
ln -sfn /mnt/pfs/zhengguantian/WoG_v2/ckpts/vision WoG/pretrained/vision
ln -sfn /mnt/pfs/zhengguantian/WoG_v2/ckpts/WoG-V  WoG/pretrained/WoG-V
ln -sfn /mnt/pfs/zhengguantian/WoG_v2/ckpts/WoG-A  WoG/pretrained/WoG-A
# 2) 补 llama2_config(在 .gitignore 里,不随仓库分发)
cp -r WoG_v2/refs/code/WoG/prismatic/models/backbones/llm/llama2_config \
      WoG/prismatic/models/backbones/llm/
# 3) source env_fresh.sh && wog_check
```

## 踩到的 2 个坑(都是**环境**问题,不是代码问题)

### 坑 1:transformers 版本

系统 python 是 `transformers 4.57.0` → `from transformers import BertModel` 直接
`ModuleNotFoundError: Could not import module 'BertModel'`
(4.57 里 `GenerationMixin` 的 lazy import 链对 `vla/qformer.py:4` 失效)。

官方 `pyproject.toml:49` 要求 `transformers==4.40.1`,`.venv-wog` 里正好是。
**必须用 `.venv-wog/bin/python`,不能用系统 python。**

### 坑 2:PYTHONPATH 注入的 timm 1.0.11(这个坑更隐蔽)

`WoG_v2/env.sh` 会 `export PYTHONPATH=$WOG_CODE/pylibs`,而 `pylibs/` 里装的是
**timm 1.0.11**,它会盖掉 venv 里的 timm 0.9.10。

timm 1.x 的 `forward_features` 返回 **list** 而非 tensor,导致
[dinosiglip_vit.py:151](../WoG/prismatic/models/backbones/vision/dinosiglip_vit.py#L151)
的 `torch.cat([dino_patches, siglip_patches], dim=2)` 报:

```
TypeError: expected Tensor as element 0 in argument 0, but got list
```

官方 `pyproject.toml:44` 要求 `timm==0.9.10`。
**解法**:`env_fresh.sh` 里 `unset PYTHONPATH`,让 venv 的 0.9.10 生效。

> 注意这个坑只在跑官方代码时触发。`exp/` 下那些自己写的实验脚本可能依赖
> `pylibs-extra` 里的包(concept_erasure / dadapy / jax 等),那些场景仍需 PYTHONPATH。
> 两套环境用途不同,不要混用。

## 资产盘点(均已完整)

| 资产 | 大小 | 状态 |
|---|---|---|
| `ckpts/WoG-V` (Stage I) | 31 GB | ✓ 完整 |
| `ckpts/WoG-A` (Stage II) | 30 GB | ✓ 完整 |
| `ckpts/vision` (DINOv2 + Wan VAE) | 1.7 GB | ✓ 已校验前向 |
| `datasets/.../fractal20220817_data` | 112 GB | ✓ 1024/1024 shard |
| `datasets/.../bridge_orig` | 124 GB | ✓ 1024 train + 128 val |

无 0 字节/残缺文件。`dataset_info.json` / `features.json` 齐全,
标准 TFDS 布局,`prismatic/vla/datasets/rlds` 可直接读。

**未下载**:`oxe_magic_soup_plus_minus` 其余 25 个数据集(kuka 778GB 等)。

## 仍然不能做的

| | 状态 |
|---|---|
| 推理 / `O^c` 提取 | ✅ |
| SIMPLER 视觉 rollout | ❌ PPU 无 Vulkan(仅 llvmpipe CPU 软渲染) |
| 训练 | ⚠️ 需真 tensorflow 2.15 + dlimp,TF 2.15 只到 cp311 |

SIMPLER 那条见 `project-ppu-no-vulkan`,换 NVIDIA 卡可解。

**更新时间**:2026-09-20
