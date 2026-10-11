# WoG Stage I —— 集群提交指令

**日期**:2026-09-21 · 平台:Neolix 训练平台 / PPU-DLC-QUEUE

---

## 一、在平台表单里填什么

对照你截图的字段:

| 字段 | 填什么 |
|---|---|
| **实验名称** | `WoG_stage1` |
| **模型** | `One-Model`(和你 DinoWM 那个一样) |
| **实验类型** | `训练` |
| **资源队列** | `PPU-DLC-QUEUE` |
| **镜像配置** | 自定义镜像 → `ai-prod-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/acr_namespace/ai_image:torchengine-torch280-cu129-py312-v103-tmp5`(和你现在用的同一个,已验证可用) |
| **代码模式** | `本地目录` |
| **训练框架** | `PyTorch` |
| **Worker 实例数** | **2**(先跑通用 2 节点 = 32 卡,正好对齐论文官方配置) |
| **GPU 类型 × 卡数** | `PPU-GP7V` × `16`(平台锁定,不可改) |
| **共享内存** | `1792` GiB(跟随节点内存,保持默认) |
| **自动容错** | 开(已默认开) |
| **Hang 检测** | 开 |

### 执行命令(直接复制这一行)

```bash
bash /home/zhengguantian/WoG_fresh/run_stage1.sh
```

**就这一行。** 分布式参数(`MASTER_ADDR`/`WORLD_SIZE`/`RANK`)由平台注入,脚本自动读。

---

## 二、先跑一个 30 分钟的冒烟测试(强烈建议)

直接上 32 卡跑 100k 步风险太大。先用 **1 节点 16 卡 + 只喂 bridge** 验证能跑通:

**Worker 实例数改成 `1`**,执行命令换成:

```bash
DATA_MIX=bridge_orig RUN_ID=smoke SAVE_INTERVAL=200 bash /home/zhengguantian/WoG_fresh/run_stage1.sh
```

看到 loss 开始下降、`outputs/stage1/` 里出现 checkpoint,就说明链路通了。然后再改成 2 节点 + `oxe_magic_soup_plus` 正式跑。

---

## 三、已经替你准备好的东西(都在 PFS 上,集群直接可读)

| 资产 | 路径 | 状态 |
|---|---|---|
| 训练依赖 | `/mnt/pfs/zhengguantian/WoG_v2/pylibs-train` | ✅ 1.8 GB,已验证 |
| OpenVLA 起点权重 | `.../ckpts/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt` | ✅ 30.1 GB 下完 |
| 官方代码 | `/home/zhengguantian/WoG_fresh/WoG` | ✅ 零改动,HEAD `872d1df` |
| 启动脚本 | `/home/zhengguantian/WoG_fresh/run_stage1.sh` | ✅ 含启动前自检 |
| 数据 | `.../datasets/open-x-embodiment/` | ⏳ 下载中(见第五节) |

### 启动前自检会做什么

脚本开头会检查并在不满足时**直接退出**(避免浪费卡时):

```
torch 2.8.0 | cuda=True | PPU-ZW810E
numpy 1.26.4 (需 1.26.x) | tf 2.16.1 | tfds 4.9.3
transformers 4.40.1 (需 4.40.1) | timm 0.9.10 (需 0.9.10)
OK   openvla 起点 (28.1 GB)
```

---

## 四、这次踩通的 6 个坑(为什么不能直接照 README 跑)

### 坑 1:镜像是 py312,但官方要 TF 2.15(无 py312 wheel)

WoG 官方 `pyproject.toml` 要 `python=3.10 + tensorflow==2.15.0`,而 TF 2.15 最低只到 py3.11。
**实测 TF 2.16.1 可完整替代** —— 真实读出 `action (16,7)` / `image (224,224,3)` / 真实语言指令。
这是能在集群上训练的前提,验证脚本见 `_rlds_probe.py`。

### 坑 2:dlimp 硬钉 `tensorflow==2.15.0`

必须 `pip install --no-deps` 装,否则解析失败。

### 坑 3:protobuf 版本三方拉锯

- protobuf 7.x → tfds 报 `FieldDescriptor has no attribute 'label'`
- protobuf 5.29 + 新版 tensorflow-metadata → `gencode 6.31 vs runtime 5.29` 冲突
- **解**:`protobuf==5.29.5` + `tensorflow-metadata==1.15.0`,两个都要锁

### 坑 4:py3.12 移除了 `distutils`,TF 2.16 仍 `import distutils`

`--target` 安装不触发 setuptools 的 `.pth` 钩子,`sitecustomize.py` 在 PYTHONPATH 里也不自动加载。
**解**:在 `pylibs-train/distutils/__init__.py` 建一个转发包指向 `setuptools._distutils`。

### 坑 5:`torchdiffeq` 会误拉进 torch 2.14+cu130,盖掉 PPU 的 torch 2.8.0

后果:`torch.cuda.is_available()` 变 **False**,训练必然失败且报错难懂。
**解**:装完后删掉 `pylibs-train/{torch,torchvision,torchaudio,functorch,nvidia,triton}`(已处理,省了 5.3 GB)。
以后往 `pylibs-train` 装任何包,**装完都要复查 `torch.cuda.is_available()`**。

### 坑 6:镜像自带 transformers 4.57 / numpy 2.3,和 WoG 要求冲突

WoG 要 `transformers 4.40.1`(4.57 下 `BertModel` 导入即失败)、TF 2.16 要 `numpy 1.26.4`。
**解**:`PYTHONPATH=$PKGS:$CODE`,`pylibs-train` **必须在最前面**。顺序反了就会用到镜像的版本。

---

## 五、数据现状 —— 这是目前唯一的未完成项

Stage I 的 `oxe_magic_soup_plus` 需要 **26 个子数据集**,当前:

```
已有:  fractal20220817_data  112 GB  ✅
       bridge_orig           124 GB  ✅
       kuka                  264 GB  ⏳ 下载中(共 778 GB)
缺:    其余 21 个
```

按论文 Table 10 采样权重算,**已有的只占 28.7%**。

下载在 tmux `oxe_rest` 会话里跑(12 并发,实测 ~113 MB/s,代理封顶):

```bash
tmux attach -t oxe_rest          # 看进度
du -sh /mnt/pfs/zhengguantian/WoG_v2/datasets/open-x-embodiment/*/
```

脚本幂等 + 断点续传,中断了直接重跑 `bash download_oxe_rest.sh`。

### 两个数据问题

1. **`fmb_dataset` 在 GCS 上不存在**(权重 0.0232)。已从下载列表注释掉,需另找源。
2. **`droid` 有 3.4 TB 但权重只有 0.06**,占总量 68%。我已**从下载列表移除** —— 性价比极低。
   代价:混合比例与论文不完全一致。要严格对齐就得补下,单独跑 `bash download_oxe_rest.sh droid`。

### 现在就能跑通(用 `DATA_MIX=bridge_orig`)

代码里 `oxe_magic_soup_plus` 会要求所有 26 个数据集都存在,**缺一个就报错**。所以:
- **现在**:用 `DATA_MIX=bridge_orig` 做冒烟测试,验证训练链路 ✅
- **数据下完后**:换成 `oxe_magic_soup_plus` 正式跑

---

## 六、预期耗时

| 配置 | Stage I 100k 步 |
|---|---|
| 2 节点 / 32 卡(对齐论文) | 约 **4~6 天** |
| 4 节点 / 64 卡 | 约 2~3 天 |
| 8 节点 / 128 卡 | 约 1~2 天(但 global_batch 会涨到 4096,偏离论文,lr 可能要调) |

**建议 2 节点 32 卡** —— 和论文 `expected_world_size=32` / `global_batch_size=1024` 完全一致,
不用调 lr,结果可直接对比。

---

## 七、一句提醒

**你大概率不需要跑完整 Stage I。** 官方 `WoG-V` ckpt(31 GB)已在本地且验证可用,
Stage I 的唯一产物就是那个训练好的 Q-Former。除非你要**改 Stage I 的结构**
(比如用 TSI teacher 替换 Q-Former),否则重跑是纯浪费。

但既然你说"总是要做出这个的",跑通一次确实有价值 —— 至少证明链路完整、
以后改结构时有个可对比的基线。

**更新时间**:2026-09-21
