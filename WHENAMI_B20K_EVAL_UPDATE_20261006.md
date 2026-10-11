# Timeline-B20k：接收方评测更新

> **2026-10-07 新交接：** A50k→B0/B1 与 C-aux/C-read 四组均已完成20k。
> 本轮四模型请阅读 `WHENAMI_FOUR_20K_SIMPLER_HANDOFF_20261007.md`，
> 并使用 `WHENAMI_FOUR_20K_EVAL_MODELS_20261007.json` 核对身份。
> **本文下方原包仍是 A10k→B20k，未被替换为新四模型。**

2026-10-06。适用于已经用B10k跑出SIMPLER结果的接收方。

## 1. 本次只升级checkpoint，不顺手改变协议

完整便携包：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b/nvidia_handoff/B20k_20261006_01/
```

包含 `action_only/`、`condition_align/` 两组 **A10k→B20k** 学生，
不是A50k→B。B0/B1各1550项状态完整加载，两组Bridge/Fractal前缀检查及
DCP→导出→加载数值回放已通过；以 `VERIFICATION.json` 为机器可读证明。
没有optimizer/RNG，不需要OXE、Stage1或A老师权重。

本轮没有另外执行20k的NVIDIA或SIMPLER。请沿用已经验通的B10k wrapper、
历史间隔、控制频率、执行horizon、相机、归一化/动作后处理、任务清单及seed，
**只替换两组对应权重**。确认旧截图的实际 `MODEL.json.step`，不要仅靠目录名称猜。

## 2. 传输、校验、回放

在原服务器替换目标地址后执行：

```bash
SRC=/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_b/nvidia_handoff/B20k_20261006_01
NVIDIA_DEST='your_user@your_host:/your/path/B20k_20261006_01/'
rsync -a --info=progress2 "$SRC/" "$NVIDIA_DEST"
```

接收方：

```bash
export RELEASE=/your/path/B20k_20261006_01
cd "$RELEASE"
sha256sum -c SHA256SUMS
export PYTHONPATH="$RELEASE/code:$RELEASE/code/WoG_16rank_probe${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python -m whenami_timeline_b_handoff.replay \
  --release "$RELEASE/action_only" --output /your/results/B0_20k_replay_01
CUDA_VISIBLE_DEVICES=0 python -m whenami_timeline_b_handoff.replay \
  --release "$RELEASE/condition_align" --output /your/results/B1_20k_replay_01
```

随后使用接收方已有SIMPLER入口，将checkpoint目录分别换成上述两组。
不要把回放退出码0当成跨设备数值验收通过；查看中间条件、velocity及最终动作误差。
不在共享 `/tmp` 中启动Python，避免 `av.py` 等脚本遮蔽依赖包。

## 3. 必须保留的接口

- 一帧真实历史＋当前；时间索引填写实际间隔。
- 每帧独立VLM及DINO/Wan，不拼接成旧Stage3五帧视频。
- 动作读取当前 `conditions`、`reader_z`；`forecast_head` 仅是训练辅助，不接入动作。
- 16×7输出；Google用`fractal20220817_data`统计，WidowX用`bridge_orig`。
- 不加载真实未来，不调用旧R/K，不在此次换权重时修改动作尺度/夹爪/旋转/采样器。
- 本包原生采样合同与已交付B10k保持一致：外部FP32状态/时间网格、BF16计算，9次Euler更新。

## 4. 输入精度审计备注

本轮新增A-causal接口审计时，严格比对“A10k权重直接推理”和保存的“B0训练前FSDP评测”，
首例发现最大动作差约0.00521。正在核对FSDP根模块的BF16输入转换：
自动混合精度本身不等于FSDP会对forward参数执行的显式转换，尤其是tau。

这不影响从**同一FSDP评测协议**保存的10k/20k预测之间的配对比较。
本次也**没有悄悄改变10k与20k便携包的原生采样合同**。
不能据此宣称已经找到了Google掉分原因；若要测试另一种输入精度，
应把10k和20k都作为单独协议重跑，不能只改20k再归因于训练。

## 5. 回传结果

同时回传B0/B1的step、模型manifest、完整评测配置及逐episode结果。
分别报告Google VM/VA、WidowX Grasp/Success、任务级结果和执行异常。
保留10k结果，不用20k覆盖原始目录。
