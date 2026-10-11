# 现在可以提交：完整语料准备 + 两节点联合训练验收

> **最新提交入口请看 `OXE27_320_SUBMIT.md`，不要继续使用下述旧两域入口做OXE主训练。**
> 旧两域数据准备已完成并保留；旧32卡tiny检查在optimizer.param_groups恢复处失败。
> 修正后的v2 checkpoint和27域读取需使用新目录重新验收，主训练选择320卡。

> **数据范围更正（2026-10-02）：以下A和主训练命令目前仅支持Bridge/Fractal，
> 不是此前`oxe_magic_soup_plus`的27域主训练。任务A先不要当作OXE准备提交；
> B仍可作多机工程验收，但不能验证另外25域的数据语义。
> 27域入口尚未实现，详见 `OXE_SCOPE_CORRECTION.md`。**

2026-10-02。用户负责平台提交；本机验证已完成，没有替用户启动集群任务。
原有20k缓存任务和checkpoint不修改。

## 任务A：完整原始训练语料打包

**建议1节点，16个CPU进程**。这是CPU/I/O工作，不需要GPU计算；
若平台只提供GPU节点，可用1节点16卡，但卡不会用于模型运算，不必申请320卡。
两节点验收任务B可以同时执行，不必等待A。

```bash
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_joint_prepare.sh
```

扫描Bridge/Fractal全部2048个source train TFRecord文件，排除固定dev/test角色，
保存train压缩原图/动作/指令与所有完整窗口索引；不提冻结H/z。
按分片完成记录可安全重跑，已完成分片校验后跳过。
完成标志：
`/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/raw_full_01/train_index.json`
内`complete=true`。

## 任务B：2节点×16卡=32卡，真实大模型验收

```bash
NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_joint_admission.sh
```

平台需给每节点正确的WORLD_SIZE(节点数)、RANK(节点序号)、MASTER_ADDR。
这个入口要求2节点，不是当前16个独立小实验。
使用已准备的真实原始分片smoke，先测试多设备FSDP/DCP，再依次测试
latent/forecast/distill_future三种真实80.7亿参数学生：
每种训练2步 → 保存分片模型/优化器 → 全新进程resume到4步 → 开发动作采样。
实际梯度必须到达VLM视觉/语言、Past、R/K、DiT。

完成标志：
`/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/admission_32_01/ADMISSION_COMPLETE.json`

这是工程验收，不是再次做小缓存效果筛选。若失败，保留日志/分片，
不要反复覆盖ADMISSION_ROOT，也不要跳过检查直接长训。

## 两项完成后：主训练入口已写好，但现阶段受验收门控

建议 **16节点×16卡=256卡**：micro2×累积2 → global batch1024。
默认50k，原学生参数LR2e-5，R/K LR1e-4，1000warmup，
每2500步保存+开发动作评测。

候选主任务：
```bash
VARIANT=forecast RUN_ID=joint_forecast_50k_01 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_joint_train.sh
```

同预算对照（另外的任务，不能同路径）：
```bash
VARIANT=latent RUN_ID=joint_latent_50k_01 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_joint_train.sh
```

可选冻结完整Future教师监督：
```bash
VARIANT=distill_future RUN_ID=joint_distill_future_50k_01 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_joint_train.sh
```

**现在先提交A/B，不要把上面主训练命令误认为已验证可直接长跑。**
主脚本拒绝缺少验收标记、源码变化或未完成的全数据索引，不提供绕过开关。
验收完成后按真实显存/吞吐确定最终microbatch及主训练资源；不再以缓存20k结果
作为工程启动前置。全量原始数据训练的科学有效性仍待实际对照。
