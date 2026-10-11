# Timeline-A：代码与真实smoke验收

完成时间：2026-10-04 20:00 UTC附近。

代码：`/home/zhengguantian/WoG_fresh/whenami_timeline_a/`

## 实现了什么

```text
真实时间锚点图像
      ↓
逐帧冻结DINO/Wan（没有整段视频编码后再切片）
      ↓
共享、可训练的局部QFormer
      ↓
  ┌──────── pooled：聚合可见节点
  ├──────── independent：只读自身节点
  └──────── timeline：读取分开的可见节点＋相对时间
      ↓
每个时间锚点的动作条件
      ↓
同一个完整可训练的DiT
      ↓
每个锚点的真实动作FM监督，尾部padding有mask
```

三组均从已有Future Stage1的同一QFormer/DiT权重开始；
VLM、DINO、Wan暂时冻结。每组521,146,151个requires_grad参数，
原DiT不是冻结reader，只训练小残差的旧方案没有被复用。

这次实现的是**动作监督塑造的条件时间线A**，不声称实现了物理状态、
动作条件动力学或逆动力学。真实动作仅作为标签，不送入条件编码器。
B阶段在线学生尚未实现，teacher训练时允许读取后续真实观测。

## 掩码不是只遮一条路径

- 隐藏节点在局部QFormer之前屏蔽；
- 对应图像产生的VLM特征在证据路径和动作reader路径同时屏蔽；
- 隐藏节点使用可训练的缺失cognition token，不通过当前z偷看图像；
- 每个目标节点有自己的证据截止时间；
- 每层跨时间注意力都读取局部、固定证据，而不是先被未来污染的其他目标节点；
- 全隐藏时有null evidence，避免空attention导致NaN。

## 已实际通过的验收

| 检查 | 结果 |
|---|---|
| CPU单元/真实双进程DDP/保存恢复测试 | 26项通过 |
| 真实OXE域覆盖 | 27域、81个实际时间锚点均编码并采样评分 |
| 三组真实共享DiT训练 | 每组8次更新，同样的16域训练样本对 |
| 参数规模与新模块初始化 | 三组匹配 |
| action loss到局部QFormer、原DiT、时间输入/attention/output的梯度 | 均实测非零 |
| 修改被隐藏节点原始图像 | 三组条件和reader cognition不变 |
| 修改未来节点特征 | 已保存真实模型的较早条件与纯噪声动作采样不变 |
| 零时间残差 | 真实模型恢复对应局部条件 |
| 连续4步 vs 第2步保存、新进程恢复到4步 | 第3、4步全部566个模型state条目、563组Adam参数状态hash逐项一致 |
| 原Future checkpoint及旧Stage3源码 | 未更改，来源校验通过 |

恢复测试使用真实原图流式训练runner；不是只恢复一个小toy网络。
独立连续/恢复进程加载后，真正重新从OXE原图编码并执行更新。

## 不能把smoke当成效果结论

27条TRAIN episode构成固定验收集，8次更新不是正式训练。
记录的纯噪声动作误差仅用于暴露实现问题，**不是泛化指标**：

| 组别 | 初始masked-fixture MSE6 | 8步后masked-fixture MSE6 |
|---|---:|---:|
| pooled | 0.10882 | 0.10823 |
| independent | 0.14092 | 0.12906 |
| timeline | 0.14092 | 0.12723 |

不能据此宣布timeline优于独立节点或聚合方案。尤其pooled的初始条件就是
可见节点的聚合，而另外两组隐藏节点起始为缺失表示；
同权重不代表相同初始输出。该区别已在计划和代码中明确。

单帧局部DINO/Wan输入也不同于原Future端点的多帧输入。这样做优先确保
节点证据隔离；不是已经证明单帧局部编码是最终最优选择。

## 主要文件

```text
whenami_timeline_a/
  PLAN.md
  README.md
  model.py           # 三组条件器、明确可见性、共享局部编码/动作头
  backbone.py        # 现有Stage1权重复用与逐帧编码
  data.py            # 原OXE TRAIN轨迹、实际锚点、标签隔离
  objectives.py      # 逐节点动作FM与纯噪声采样
  train.py           # 原图流式训练入口
  checkpoint.py      # model/Adam/RNG原子保存恢复
  smoke.py           # 真实27域三组验收
  test_model.py      # 单元与CPU DDP测试
  LOCAL_VERIFICATION.json
```

产物：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_a/
  LOCAL_VERIFICATION.json
  SOURCE_REUSE.json
  smoke_real27_01/RESULT.json
  SAVE_RESUME_AND_REAL_CUTOFF.json
  raw_resumed4_01/checkpoints/step-000004/
  source_snapshot/
```

参考了本地WoG原模块，另clone并阅读官方I-JEPA的mask/context/predictor实现；
没有复制或运行其训练代码。参考来源、commit、license和阅读文件hash均有记录。

仅删除了本轮自己的中间/重复smoke checkpoint约11.17GiB；
保留最终可加载的4步checkpoint、所有日志、逐tensor比较hash、RNG与提交元数据。
清理记录在`cleanup_01/MANIFEST.json`。未删除任何历史实验或生产模型。

## 下一步

现在可以进入**匹配三组的正式teacher可行性实验**准备，但需先固定独立开发集、
训练预算和masked/unmasked评分，避免以本次smoke数字选结构。
代码具备DDP路径且CPU双rank已测；**真实多节点PPU尚未验收**，
不能把单PPU成功说成320卡训练已就绪。
教师结构显示收益后，应尽早做B阶段撤除未来的验收，不能只追求特权教师分数。
