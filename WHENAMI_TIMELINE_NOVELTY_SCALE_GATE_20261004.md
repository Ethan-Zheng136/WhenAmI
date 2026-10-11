# Timeline-A：新颖性复核与320卡大训练提交条件

检查时间：2026-10-04 23:56–23:59 UTC附近。用户表示已开始提交32卡实验。
此时本机尚未看到预期`outputs/whenami_timeline_a/training/`目录，
所以没有证据确认任务进入训练；可能在排队/启动/入口preflight或使用另一输出目录。
需要任务ID或启动日志定位，不建议重复提交。

## 1. 新颖性结论必须收紧

**当前实现是研究原型，不能确认已具有充分新颖性。**

本轮补查了两篇与“masked timeline”比普通belief更直接的近邻：

1. **Masked Trajectory Models for Prediction, Representation, and Control**
   （MTM, ICML2023, PMLR202）。
   已下载原论文并阅读摘要、§3.1–3.2。
   将轨迹中状态、动作、return等模态嵌入为带时间/模态标记的序列，
   随机遮挡并用双向Transformer重建；不同mask可支持前向动态、
   逆向动态、模仿/控制等任务。
2. **Masked Autoencoding for Scalable and Generalizable Decision Making**
   （MaskDP, NeurIPS2022, arXiv:2211.12740）。
   已核对会议/arXiv摘要及公开方法说明：随机遮挡状态/动作token，
   重建缺失部分并用于BC/RL。会议PDF本地下载超时，只保留未完成文件，
   不宣称全文阅读；网络原论文可访问文本已核查。

来源：

```text
https://proceedings.mlr.press/v202/wu23o.html
https://proceedings.mlr.press/v202/wu23o/wu23o.pdf
https://arxiv.org/abs/2211.12740
https://papers.nips.cc/paper_files/paper/2022/hash/51fda94414996902ddaaa35561b97294-Abstract-Conference.html
```

因此“多时间点＋遮挡＋双向attention＋动作任务”不能当作首创。
此前已查TD-VAE、HiF-VLA、RB-VLA、ChainVLA等结论不变。
这轮不是将其与我们的实现判定为完全同一系统，而是明确已有基础机制的范围。

## 2. 值得检验、但尚未成立的贡献候选

在WoG“动作监督塑造future condition→撤除future特权”的学习范式下，
比较**整体条件**与**按时间寻址、共享动作读出塑造的condition timeline**，
检验后者是否带来可转化为在线控制的结构收益。

不同点不能只靠措辞，必须排除：

- 仅仅多了时间查询/额外参数/监督位置；
- 仅仅因为聚合baseline被迫所有节点输出同一个条件；
- 仅仅因为independent被剥夺了上下文；
- 仅仅特权教师更强，而在线学生不受益。

现有pooled和independent是可行性控制，尚非完整的新颖性对照。
尤其pooled在相同证据下各目标共用条件，后续需有同容量、同时间查询、
同可见信息的强聚合/上下文对照。
不能在用户已提交的代码中悄悄改这个控制；新增于独立版本。

## 3. 当前32卡验收的准确地位

- 只确认真实多节点的梯度、DDP、保存恢复、开发评测和吞吐。
- 100步保存，新进程到120步。
- 不要求它已经显示科学优越性，也不依据120步分数挑结构。
- 验收成功不等于“新颖性确认”。
- 不需要等冻结VLM的10k跑完，也不需要等待旧Stage3 latent对照，
  才允许准备更充分的timeline训练。

## 4. “大训练”明确指全VLM联合Timeline-A，不是只加卡

候选训练范围：

```text
VLM视觉/投影/语言主干 + 局部QFormer + 时间关联模块 + 完整DiT：训练
独立DINO/Wan基础编码：冻结
```

它仍是A阶段的特权教师；B阶段在线转化另行实现，不能漏掉。
共同初始化仍取同一Future Stage1端点，不是两组使用不同模型。

当前实现中Foundation.encode是no_grad，并将图像相关z作为冻结结果传出。
所以**设置requires_grad=True或把2节点改成20节点并不构成全模型训练**。
需要独立联合forward、正确保留VLM计算图、分片优化器和对应保存恢复入口。
这部分尚未实现/验收，当前不能交付“已验证可跑”的320卡命令。

## 5. 明确提交门槛，而非无限等待

并行准备320卡联合训练入口和新颖性强对照方案。提交大训练前只有以下硬门槛：

1. 用户正在提交的32卡验收实际完成100→120恢复；
2. 新联合训练入口的action loss确实到达VLM/时间模块/DiT，
   掩码无旁路泄漏，初始策略和checkpoint恢复检查通过；
3. 在预定大训练拓扑上完成短100步保存/恢复验收，确定micro/global batch；
4. 明确并冻结比较对象、可见信息和监督预算，避免强方法对弱基线。

满足后即可开启充分训练，不以“先等小实验显著胜出”为前提。
第一轮建议timeline与匹配强对照各320PPU，共640PPU；
这是资源规划，不是已经测得的最优并行度。global batch/LR/总步数须按
实际三锚点前向和梯度计时确定，不复制旧Stage3的参数冒充等预算。

不得将原32卡DDP optimizer/RNG跨拓扑直接resume为320卡FSDP。
正式匹配实验应共同初始化，或者走明确验证过的权重迁移流程，
完整记录优化器重置与新的训练配方。

## 6. 当前交付边界

- 已完成：原Timeline-A单PPU/CPU DDP验收，27域开发数据，32卡提交脚本，
  原论文近邻复核及本计划。
- 未确认：用户集群任务是否开始训练（本机未见目标输出）。
- 未完成：全VLM联合大训练入口、大拓扑验收、强聚合对照、B转化。
- 未证明：当前timeline结构的独立创新与闭环收益。

保持原任务和源码不动；本文件只记录检查结论和下一步标准。
