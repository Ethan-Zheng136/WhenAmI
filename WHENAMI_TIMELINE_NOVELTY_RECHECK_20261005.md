# Timeline A→B 新颖性复核

核查日期：2026-10-05。仅阅读代码、检索和保存原论文；未修改训练代码、配置或任务。

## 结论

不能确认“当前架构没人做过”。更准确的结论：

1. B 的未来特权监督、辅助条件对齐、推理移除老师均有直接先例。
2. A 的时间标记、多节点、遮挡和双向信息读取也不是新概念。
3. 尚未确认有论文与我们当前 A→B 的全部实现完全相同；这不是首创证明。
4. 候选贡献只能落在具体的动作条件时间线及其可部署结构收益上，
   不能把通用机制或当前尚未实现的在线修订写成创新。

## 实际实现，而非愿景

读取：

- `whenami_timeline_joint/model.py`、`train.py`
- `whenami_timeline_b/model.py`、`train.py`、`loading.py`、`data.py`

A：三个实际时间锚点、逐节点视觉/语言编码、相对时间寻址、共享动作读出；
完整观测与遮挡中间节点交替训练。动作标签只进入动作监督/带噪动作解码器，
不是时间状态编码器的输入。当前不是动作条件物理转移模型。

B：学生只得到一个过去锚点和当前锚点；老师额外得到第三个真实未来锚点。
两者初始化自固定 A10k，老师冻结，学生主体训练。
监督目标是老师的**当前中间节点条件**，不是未来节点的条件或未来视频。

```text
C_teacher(t) = Timeline_A(past, current, future)[t]
C_student(t) = Timeline_B(past, current)[t]

B0: L_action
B1: L_action + 0.1 * (1 - cosine(P(C_student(t)), stopgrad(C_teacher(t))))
```

辅助预测 P 不进入动作头；动作头直接读取学生 C 和当前 cognition。
B 没有显式预测未来节点、跨执行调用保存/修订过去节点或全节点蒸馏。
B0/B1 都继承 A 预训练，不能用这组比较隔离全部 timeline 的价值。

## 直接近邻

### 1. WoG — arXiv:2602.22010，§3.2–3.3

Stage I 用动作监督塑造真实未来的低维条件；
Stage II 冻结未来编码器，同时优化动作与余弦条件对齐。
未来条件预测分支不直接进入 Stage II 动作头，监督通过共享表示发挥作用。

这与我们 B 的训练思想非常接近，不是只有“用了 future”这一表面重合。
区别是 WoG 对齐未来编码器的条件，而我们对齐具有前后文的当前时间节点条件；
我们的学生还保留显式过去节点和时间条件读取。
这个目标构造差异需要实验支持，不能仅凭形式不同宣称实质创新。

### 2. PFD — arXiv:2604.25859v2，§3.2–3.4

题名：Privileged Foresight Distillation: Zero-Cost Future Correction for World Action Models。

真实未来可见的老师与 current-only 学生通过不同 attention mask 构成；
老师和学生每步共享主体参数，不是固定独立老师。
蒸馏目标是 detached 的老师与学生动作速度差，adapter 预测修正，
修正分支在推理时保留并直接影响动作去噪；无需生成未来视频。

与我们的重合：训练期真实未来特权→无真实未来的在线动作策略。
关键不同：动作速度残差 vs 当前节点条件；共享参数老师 vs 固定 A 老师；
推理保留修正头 vs 我们的辅助头不进动作。
不能把二者说成同一架构，也不能再把“未来带来的动作修正蒸馏”泛称空白。

### 3. Think Like a World Model, Act Like a VLA — arXiv:2609.24682，§III-B–C

冻结世界模型，预先缓存表征；学生动作损失加投影后的余弦特征对齐；
推理丢弃对齐头，保持基线动作网络。

它直接覆盖“世界模型特征作为辅助监督，部署不增加该分支”的学习范式。
但其主要目标写作 f_T(o_t)，是当前观测的世界模型表征，
不能误写成其老师也逐样本读取我们这种真实未来时间线。

### 4. MTM — ICML 2023，arXiv:2305.02968，§3 与附录架构

题名：Masked Trajectory Models for Prediction, Representation, and Control。
状态/动作等轨迹 token 带时间和模态编码，通过随机遮挡与双向模型重建，
支持多种预测和控制用途。因此时间节点、mask、双向读取不能单独作为首创。

我们学习供动作解码器使用的时间条件，不等于 MTM 的通用轨迹重建目标；
需要强时序基线，而不是仅凭目标名称排除这条先例。

### 5. HiF-VLA — arXiv:2512.09928

题名：Hindsight, Insight and Foresight through Motion Representation for
Vision-Language-Action Models。
通过历史运动表征、未来运动推理和联合 expert 辅助动作；
已经覆盖 past/future 联合控制这一宽泛主张。

与我们的 per-time action-condition / 当前节点老师目标不是完全相同结构。
本轮读取此前下载的 v2 正文；本轮尝试重新下载 v1 超时，
不把该失败记为成功的新全文获取。

## 可能的贡献与必要验证

候选研究问题：

> 相比同等信息和预算下的聚合条件/普通时间上下文，
> 按时间寻址且由共享动作读出塑造的条件时间线，
> 是否产生在撤除未来特权后仍能改善在线控制的收益？

最低证据：

1. 匹配的 pooled A→B 与 timeline A→B 整条流水线，而非只比较 A。
2. B0/B1 检验额外当前条件监督；不能把此实验称为整个 timeline 的消融。
3. 同等信息、容量和训练预算的强时序上下文/遮挡基线。
4. 区分时间关系收益、额外动作监督收益与普通表征蒸馏收益。
5. 多训练 seed、固定闭环协议，不用离线 MSE 代替任务成功率。

将对齐扩展到所有节点或加一个显式未来头，不会自动建立新颖性。
论文主张必须与已实现且经验证的机制一致。
当前 B 可继续作为转移机制实验，但不应称为已确认新颖的最终大训练。
本核查不建议自动取消或改动任何正在运行的任务。

## 来源与审计

本轮保存的原始 HTML、正文、获取状态、代码及正文 SHA256：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_novelty_recheck_20261005/
  WoG.html / WoG.txt
  PFD.html / PFD.txt
  WD.html / WD.txt
  MTM.html / MTM.txt
  FETCH.json
  AUDIT_MANIFEST.json
```

原文入口：

```text
https://arxiv.org/html/2602.22010v1
https://arxiv.org/html/2604.25859v2
https://arxiv.org/html/2609.24682
https://arxiv.org/html/2305.02968
https://arxiv.org/abs/2512.09928
https://proceedings.mlr.press/v202/wu23o.html
```

阅读范围为摘要和相关方法章节，不宣称逐字精读全部附录或复现外部结果。
检索包括题名，以及 robot future observations distillation、
hindsight teacher VLA、time-indexed teacher、timeline distillation 等组合。
公开检索不能覆盖未发表工作或保证没有漏检，因此结论不使用“全球无人做过”。
