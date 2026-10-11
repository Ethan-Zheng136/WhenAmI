# 研究轨迹数据合同（可交给仿真端，非主表评测要求）

2026-10-03。这份文件只定义新Revision研究之后所需数据。
**不要求另一agent中断eval/部署接入，也不意味着本轮已经采集了这些轨迹。**

## 最有判别力的组

同一可见初始条件及相同已执行命令下，记录：

1. 没有抓住；
2. 抓住后，在较晚时刻滑落；
3. 抓住且保持；
4. 发现失败后的一次真实成功恢复（如果环境/现有策略能产生）。

事件标签必须按每个时刻真实状态记录，不能只按最终成功/失败给整个episode回填。
尤其不能把“后来滑落”标成“当时没抓住”。

## 每条记录必需

```text
episode_id / paired_group_id / environment_seed
camera_name / observation_frame_index / control_step / timestamp_if_known
RGB observations
policy instruction at each observation
executed controller command sequence
proprioception actually available to all compared policies (optional but explicit)
controller convention / units / reference frame / gripper meaning
true state labels with acquisition source (simulation-only teacher/diagnostic)
event timestamps: attempt/contact/acquired/lost/placed where definable
terminated/truncated/success/failure reason
policy provenance / intervention provenance
```

状态标签定义要写清：使用接触并不等于已抓稳，夹爪闭合并不等于有物体。
优先同时保存可验证的接触、物体相对位姿及运动状态，再由固定规则生成事件标签。
规则及阈值版本化，不能看我们模型输出后再改。

## 信息隔离

- RGB/proprio/已执行动作属于在线可见输入。
- 模拟器物体状态、未观测接触、未来结果属于teacher/诊断标签，不能进入在线forward。
- 未来专家动作不是当前policy输入；候选动作只能来自部署时合法生成器。
- 配对组必须按组分割train/dev/test，不能一条分支进训练、另一条几乎相同的分支进测试。
- 训练/诊断不能输入随机种子、branch_id、绝对任务进度等标签捷径。

## 不改变任务定义来制造优势

先保留原任务协议，再单独记录受控遮挡/扰动条件。
所有方法看到相同模态、相同历史跨度和频率；对照不能人为不给动作历史或不给当前证据。
如果某事件在当前信息下不可辨识，正确要求是保持合理分布，不是让模型猜中隐藏分支。

这份数据将用于检验“修订旧事件是否帮助当前决策”，不是直接计入现有SIMPLER主表。
