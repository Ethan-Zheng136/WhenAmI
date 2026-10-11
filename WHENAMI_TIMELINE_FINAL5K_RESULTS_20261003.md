# Real Timeline最终5k：实际动作采样结果（2026-10-03）

两组均完成5000步，固定最终checkpoint；没有挑各自最佳步，没有重训。
本文件为timeline独立研究评测，不是另一agent负责的Stage3主表。

## 协议

27域共351条不同episode、1053个时间锚点，采样seed42/43/44。每域最多16条，最少1条，6个域不足8条；小域结果不稳定。
所有模式共享同一冻结Past/VLM/DINO/Wan/DiT、合法输入和初始Gaussian噪声；9个Euler区间，FP32积分、BF16动作头。不是目标动作加噪的FM评测。
主指标为按有效动作mask平均、各episode和域等权的归一化截断MSE；另报未截断误差。不能跨旧表直接比较绝对分数。三个采样seed不是三个训练seed。

## 主结果

| 模式 | macro MSE6 | 未截断MSE6 | 首动作MSE6 | gripper accuracy |
|---|---:|---:|---:|---:|
|past|0.0534508|0.0544933|0.0442998|0.94734|
|direct_history|0.0541812|0.0550455|0.0452781|0.94427|
|revision|0.0541152|0.0549394|0.0452528|0.94375|
|revision_no_update|0.0543959|0.0552781|0.0453398|0.94342|

## 配对比较

正数表示后者MSE降低；区间为按域分层重采样episode，仅对固定模型/固定三噪声seed。不是跨训练种子区间。

- past_to_direct_history: -1.367%（95%区间[-2.705, -0.059]%）；12/27域点估计有利。
- past_to_revision: -1.243%（95%区间[-2.554, 0.025]%）；12/27域点估计有利。
- direct_history_to_revision: 0.122%（95%区间[-0.114, 0.376]%）；10/27域点估计有利。
- revision_no_update_to_revision: 0.516%（95%区间[0.219, 0.850]%）；16/27域点估计有利。

## 分锚点

| 模式 | 第1点 | 第2点 | 第3点 |
|---|---:|---:|---:|
|past|0.0531948|0.0543144|0.0528432|
|direct_history|0.0539647|0.0551922|0.0533867|
|revision|0.0539134|0.0551479|0.0532843|
|revision_no_update|0.0539134|0.0554492|0.0538252|

## 解释边界

- 原Past不是等预算重训对照；direct_history/revision才是核心匹配实验。
- 关闭gate是固定权重推理干预，不是公平重训对照。
- 全部样本来自timeline internal_dev，原端点预训练可能见过；不宣称封存测试或闭环效果。
- 没有接触/事件真值标签，动作结果不能直接证明历史事件解释正确。
- 本轮首次补评进程在约300条轨迹被SIGTERM终止，未生成结果；第二次仅增加逐轨迹落盘和恢复，保持相同采样/模型/指标。
- 主训练源码、checkpoint和另一边评测工作没有修改。

## 完整产物

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline/action_eval_final5k_02
  RESULT.json
  ANALYSIS_AUDIT.json
  cases.json
  manifest.json
  {past,direct_history,revision,revision_no_update}.npz
```

## 本轮结论（按预定主指标，而不是挑有利数字）

- 相对原Past，direct_history的macro动作MSE升高1.37%，revision升高1.24%；未显示整体策略增益。
- revision相对direct_history仅降低0.122%，配对episode区间[-0.114%,0.376%]跨零；只有10/27域的点估计有利。不能宣布后向修订胜出。
- 相同revision权重关闭更新后，MSE高于正常revision约0.52%（以关闭版本为分母计算改善），说明分支有作用，但不代表胜过强对照或原Past。
- 三个锚点的revision/direct差异区间均跨零，没有明确的“越往后修订越有效”证据。
- 首动作误差及夹爪准确率相对Past也没有改善，不是仅主指标选择造成的反例。
- 因此当前1.26M冻结基础模型+条件残差方案在本轮采样诊断中不算方法成功；不把它扩大解释成所有timeline思想无效。
- 351条episode分布到27域，6个域不足8条；固定单训练seed且基础预训练可能暴露过数据，不能将当前区间当成最终泛化结论。
