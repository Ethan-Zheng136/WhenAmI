# WhenAmI A+ 5k：checkpoint 与独立机制验证

报告生成：2026-10-13T01:55:27.076010+00:00。仅本机单卡只读评测，未提交/修改正式训练。

## 1. 保存与训练状态

Checkpoint：`/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/training/timeline_force_Aplus_320_50k_01/main/checkpoints/step-005000`。
- COMMITTED.complete=true，step=5000；320 ranks，hybrid_zero2，shard_width=16。
- 16 model DCP shards/.metadata、16 optimizer shards、320 RNG 文件均在；实际加载 1551 项，新增/遗漏为0，模型浮点权重有限。
- save=19.42秒，DEV=13.54秒，之后正常继续训练；不是保存卡死。
- 报告时 main 到 5458；日志自121连续、loss/grad有限；最近进度约3.07s/update。
- 正式源码145文件指纹未变，原 preflight 通过；不绕过正式U对“完整A+ teacher”的校验。

## 2. 这次验证了什么
- 自动 FSDP DEV：410条、27域，三推理采样seed；新架构init→5k；不要把这个init当作旧A原生forward。
- 独立 native DEV：同410 episode、相同targets/masks/source hash、同Gaussian identity、9 Euler；A+5k/旧A50k，以及stale/隐藏中间帧干预。
- 初始27条仅工程诊断。每域1条的固定域bootstrap退化，不使用其CI作统计保证；本报告主体为完整410条。
- 模型真实smoke校验：cached stale读取与实际 stale_current forward逐位相等；因果中间节点与物理截断prefix逐位相等。
- 离线MSE不是SIMPLER成功率；seed42/43/44是推理噪声，不是独立训练重复；此DEV不是未见过的封存test。

## 3. 自动完整DEV：A+确实在学习

|模式|新forward init|A+5k|相对MSE下降|
|---|---:|---:|---:|
|causal_last|0.051763|0.042678|+17.55%|
|causal_middle|0.046463|0.044960|+3.23%|
|hindsight_middle|0.045439|0.042181|+7.17%|

中间causal学习的MSE区间跨0；不能说全部模式均已稳定改善。这些数值只比较新forward的学习，不证明A+超过旧A。

## 4. 原生完整DEV：与真正的旧A50k对照

|模式/指标|旧A50k|A+5k|改善（正为好）|固定域episode 95%区间|
|---|---:|---:|---:|---:|
|causal_last/mse6|0.040806|0.042617|-4.44%|[-7.41, -1.71]%|
|causal_last/first_action_mse6|0.030896|0.032386|-4.82%|[-8.75, -1.44]%|
|causal_middle/mse6|0.044021|0.044914|-2.03%|[-4.02, +0.18]%|
|causal_middle/first_action_mse6|0.034434|0.034541|-0.31%|[-3.26, +2.53]%|
|hindsight_middle/mse6|0.039962|0.042136|-5.44%|[-8.00, -3.06]%|
|hindsight_middle/first_action_mse6|0.031139|0.032053|-2.94%|[-6.99, +0.71]%|

当前夹爪准确率：95.724%→95.162%，差-0.562个百分点；它是阈值准确率，非抓取成功率。

特别说明：最初27条工程诊断里的“首动作更好”不在完整410条上成立。完整对照当前首动作也落后旧A，应以完整结果为准。

## 5. 核心：重读过去 vs 帮助当前控制

```text
相同三帧与当前动作目标
  ├─本次更新旧节点 → 当前节点 → DiT → 当前动作（updated）
  └─保留上次旧节点 → 当前节点 → 同一DiT → 当前动作（stale）

另一个问题：旧event1只看截止1 vs额外看截止2 → event1动作
后一项是解释过去，不等于前一项的控制收益。
```

|消融/问题|干预前MSE|干预后MSE|改善（正为好）|固定域episode 95%区间|
|---|---:|---:|---:|---:|
|旧时刻causal→hindsight|0.044914|0.042136|+6.18%|[+4.19, +8.45]%|
|当前stale→updated|0.042809|0.042617|+0.45%|[+0.18, +0.73]%|
|当前中间历史hidden→visible|0.042921|0.042617|+0.71%|[-0.36, +1.80]%|

当前首动作修订收益+0.55%；整段+0.45%，18/27域方向有利。是弱正向离线信号，不是大幅机制成功；固定域episode CI不包含训练随机性/域泛化不确定性。

## 6. 当前动作不同预测位置

|预测区间|旧A→A+相对下降|stale→updated相对下降|
|---|---:|---:|
|first_action|-4.82%|+0.55%|
|first_4|-6.22%|+0.54%|
|steps_4_7|-5.18%|+0.27%|
|steps_8_15|-3.42%|+0.38%|
|whole_16|-4.44%|+0.45%|

图：`/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_aplus_audit_20261013/dev410_01/HORIZON_AUDIT.svg`。有效标签mask逐位置处理；“预测16步”不代表部署真的执行16步。

## 7. 当前还不能下的结论
- 尚无完整4锚点DEV：训练teacher是p0,p1,t,f；当前DEV仅3锚点，只有中间节点能获得后来的观测，last/t之后无第四帧future。
- A+新增训练5k从旧A50k warmstart；不是匹配预算的从头架构对照。模型参数/encoder调用batch形状有差异，BF16接近零的差异不宜过度解释。
- teacher局部事后预测优于因果预测，不证明可蒸馏、U已更好或闭环已更好；要分别验证。
- hidden own-frame误差变高不证明模型成功重建隐藏状态；需完整可恢复任务与对照。

## 8. 下一步建议
1. 现有A+50k继续，不因5k弱信号热改/停主线；10k/后续ckpt按同协议跟踪。
2. 补四锚点teacher完整考试：同t目标因果(p0,p1,t)对比事后(p0,p1,t,f)，包括历史隐藏及匹配旧A对照。仅使用已登记DEV episode，不混入训练/封存test。
3. 更重视“修订后的历史改善当前动作”是否随训练增强；若始终微弱，就不是再加loss名字或让U照抄修订即可解决。
4. U仍需合法输入训练、匹配action-only/direct对照和闭环结果。若提前做5k teacher学生实验，必须独立研究入口/recipe/smoke，不冒充正式完成teacher。

## 9. 可追溯产物
- `/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_aplus_audit_20261013/FSDP_DEV5K_ANALYSIS.json`
- `/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_aplus_audit_20261013/dev410_01/RESULT.json`
- `/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_aplus_audit_20261013/dev410_01/aplus/RESULT.json`
- `/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_aplus_audit_20261013/dev410_01/old_a/RESULT.json`
- `/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_aplus_audit_20261013/dev410_01/HORIZON_AUDIT.json` / CSV / SVG
- `/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_aplus_audit_20261013/SAFETY_VERIFIED.json`
- 独立新增源码仅 `whenami_aplus_audit/`：evaluate.py、test_audit.py、report.py、test_report.py、visualize.py、finalize.py；正式训练源码不修改。
- 最初失败smoke原样保留：BF16投影batch形状导致cached stale与真实forward不一致；独立评测修复为投影完整version再切片，真实5k parity现已通过。
