# 2026-10-04：Stage3完成，27域35k对比结果已复核

## 同日17:25 UTC后续：用户提供50k闭环结果

远端截图明确标注35k/50k/WoG-A：50k Google VM72.9%、VA44.4%、Overall58.7%，
本机WoG-A分别74.5%、54.1%、64.3%；50k WidowX Grasp70.8%、Success47.9%，
WoG-A分别74.0%、51.0%。截图报告2655/2655、0错误，原始逐episode数据尚未同步本机。
因此50k仍未在这份闭环主表整体超过WoG-A，不能用下述离线19.41%的改善替代闭环结论。
具体架构评审及下一步见`WHENAMI_DESIGN_REVIEW_20261004.md`。

## 05:43 UTC后续更新：50k同协议评测也已完成

50k的1348条episode预测、完成标记和汇总均已落盘；确认加载step=50000、
1664个state条目严格匹配且所有加载张量有限。数据manifest与35k相同，
复用的WoG-A/Past预测文件hash及全部指标相同（报告中的路径因软链接目录不同而不同）。

| 模型 | 域等权MSE6↓ | 首动作MSE6↓ | gripper accuracy↑ |
|---|---:|---:|---:|
| WoG-A | 0.0513189 | 0.0462216 | 0.90547 |
| 原Past | 0.0523171 | 0.0481758 | 0.90130 |
| Stage3-35k | 0.0422968 | 0.0366779 | 0.92141 |
| Stage3-50k | **0.0413581** | **0.0356455** | **0.92318** |

50k比35k的macro MSE降低2.22%；比WoG-A降低19.41%，19/27域点估计有利；
比Past降低20.95%，24/27域点估计有利。完整表位于
`outputs/stage3_oxe_eval/compare50k_01/summary/RESULTS.md`。

用户同期提供的另一服务器闭环截图显示其所评Stage3的Google Overall低于WoG-A
5.3个百分点、WidowX Success低5.2个百分点，但Drawer任务更好。
截图未给出checkpoint身份或完整运行清单，不能假定它是35k或50k。
离线指标不能覆盖这些闭环负结果；应固定同一闭环协议只替换为最终50k，
记录权重来源与导出manifest，再检验是否改善。以下“50k已启动尚未完成”段落是本日较早时点记录。

## 主训练

`oxe27_forecast_zero2_320_50k_save5k_01` 于2026-10-04 01:33 UTC完成。
本次核查：日志更新数连续1–50000，记录的loss/grad norm均有限；
50k有COMMITTED、16个模型分片、16个optimizer分片、320个RNG文件，
最终在线开发评测与complete.json均落盘。

最终权重：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_joint/
  oxe27_forecast_zero2_320_50k_save5k_01/checkpoints/step-050000/
```

## 昨日评测已完成，但评的是35k，不是50k

27域、1348独立episode、2694动作窗口、3个推理noise seed。
全部来自既定Stage3开发划分，实际训练索引重叠为0；基础预训练可能见过。
同一数据、同一初始噪声数组、同一采样/精度，纯噪声生成动作，不是FM训练loss。
2026-10-03 18:10 UTC完成全部三种模型及汇总。
今天重新读取全部预测NPZ、核对配对信息、重算指标和区间，与昨日结果完全一致。

| 模型 | 域等权MSE6↓ | 首动作MSE6↓ | gripper accuracy↑ |
|---|---:|---:|---:|
| 本地WoG-A | 0.0513189 | 0.0462216 | 0.90547 |
| 原Past | 0.0523171 | 0.0481758 | 0.90130 |
| Stage3-35k | **0.0422968** | **0.0366779** | **0.92141** |

- 对WoG-A：MSE下降17.58%，episode分层配对95%区间[12.20%,22.47%]；
  16/27域点估计更好，首动作MSE下降20.65%。
- 对原Past：MSE下降19.15%，区间[15.37%,22.88%]；23/27域点估计更好。
- 夹爪准确率不是任务成功率。推理seed也不是独立训练重复。
- WoG-A/我们输入信息量与训练历史不同；这是实际策略性能比较，不是机制因果消融。

全部27域表：

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_oxe_eval/
  compare35k_01/summary/RESULTS.md
  compare35k_01/summary/RESULT.json
  compare35k_01/summary_recheck_20261004/
```

## 不应只挑有利的域

有明显优势的点估计包括FMB、Stanford Hydra、UTAustin Mutex、
IAMLab、Berkeley Autolab、Jaco等；不仅是Bridge/Fractal。

相对WoG-A仍较差的域包括Cable Routing、DOBBE、Language Table、
RoboTurk、Austin系列、Viola等。Fractal与NYU Franka基本接近但略差。
16/27域更好不等于27/27全面胜出。

四个域少于16条episode，小域结果需要谨慎看待。
以下为**事后敏感性分析**，不是重新选定主指标：

- 排除两个最大的正向绝对贡献域UCSD Kitchen/FMB，剩余25域MSE仍下降8.70%。
- 排除全部少于16条episode的域，剩余23域MSE下降13.35%。
- episode-micro及训练混合权重汇总也保持正向；完整结果在JSON。

## 结论边界

主Stage3在本轮固定多域离线协议中取得了明确正向结果，
不再只是训练loss或两个监控域的改善。
尚未证明收益来自Future/Past特定模块，更没有证明显式后向timeline机制。
之前冻结旧Past基础模型的小timeline实验的负结果不被本结果推翻。
也不能将这些动作误差直接写成闭环成功率。

## 已启动最终50k同协议复测

仅重算50k学生，复用固定的WoG-A/Past预测。数据/实际噪声/采样/指标不变，
旧35k代码和结果保留不动。隔离代码位于`stage3_oxe_eval50/`，
其中保存了相对35k的逐行diff和来源hash。

```text
outputs/stage3_oxe_eval/compare50k_01/
outputs/stage3_oxe_eval/evaluate_stage3_final50k_01.log
```

此报告写入时50k复测尚未完成，不能把35k成绩冒充50k成绩。
复测结束后自动汇总到`compare50k_01/summary/`。
之后优先做同初始化/同数据/同优化预算的Past-only joint与无Future对齐控制，
将“策略整体更好”与“机制确实有贡献”区分开。

本次机器可读审计：`outputs/stage3_oxe_eval/AUDIT_20261004.json`。
