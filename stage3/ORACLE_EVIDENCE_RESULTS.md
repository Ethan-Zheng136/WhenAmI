# 真实Future信息接口诊断：已完成

2026-10-02。当前R+K正式结构不变；下面是独立的特权证据诊断，不是新部署策略。
pipeline对比图见 `PIPELINES.md`；执行前计划见 `ORACLE_EVIDENCE_PLAN.md`。

## 1. 先排除“只是R预测不准/学生与老师混合训练”这一解释

单独训练K，只输入真实C_F，不训练R，不混合学生分支、不加alignment或蒸馏：
同一K初始化、batch8、LR1e-4、seed42、1000步，固定Past DiT。

| 动作块前6维normalized MSE ↓ | Fractal | Bridge |
|---|---:|---:|
| 普通Past K适配器（无逐样本Future） | .036805 | .042156 |
| 只用真实C_F训练K | .037858 | .043033 |
| 原混合训练K的真实C_F oracle读出 | .036721 | .042681 |

纯真实C_F路径并没有变成更强的老师。这个结果不能只怪R预测误差，
也没有支持“去掉学生混合就能解决问题”。

## 2. 进一步区分条件压缩、丰富特征与真实未来

四个H接口probe完全相同：输入16×1024，K可训练参数均1,334,688；
相同初始化/样本/动作噪声，1000步，统一使用真实H_F的TRAIN均值/std。
所有组只用Past DiT的action FM；无R、alignment、蒸馏、多步采样loss。

- **mean H**：固定训练均值，不含逐样本Future。
- **C-lift H**：用已知Future投影W的伪逆，将真实C_F还原到H空间；
  未保留的坐标固定为训练均值。严格只含逐样本C_F信息，不偷看真实H_F。
- **real H**：真实未来在最终1024→32投影之前的H_F。
- **static H**：同一个Future编码器，四张“未来帧”全换成当前帧；
  VAE也只看到重复当前帧。因此它是额外当前图像表征控制，不含真实未来观测。

### 特征/控制审计

- 新提取1024个static H；12个跨样本复算的原真实H与原缓存**逐值一致**。
- W的行秩32。C-lift重投影误差：float64最大1.55e-14，
  float32最大1.43e-6，确实保留了C_F。
- mean H组动作指标与之前32维均值接口的普通K相同；
  其更宽投影接收全零标准化输入，额外权重不提供样本信息。
- C-lift与真实H虽然匹配结构和统计，但秩/协方差不同，仍可能有优化条件差异；
  不能将任何差值唯一归因于“信息被丢掉”。

## 3. 实际动作结果

全211验证窗口，统一FP32、逐窗口seed42、9个Euler区间。
主指标为整个16步动作块前6维normalized MSE，越低越好：

| K证据输入 | Fractal | Bridge |
|---|---:|---:|
| mean H：无逐样本Future | **.036805** | **.042156** |
| C-lift H：仅真实C_F信息 | .037876 | .042371 |
| real H：完整投影前Future特征 | .037030 | .042969 |
| static H：重复当前帧编码 | .037539 | .042479 |

关键配对差值：

| 比较（右者相对左者） | Fractal MSE变化 | Bridge MSE变化 |
|---|---:|---:|
| mean H → real H | +.61% | +1.93% |
| C-lift H → real H | −2.23% | +1.41% |
| static H → real H | −1.36% | +1.15% |

上述三种比较的episode-bootstrap 95%区间均跨零。
真实H相对压缩条件在Fractal有一些较好的点估计，但跨数据集不一致；
更重要的是，它仍未超过不使用逐样本Future的普通适配器。

## 4. 应怎样解释

本轮没有找到比普通Past适配器更强、且跨两数据集一致的Future老师。
因此当前证据不支持：
- 只把问题归结为R预测Future不准；
- 只靠取消学生/oracle混合训练；
- 只把32维条件换成更宽H特征，就能恢复稳定的回溯收益。

**但这也不证明“未来没有用”或“Past本来就包含一切”。**
尚未排除：冻结Past读出/表征的限制、单中心窗口目标与时间进程理解不匹配、
样本规模/分布、单种子优化，以及只有特定轨迹阶段才需要Future证据。

当前最可靠的工程结果仍是Past残差适配；Future回溯的独立贡献还未成立。
下一步应先确认未来证据能够改善哪个“当下状态/动作解释”，
再设计跨真实时间锚点的约束，而不是继续叠R/K循环或把WoG条件当通用时间状态。
这些是后续待检验的问题，不是本轮已验证的新架构。

## 5. 局限

- 813训练/211验证，按episode隔离；端点预训练可能见过这些episode。
- 已经观察过的验证集，单训练/动作采样种子；不是最终test或闭环成功率。
- oracle结果读取真实未来，不可当作部署分数；static/mean只是诊断控制。
- 所有首动作、r/R²、gripper指标及逐样本预测均保留，未只保存主指标。
- 冻结Past参照在所有probe中逐值一致；Stage1/2权重未写入，文件属性及参考
  源码/缓存hash检查通过。

## 6. 执行与文件

真实Future C oracle已完成；静态编码与四组H对照于05:27—05:41 UTC执行完成。
40项测试通过；本轮无遗留训练/评测进程。

代码：
`oracle_evidence.py`、`prepare_static_future.py`、
`compare_oracle_evidence.py`、`test_oracle_evidence.py`。

产物根：
`/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_oracle_evidence/`

```text
condition_oracle_01/{config.json,train.jsonl,checkpoint.pt,results.json,*.npz}
static_future_cache_01/{manifest.json,future_static_h.npy}
{mean,projected,hidden,static}_h_01/{config.json,train.jsonl,checkpoint.pt,results.json,*.npz}
comparison_01.json
comparison_01.log
execution_01.log
tests_01.log
```
