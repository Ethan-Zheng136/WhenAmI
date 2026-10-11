# R + K 首轮：实现、共同动作监督与多步采样loss对照

2026-10-02，全部训练/评测已完成。固定计划见 `REFINEMENT_01_PLAN.md`。
**实现可用、梯度路径成立；尚未证明Future/回溯机制有额外行为价值。**
不是Stage4在线管线集成，也不是闭环成功率实验。

## 1. 到底是谁共同指导action

并非Future/Past两个DiT投票。在线只有一个冻结Past DiT：

```text
Past H,z -> R -> 预测Future条件 ----+
                                  |
Past H,z ------------------------ K -> delta_C_P
                                             |
                         原C_P + delta_C_P -> 冻结Past DiT(z_P) -> action loss
```

同一份在线action loss通过DiT、K反传到R。Future端点只供应训练目标/特权条件，
其DiT不参与本轮动作读出。真实Future oracle是单独接口，不进入在线forward。

K出口零初始化，所有组step0的delta严格为0，online FM与Past逐值一致。
此时R的纯action梯度为0是预期的；K出口进行一次参数更新后，即可到达R。
这是**优化器更新**，不是“采样多走一步”。

真实Past DiT、BF16单PPU smoke（20步，包含4区间rollout）：

| 步数 | R纯在线action梯度范数 | K出口梯度范数 |
|---|---:|---:|
| 0 | 0 | .060711 |
| 1 | .00009859 | .060180 |
| 20 | .00035673 | .057876 |

正式teacher组step1000：R=.005205，K主体=.019323，K出口=.058383。
这些审计只用online FM，不混入Future alignment/oracle loss；不是用辅助loss
的梯度冒充action梯度。所有reader参数无梯度且版本未改变。

## 2. “多走几步”实现了什么

- **标准FM**：一个随机tau/noise样本，已监督完整16步动作块，并能沿整个
  R→K→DiT链反传；不是只监督一个物理动作，也不是只反传一层。
- **多tau重复**：实现 `--fm-draws`，独立随机抽样后取平均，本轮训练固定1；
  已有单元测试，但未把多draw训练的效果冒充成已评测。
- **多步动作采样**：额外试验从纯Gaussian噪声展开4个Euler区间，对最终
  动作计算MSE，全程不detach中间x、不以真值动作初始化；权重.25，只加在线分支。
- **R/K多轮来回修正**：没有实现。那需要显式可更新的中间状态，不能把
  同一套静态输入重复喂入称为多轮时间推理。

全部动作评测都维持原10点网格（9个Euler区间），没有更换评测步数。
4区间训练本身有粗求解器近似，结果仅对应这个辅助目标及权重，不能推出
“所有多步训练都无效”或“多步一定更好”。

## 3. 实验设置与控制

- 同一813训练窗口、211验证窗口（Fractal105/27 episodes，Bridge106/28）；
  按完整episode隔离。仍是已看过的验证集，不是最终test。
- 五组均1000步，batch8、LR1e-4、seed42；最终1000步checkpoint，不挑best。
- R：256宽/3层；仅正向部分可训练3,704,096参数。
- K：128宽/2层/4heads，1,207,712参数。
- 普通K组只有1.21M可训练参数；其余组4.91M。额外加入等参数、无逐样本
  Future监督的R+K latent组，避免把容量增加当成Future收益。
- teacher action项为 `(online_FM+oracle_FM)/2`；其R收到的在线action权重为.5，
  与predict的1不同。该归一化避免K总动作梯度直接翻倍，但不能将比较解释为
  完全隔离了所有梯度预算因素。不是等算力对照。
- 不加行为蒸馏，不解冻任一Stage1/2端点；在线只用Past的z，无Future VLM开销。

## 4. 实际动作主结果

整个16步动作块前6维 normalized MSE，越低越好：

| 路径 | Fractal | Bridge |
|---|---:|---:|
| 原Past | .039149 | .046949 |
| Past + 普通K（固定Future均值） | .036805 | **.042156** |
| R+K普通latent（等参数，无Future监督） | **.036693** | .042457 |
| R+K预测Future，无oracle训练 | .036798 | .042658 |
| R+K预测Future + oracle训练 | .036710 | .042798 |
| 上组 + 4区间采样loss | .037338 | .043127 |

### 可以支持的结论

1. **Past残差适配路线有值得继续验证的改善。** 普通K相对原Past的主误差
   降低Fractal5.99%、Bridge10.21%；固定种子条件下episode bootstrap区间均为负。
2. **不能把改善归因于Future或回溯。** 预测Future相对等参数latent反而
   Fractal +.286%、Bridge +.472%，两个区间均跨零。
3. 加oracle训练相对预测Future：Fractal −.239%、Bridge +.328%，区间均跨零。
4. 4区间采样loss相对teacher FM：Fractal +1.711%、Bridge +.770%，区间均跨零；
   本轮未见收益。最后100步中位步时 .285s 对 .109s，约2.61倍。

这些区间不覆盖训练种子、采样种子、预训练预算或新数据分布的方差。

### 次指标也保留，不能只报有利维度

| 路径 | Fractal首动作MSE | Bridge首动作MSE |
|---|---:|---:|
| 原Past | .036345 | .036182 |
| 普通K | .035647 | .034422 |
| 等参数latent | .035120 | .034594 |
| 预测Future | .034679 | .035036 |
| 加oracle训练 | .034996 | .034353 |
| 加4区间采样loss | .035525 | .034233 |

4区间组Bridge首动作稍好，但不能据此遮盖主指标两数据集都没有改善。
Fractal gripper准确率：原Past .89702，普通K .89048，teacher .88631；
并不是所有动作分量都改善。全部r/R²/首动作/gripper指标已保存在JSON中。

## 5. 真Future有没有给K提供更强的老师

teacher组把真实Future送入K的特权评测：
Fractal MSE=.036721，Bridge=.042681；对应在线=.036710/.042798。
与普通K相比没有明确的额外优势。
因此当前没有依据马上做更多R/K循环、更多蒸馏或扩大训练。
下一步应先诊断真实Future证据是否在这个接口/数据集上提供了有用增量，
而不是只要求R更像原Future条件。

输入干预：teacher组在线预测Future比换成固定均值略好，但这不等于胜过
单独训练的普通K。已在完整对照中同时记录，不能只引用干预结果。

## 6. 执行与审计

UTC：
- adapter训练04:43:26—04:45:32；
- predict训练04:45:32—04:47:09；
- teacher训练04:47:09—04:49:37；
- teacher_roll4训练04:49:37—04:55:05；
- 四组动作评测04:55:05—05:01:29；
- 等参数latent训练05:01:44—05:03:21，评测/比较05:05:09完成。

36项测试通过，含真实CPU双进程DDP旧入口回归、新R/K来源隔离、
oracle detach、零初始化、纯action到R梯度、Euler全链梯度、multi-tau平均、
save/load。真实单PPU smoke及全部五组训练/重载/实际动作评测通过。

所有五组固定Past参照预测数组逐值一致，并与旧Pilot01的Past参照逐值一致：
样本ID、episode、timestep、raw target、预测全相同。
Stage1/2 checkpoint未写入，文件大小/mtime与manifest一致；
参考源码/缓存hash保持不变；没有在线未来泄露。
没有cluster任务提交，也没有遗留本轮训练/评测进程。

## 7. 文件

代码：`stage3/{refinement,train_refinement,evaluate_refinement,compare_refinement,test_refinement}.py`

四组固定入口：`run_stage3_refinement_pilot.sh`，仅本机单设备，拒绝已有输出目录。
等参数latent补充使用 `train_refinement --mode predict --alignment-weight 0`，
1000步后用 `evaluate_refinement` 评测，并运行 `compare_refinement --include-latent`。

产物根目录：
`/mnt/pfs/zhengguantian/WoG_v2/outputs/stage3_refine/`

```text
rk_{adapter,predict,teacher,teacher_roll4,latent}_01/
  config.json
  train.jsonl
  validation.jsonl
  gradient_audits.jsonl
  checkpoints/step-001000.pt
  complete.json

rk_{adapter,predict,teacher,teacher_roll4,latent}_01_eval/
  results.json
  *.npz

comparison_01.json
comparison_extended_01.json
pilot_01_execution.log
tests_01.log
```
