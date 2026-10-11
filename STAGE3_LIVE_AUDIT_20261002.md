# Stage3性能版本：100步后的运行与信息流自查

2026-10-02，核查至14:36 UTC（北京时间22:36）附近。
主任务：`outputs/stage3_joint/oxe27_forecast_perf320_50k_01`。
只读检查正在运行的源码/产物，测试文件位于recipe指纹目录之外。

## 1. 100步没有卡死

北京时间2026-10-02：
- 22:24:55：100步checkpoint提交完成，保存耗时18.38秒。
- 22:25:13：2048个开发动作窗口评测完成，耗时17.73秒。
- 22:25:25：首段完整性检查通过。
- 22:27:35：新进程恢复后101步完成，生成`RESUME_VERIFIED.json`。
- 本次核查期间继续推进到300步以上，20节点日志未检出OOM、通信超时或异常退出。

中间的空档为计划内保存/评测和重新加载，不是每100步都重启。
后续默认100、2500、5000……50000步保存与评测；周期点不退出进程。

## 2. 真正执行的信息流

```text
Past图像[-15,-11,-7,-3] + current
    → 冻结DINO/Wan → 可训练Past QFormer → H_P、C_P
current + language → 可训练VLM → z
H_P,z → R.forward → C_F_hat
H_P,z,标准化C_F_hat → K → delta
C_P + delta, z, noisy_action, tau → 完整可训练DiT → velocity

训练目标：
 action_FM + 0.1 * standardized_future_alignment_MSE

真实Future图像[+3,+7,+11,+15] + current
    → 冻结DINO/Wan/Future QFormer → C_F_target（仅监督）
```

已检查：
- 历史VAE顺序为旧→新→当前，Future VAE为当前→未来。
- 真Future不作为学生参数传入；改变Future不会改变history_features/history_vae。
- `C_P+delta`、R/K/VLM/Past链路没有错误detach；仅action loss能给各组非零梯度。
- 辅助Future loss会更新R/Past/VLM；目标冻结，不会误更新teacher。
- 16步动作的padding有loss mask；4次FM抽样与条件repeat_interleave顺序一致。
- 训练/评测使用同一因果condition路径；动作评测从纯噪声采样，不把动作真值当输入。
- HSDP节点内16卡分片、跨20节点复制；CPU4-rank额外测试对照未分片全batch，
  两步的梯度平均、裁剪范数、Adam更新与参考数值吻合（FP32容差内）。
- 保存100步有16份Adam分片与320份独立RNG；实际320卡新进程恢复已成功。
- 主任务第1和101步各模块梯度均有限非零；这不是每一后续步的逐模块梯度审计。

额外自查+既有回归：31项通过，日志：
`outputs/stage3_joint/perf_live_audit_tests_20261002.log`。
没有穷举全部模型行为，不把这些检查说成形式化无bug证明。

## 3. 发现的科学范围限制

- 当前为forecast+Past条件修正，**不是完整正逆双向互译**。
  R的future输入adapter/past读出头被冻结，没有独立backward/cycle loss。
  K的“回溯修正”不能替代独立逆向监督证据。
- 当前真实动作dev仅Bridge/Fractal各1024窗口，不代表全部27域或机器人闭环。
- 尚无同精度/同初始化step0和同预算latent结果，不能从100步误差直接断言Future增益。
- Future目标含current信息且短轨迹有边界重复，需干预对照区分真正时间信息贡献。

## 4. 100步评测已落盘，可随时查看

`development/step-000100/results.json`：

| 域 | n | 16步动作前6维归一化MSE | r_6 |
|---|---:|---:|---:|
| Bridge |1024|0.04230225|0.80302088|
| Fractal |1024|0.05427819|0.69849360|

已从NPZ重算MSE/MAE/首动作/夹爪/r/R²，与JSON一致；
2048样本ID、原动作标签、episode/timestep与dev清单一致，分rank拼接无重复缺失。
不能与旧精度、旧样本表直接横比“变好多少”。

## 5. 速度与640卡

旧版step11—100平均12.516秒；新版step11—99非profile步平均2.552秒，
观测约4.90倍（不同时间段，不是控制所有外部条件的严格基准）。
最近约2.6秒/步，50k剩余约36小时量级；ETA会随实际速度变化。
profile的data_wait已接近毫秒量级，不能把目前主要开销继续归因于数据等待。
反向计时包含collective等待，分段各rank最大值不能直接相加。

已准备 **独立640卡、micro2、global1280** 100步短测：

```bash
RESUME=0 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_perf_640_probe.sh
```

申请40节点×16卡，仍用同一模型、全量OXE、FM draws4、50k的LR计划，
只到100步保存/评测后退出，不会自动跑第二个完整50k。
保留当前320训练；不跨world恢复，不改变现有ckpt。入口仅完成语法/dry-run，
640卡实测需要用户提交。比较`FIRST_SEGMENT_COMPLETE.json`的稳态步时再决定。

当前已经accum1；在320卡将micro4改成8会把global变2560，不能保持原1280。
640卡micro4同样把每步训练量翻倍；若仍50k，就不是原64M观察抽样预算。
640卡micro2保持预算，但通信占比可能上升，不能保证线性减半时间。

## 6. 保存/评测频率

当前每2500步约1小时45分至1小时50分有一个新结果，不会只保存100步。
每个完整模型+Adam checkpoint约90GiB，全部21个约1.85TiB量级。
若要更密集观察，更合理的是独立设置“每500步评测、每1000/2500步保存”，
不必为了每次评测都留下90GiB快照。
当前recipe将周期写死并绑定resume；本轮没有热改正在运行的配置。
换640或新版本时可一并调整，不应为误以为只存100步而立刻停掉有效训练。

机器可读审计：
`outputs/stage3_joint/perf_training_flow_audit_20261002.json`。
