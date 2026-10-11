# 20k学生任务中断：故障与阶段性结果分开判断

2026-10-02 10:07 UTC。原任务尚未跑完，未改训练源码、未重启、未删除checkpoint。

## 故障事实

用户提供的平台日志显示先收集17个PID的callstack，其中16个为训练进程，
随后多次重新执行启动入口，并被`RUNS exists. Use RESUME=1 ...`拒绝。
明确的二次故障是：已有运行目录却以新建模式重新启动，尚未进入Python续训。
这不是“原checkpoint损坏”的证据。

最初为何停滞/中断仍未查明。当前容器不能访问任务容器内的
`/pai_aimaster/job/stk_results/dlc16zw22ttd4am2-master-0/`。
`stack_saved.txt`、`ppu-gdb_saved.txt`及平台退出原因需要另行取得。
不能从pynvml警告、requirements缺失提示或PPU环境中nvidia-smi无进程输出
单独推断原始OOM、设备损坏或代码死锁。

## 恢复检查

16个实验最新checkpoint全部可读，模型与optimizer张量有限；
原source hashes、默认训练settings、base/teacher小checkpoint SHA256匹配，
optimizer和torch/cuda RNG记录存在。

- 普通/forecast组：最近完整保存点主要15k，一组17.5k。
- distill组：最近完整保存点10k或12.5k。
- 最多重放2470个未保存更新，不是从头训练。
- 所有组尚无complete.json，正式动作评测未完成。

用户确认原平台任务已停止或把其重启入口改成恢复模式后，可提交1节点16卡：

```bash
RESUME=1 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_stage3_causal_student_20k.sh
```

不要删除RUNS，不要同时让两个任务写同一目录，不改本轮学习率/数据/预算。
恢复命令已dry-run通过，但本轮未实际替用户恢复任务。
若平台会自动重试，应使它重试的是上述恢复入口。

## 同一步数的初步结论：开发FM，而非动作成功率

四训练种子均值，全部比较学生新增10k这一共同节点：

| 组别 | 初始开发FM | 10k开发FM | 增加 |
|---|---:|---:|---:|
| latent普通学生 | .183178 | .206419 | 12.69% |
| forecast | .183178 | .206457 | 12.71% |
| distill_past | .183178 | .196581 | 7.32% |
| distill_future | .183178 | .196931 | 7.51% |

训练FM降低、固定开发FM持续升高，支持当前缓存/训练配方出现过拟合。
教师蒸馏缓解退化，但Future教师暂未优于Past教师。
不能把12.5k蒸馏组与17k普通组直接比较并归因机制；此表已统一为10k。
未来对齐loss也不能代替实际动作误差。

最终20k/三噪声动作评测尚缺失：没有足够证据宣称因果Future学生有效，
也不能据此宣称所有WhenAmI双向方法无效。
恢复本轮的目的在于补齐预先指定的预算及动作评测，不是声称再训练就会变好；
不应原样放大这个重复小缓存的训练配方。

机器可读核查：
`outputs/stage3_causal_student/interruption_resume_audit_20261002.json`
