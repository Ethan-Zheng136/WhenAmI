# WhenAmI-D (revision_control) 20k：SIMPLER 闭环评测交接

更新时间：2026-10-11（北京时间）。接收方：NVIDIA 服务器 14 (discover-01) / 15 (discover-02)。
本文只新增，不改任何已有 A/B/C/D 训练、已有 release、已有适配器或已有结果。

## 1. 模型身份（训练均已完成，不需要再训练）

| eval_id | variant | 原始 checkpoint（neolix_al_2） | COMMITTED.json SHA256 |
|---|---|---|---|
| `D0_direct_history_20k` | `direct_history` | `/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_revision_control/training/revision_control_direct_history_B1A10k20k_A50k_320_20k_01/main/checkpoints/step-020000` | `23355378744f393865b186670c08a22580daadd3991ec8a006b6a3c4d8c3eaa6` |
| `D1_revised_nodes_20k` | `revised_nodes` | `/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_revision_control/training/revision_control_revised_nodes_B1A10k20k_A50k_320_20k_01/main/checkpoints/step-020000` | `4f36739346e78d383f9458b7d2bcce5437750b15bc5a0047ebe9765df14ba717` |

- 两者都是 `main/COMPLETE.json` 存在、step=20000、world=320；`model_dcp/` 16 个分片，各约 31 GiB。
- 学生初始化：A10k→B1-20k（COMMITTED `4d9f3b3e…05bb`）；固定教师：Timeline-A50k（`f785848d…1714`）。推理不加载教师。
- 训练设置：body lr 1e-6、memory/bridge lr 1e-4、warmup 1000、20k 步、batch 320、node/revision 权重 0.1/0.1。
- 模型类：`whenami_revision_control.model.RevisionPolicy`（= 原 B Student + `memory.`(NodeMemory) + `bridge.`(ActionBridge，零初始化残差)）。
- 源码契约：checkpoint recipe 中 131 个源文件哈希，已在 14 上与 `Four20k_B_A50k_20261007_local/code` + 本次拷贝的 `whenami_revision_control/` **逐一核对全部一致**。

## 2. D 的推理输入——和 B/C 不同，必须用新适配器

```text
节点  = {t-2g, t-g, t}（截断到 0、去重；g = history_gap = 4；冷启动 K=1..3）
base  = 最近两节点 → 与 B wrapper 完全相同的帧 {max(t-g,0), t}，走原 B 路径
window= 全部 K 帧的 DINO/Wan 特征（与训练 batch() 一样联合计算）+ 整数时间
frames= 每节点独立单帧 VLM 输入（各自 cognition，不广播当前帧）
executed[j] = 控制步 nodes[j-1] … nodes[j]-1 实际执行的动作（集成后），
              用训练标签同一 BOUNDS_Q99 变换（normalize_actions）重新归一化；
              夹爪 = 执行时的开合命令（open_gripper>0.5）
→ RevisionPolicy(inputs, condition_only) → conditions = base + bridge 残差 → 原 DiT
```

- 旧 `timeline_b_adapter/whenami_timeline_b_policy_sim.py` 不认识 D 类（会报 unsupported model class），**不修改它**。
- 新文件（14 上）：`WhenAmI/d_adapter/whenami_d_policy_sim.py`（继承 B wrapper；集成、夹爪后处理、反归一化、噪声 seed、reset 全部沿用）
  和 `WhenAmI/scripts/wog_main_table_worker_d.py`（原 worker 不改，只把策略类替换为 D 类）。
- 训练时节点间隔为 1–12；g=4 在范围内。executed 区间宽度 12。
- 上线前验收：`scripts/d_equivalence_smoke.py`——用已验通 B 权重 + 零初始化 bridge，D 输出必须与 B 路径**逐位相同**（K=1/2/3）。
  2026-10-11 已在 14 GPU7 通过（4 个窗口全部 0 误差，峰值 17.08 GiB）。
- 实现注意：本机 torch 2.2 CUDA 没有 BF16 fused GRU-cell kernel；适配器把 `memory.actions`（7→128 GRUCell）
  在 FP32 下计算后转回 BF16，参数与 state_dict 仍为 BF16（训练时在 PPU 上为 BF16 计算，存在极小数值差异）。

## 3. 打包/导出（在 14/15 上，照 Four20k 流程；全部由一个无人值守脚本完成）

```bash
# 14（D0）与 15（D1）各一个 tmux；可重复执行，已完成阶段自动跳过
MODEL=D0 GPUS="4 5 6 7" MIN_GPUS=2 bash /DATA_EDS2/zhenggt2407/WhenAmI/scripts/run_d_simpler_pipeline.sh
MODEL=D1 GPUS="3 4 5 6" MIN_GPUS=2 bash /DATA_EDS2/zhenggt2407/WhenAmI/scripts/run_d_simpler_pipeline.sh
```

阶段：download（远端只读 sha256 + scp 断点重试 + 逐文件 sha 校验）→ export → replay（评测环境经 SIMPLER wrapper）→ equivalence → 删除本次 DCP 副本 → 评测 worker → aggregate。失败写 `results/<eval_id>/FAILED`，成功写 `DONE`，状态见 `PIPELINE_STATUS.json`，日志 `pipeline.log`。GPU 只用候选集中显存占用 <2000 MiB 的卡。


1. 传原始 DCP（只要 `COMMITTED.json` + 整个 `model_dcp/`）到 `WhenAmI/dcp/D20k_20261011/<eval_id>/`，逐文件 size + sha256 校验。
2. `export_d20k_local.py`：严格加载（DCP key 集合与 shape 完全一致、全部 finite），导出 BF16 safetensors 到
   `WhenAmI/release/D20k_20261011_local/{D0_direct_history_20k,D1_revised_nodes_20k}`，
   MODEL.json 记录 model_class/variant/源哈希；导出→重新加载→同输入同噪声回放必须 0 误差；生成 SHA256SUMS。
3. 代码快照 = `Four20k_B_A50k_20261007_local/code` + `whenami_revision_control/`。
4. 导出完成后删除本次下载的 D DCP 副本（只删本次新建目录）。

## 4. 评测协议（与 B1_A50k_20k 主表完全一致）

- worker：`wog_main_table_worker.py` 的同一 protocol（321 组、2655 episode、Google VM+VA 与 WidowX），
  `--history-gap 4 --seed-base 420000 --num-workers 4 --save-sample-videos --retry-errors`。
- 结果目录：`WhenAmI/results/D0_direct_history_20k`、`WhenAmI/results/D1_revised_nodes_20k`（新建）。
- GPU：D0 在 14 的 GPU 4,5,6,7（tmux `whenami-d0-simpler`）；D1 在 15 的 GPU 3,4,5,6（tmux `whenami-d1-simpler`）。
  共享服务器：启动前 nvidia-smi 确认这些卡空闲；不动其他卡、其他用户进程或已有 tmux。
- 参考耗时：B1_A50k_20k 每 worker 约 8.7 h（78 episode/h）；D 每步多 3 次单帧 VLM 前向，预计更慢。

## 5. 回传

按 Four20k 格式：protocol.json、worker-*.jsonl、worker-*-status.json、summary.json（aggregate_wog_main_table.py）、sample_videos；
另附 D 的 model_identity（eval_id/variant/源哈希/本地权重哈希）和 equivalence/replay 结果。
Google VM、VA 分开报告；WidowX grasp 与 success 分开。
