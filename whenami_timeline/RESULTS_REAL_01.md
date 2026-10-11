# Real Timeline 01：这次已接上真实图像和真实动作

2026-10-03。此文件只描述本轮真实数据/预训练模型链路验收。
不将8步或4步结果当作方法效果，不接手另一agent的主表评测。

## 1. 相比前几轮的实质变化

之前：低维合成观测、confirm/retry等分类标签，小型GRU/MLP。

现在：

```text
真实OXE轨迹三个明确时间点
→ 每点实际DINO/Wan历史视觉 + 原VLM + Past QFormer
→ 16个128维token/时间点的timeline、实际区间动作和Δt
→ prefix内正向filter与后向门控修订
→ 每点16×32条件残差
→ 同一个真实原Past DiT
→ 每点对应16步真实动作的flow-matching loss
```

三个时间锚点≠16个query。各动作loss只使用该锚点之前合法观测和动作，
不把已见后续动作的旧事件状态拿去预测那个旧动作窗口。
真实训练runner每一步重新从全量A中抽轨迹和编码图像，不依赖主训练结束。

## 2. 已完成的实机检查

### CPU合同

13项通过，包含：
- 所有四模式的prefix因果性；
- 实际Δt及完整区间动作；
- 未执行padding不影响输出；
- 各锚点动作都获得监督；
- 只有动作loss时，梯度到F/B/gate/condition桥；
- 零初始化桥保持原条件；
- direct-history/revision同容量、同初始化；
- 真双进程DDP小timeline保存恢复后，下一更新逐值一致。

日志：`outputs/whenami_timeline/tests_ddp_final_01.log`。

### 本机PPU：真实27域、81锚点

`real_smoke_01/RESULT.json` complete=true。
- 27个域每域一条实际TRAIN轨迹，每条3点，共81个真实锚点；
- DINO/Wan/VLM/Past均为已有实际权重，不是小随机视觉骨干；
- 一个原Past DiT供全部锚点共享，基础模型冻结；
- independent/forward/direct_history/revision分别8个更新；
- timeline各1,256,513参数，输入/输出浮点有限；
- 四组第一步条件与原Past一致；
- 后续仅action loss可到相应时序分支；
- 改变最后锚点，不改变前面锚点输出；
- timeline保存/重载的条件逐值一致；
- 原端点大小/mtime与配套元数据复核未变；
- 本机峰值约17.51GiB（冻结BF16主干、小timeline optimizer），
  不能用此数值保证32PPU所有batch的显存峰值。

revision的纯action梯度范数示例（8步之后）：

| 模块 | 梯度范数 |
|---|---:|
| 正向transition |2.71e-4|
| backward attention |2.60e-6|
| backward update |2.11e-5|
| relevance gate |6.19e-7|
| bank attention |8.43e-4|
| condition readout |1.29e-1|

非零只证明通路；B/gate梯度明显较小，需要正式研究时监测。
不能由非零梯度推出有效语义、正确证据归属或闭环收益。

### 本机PPU：实际流式训练器

`raw_runner_single_01/COMPLETE.json` complete=true：
- 每步重新取实际图像和动作，不是复用固定27条；
- 2步训练→保存→退出→新进程恢复→3/4步训练；
- 内部动作FM监测、权重/Adam/RNG恢复跑通；
- `RESUME_VERIFIED.json`记录2→3；
- 无生产模型参数更新。

这与CPU DDP证明互补，但仍不等于32个不同PPU已经运行成功。
集群入口会先2步保存恢复，失败不会继续消耗剩余5000步。

## 3. 科学范围

本轮真正实现的是**真实图像与动作上的多锚点因果timeline训练链**。
仍未实现/验证：

- 完整概率smoother或原始共享转移因子的严格推断；
- 未来特权教师对全部事件的后见监督；
- 旧事件修订后重算整条filter轨迹；
- 真实抓取/失败/状态标签；
- 大模型联合解冻或20k主Stage3权重继承；
- 相对强memory/direct-history的显著收益；
- 闭环与方法新颖性结论。

这些不是当前任务要等待50k的原因，而是新的独立研究任务。
本轮先固定基础动作能力，用共享DiT检验timeline到底怎样改变真实动作，
随后用正式对照决定是否需要更强后见目标、改读出或联合解冻。

## 4. 下一步已有可提交入口

`run_whenami_timeline_research.sh`：

- 每任务2节点×16卡；
- `TIMELINE_MODE=direct_history` 与`TIMELINE_MODE=revision`两个独立任务；
- 各5000步，global64轨迹×3锚点×2FM；
- 每500保存，每250监测固定internal-dev动作FM；
- 所有来源为原TRAIN，再做研究internal-dev隔离，不访问原dev/sealed；
- 不把固定internal-dev FM监测称为机器人成功率或最终主表。

本地源码/数据preflight验收后可提交。32PPU实际性能和稳定性仍需检查首段。

注意：平台`BASH_ENV`会把环境变量`MODE`覆盖成`CUDA`，
本轮dry-run发现并已用`TIMELINE_MODE`避开，防止再出现模式选错却启动训练。

## 5. 产物

```text
/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline/
  real_smoke_01/
    RESULT.json
    admission_features.pt        # 仅27条数值验收，不是正式训练数据
    *_smoke.pt
    source_snapshot/
  raw_runner_single_01/
    recipe.json
    train.jsonl
    internal_dev.jsonl
    checkpoints/
    RESUME_VERIFIED.json
    COMPLETE.json
  tests_ddp_final_01.log
  launcher_tests_01.json
  launcher_dryruns_01.log
```

现有Stage3及eval工作保持原样，所有新代码和实验在独立目录。
