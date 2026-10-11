# Reader Study 01：立即执行的有界读出适应实验

2026-10-02。用户授权直接推进本地探索，需集群时给用户提交命令。
不启动未限定的大规模架构搜索；优先把可并行的独立对照放到16卡同时完成。

## 四格问题，不是“微调是否更好”

| | 原固定Past DiT | 固定Past DiT + 可训练残差适配器 |
|---|---|---|
| 固定Future训练均值 | frozen_mean | adaptive_mean |
| 真实Future条件C_F | frozen_future | adaptive_future |

- 每个单元同时训练同样的K条件修正模块，原Past条件残差保留。
- 四组均装配相同reader adapter结构；固定列的adapter冻结且输出始终为零。
- 适应列：每个DiT block之后加LayerNorm(无affine)→Linear(width,32)→GELU
  →Linear(32,width)残差；第二层零初始化。原DiT块参数全部冻结。
- 不是LoRA实现，不把训练量“能动一点点读出”夸大成已解冻VLM/完整DiT。
- Future/Past编码器及VLM均冻结；只使用缓存；真实Future是oracle，
  不能把该对照结果当部署策略。没有R、蒸馏、cycle或多步去噪loss。
- 首步四格原行为严格一致（同种子同样本噪声）。只有适配器训练开关/
  K的证据输入不同；原权重文件永不写入。

主对比：`adaptive_future - adaptive_mean`，并看它相对
`frozen_future - frozen_mean`的差上差。负值表示误差改善。
普通微调收益不能算作Future增益。

## 新数据与保留测试

Fractal和Bridge各自：
- train 8192窗口；
- dev 1024窗口；
- sealed test 1024窗口。

合计train16384、dev2048、test2048。每episode最多8窗口。
沿用与旧pilot相同的确定性原始RLDS读取/动作标准化流程，先跳过2048个episode，
远超旧pilot扫描范围。按episode内容SHA256固定划分80/10/10，并按配额采样。
内容ID包含完整标准化动作与若干原始图像/指令，仅用于分组，不输入模型。
开发/训练/测试不允许同episode；测试物理独立存入`sealed_test/`，不编码、不训练、
不评测。旧端点可能见过这些数据，因此不称全模型预训练未见数据。

新数据目录：
`outputs/stage3_reader_study/data_01/{train_dev,sealed_test,shards,split_audit.json}`

读取原始数据在本机CPU的两个独立进程上执行。
编码仅`train_dev`，复用已验证的`stage3.prepare encode`，不改变参考源码。

## 资源与训练预算

**1节点×16卡，一次提交**：
1. 16卡并行提取两个端点的train/dev特征；端点顺序加载。
2. 4格×4种子（42/43/44/45）=16个独立实验，每卡一个，**不是16卡DDP一个模型**。
3. 每组5000步，batch32，LR1e-4，AdamW weight_decay0，grad clip1；
   BF16训练，FP32实际动作采样，动作推理seed固定42。
4. 每1000步开发FM和保存；预先指定最终5000步做完整开发动作评测。
5. 自动汇总每种子结果、种子均值/标准差、episode配对区间和差上差。

集群launcher：
`/home/zhengguantian/WoG_fresh/run_stage3_reader_study.sh`
要求1节点16卡，拒绝旧RUNS目录。不会自动提交平台任务。
保存probe+adapters+optimizer小checkpoint，不重复存32GB端点。
本版没有精确resume；失败后按出错阶段检查，不覆盖已有结果。

## 提交前验证

- 单元：内容ID/分组、零初始化、原权重不变、K与adapter梯度、
  固定控制隔离、compact reload、16个独立任务唯一映射。
- 本机真实Past DiT：四格各20步、batch32训练/保存/重载/小范围动作评测。
  这些只是正确性检查，不从4个开发样本估计收益。
- 新数据端到端小缓存提取验证后才发正式指令。
- 测试集保持封存；最终测试启用必须另行锁定协议。
