# Revision 02：同容量直接历史读出对照（追加实验前固定）

2026-10-03。第一轮结果已保留于`diagnostic_01/`，源码快照也已保存。
发现：合成任务可学习，但关闭revision不影响当前决策，不能宣布新增机制有用。

本轮只增加一个更有判别力的**重训控制**，不改变数据、标签、种子、800步、
batch48、hidden64、lr3e-3和loss。使用seed42/43/44。

- `incremental`：真实新证据的innovation逐次修订旧事件bank。
- `direct_history`：同一filter编码全部合法历史，由当前filtered state直接重读
  每个旧事件；不递推旧revision。给它完整当前hidden而不是只给innovation，避免弱基线。

两者使用相同参数名、参数量、初始权重；每个cutoff对每个旧节点调用同一个大小的MLP，
不通过给baseline更少训练/信息/参数来制造优势。

本轮仍是低维合成合同诊断，不代表8B/OXE/机器人性能。主结论可为“两者都解决该任务”，
或“只状态读出不同但动作一样”，都如实保留。不得按结果改难度再覆盖这次结论。

输出：`matched_controls_02/{incremental,direct_history}/seed*/`。
