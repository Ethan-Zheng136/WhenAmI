# WoG 与已实现的 WhenAmI A/B/C：信息流对照

2026-10-07 UTC。仅画已有实现，不包含新提出的动作条件执行时间线。
`→`表示前向；“对齐loss”连接为训练监督，不是向动作头输送真实未来。
所有动作DiT还读取带噪动作/flow time，图中为简洁省略。

## 0. 阶段关系

```text
WoG:
Stage I：学习真实未来条件
       └→ Stage II：撤掉动作头的未来条件输入，辅助预测未来条件

WhenAmI:
A：学习按时间组织的动作条件（pooled / timeline 对照）
├→ B0/B1：只看历史+当前，读当前条件；B1额外对齐教师当前条件
└→ C-aux/C-read：只看历史+当前，预测当前和future(t+4)条件
```

B和C是从A出发的两条学生路线，**不是A→B→C串行训练**。
旧B使用A10k；新B使用A50k；C使用A50k。
A不等于旧项目“Past Stage2”，这里讨论的是当前Timeline-A/B/C实验。

## 1. WoG

### Stage I：用真实future塑造动作条件

```text
当前图像 + 指令 → VLM → z_t ───────────────┐
                                         │
真实future → 冻结DINO/Wan → 可训练QFormer → C_F
                                         │
                              DiT(z_t, C_F)
                                         │
                                      当前动作
                                         │
                                     action loss
```

action loss训练可学习的未来条件编码器和动作路径，
不是靠单独重建未来图片来决定C_F包含什么。

### Stage II：未来是辅助监督，不直接输入动作头

```text
当前图像 + 指令
        │
       VLM
        ├→ cognition z_t → DiT → 当前动作 → action loss
        │
        └→ 查询/预测头 → predicted C_F ──┐
                                        ├→ 对齐loss
真实future → 冻结的未来条件编码器 → C_F* ──┘
                 [仅训练期]
```

关键：Stage II 的predicted C_F也不直接输入动作DiT。
部署的动作路径是当前图像/指令→VLM cognition→DiT。
不能把WoG Stage II画成“预测future条件→注入DiT”。
依据：WoG §3.2–3.3，及本地参考代码的非temporal、`train_venc=False`路径。

## 2. WhenAmI-A：从一个未来摘要，变为每个时间点的动作条件

```text
真实past      真实current      真实future
   │              │               │
每帧独立编码：VLM cognition + 冻结DINO/Wan + 可训练local QFormer
   │              │               │
 (h_p,z_p)      (h_t,z_t)        (h_f,z_f)
   └──────────────┼───────────────┘
                  │
        实际时间索引 + 时间条件器
                  │
          C_p     C_t      C_f
           │       │        │
           └→ 同一个共享DiT ←┘
                  │
      分别预测从p/t/f出发的动作块
                  │
          三个节点的动作FM监督
```

每个节点的动作读出同时读取其condition与对应cognition。
训练交替使用完整观测/遮住中间节点；隐藏节点仍有对应动作监督。
A可以用真实future，是特权预训练/教师阶段，不把其完整观测指标当作因果部署指标。

### A中pooled与timeline的唯一核心区别

```text
pooled:
过去16 tokens ─┐
当前16 tokens ─┼→ 带时间信息的跨帧平均 → 聚合memory → 目标时刻query读取
未来16 tokens ─┘

timeline:
[过去16 tokens | 当前16 tokens | 未来16 tokens]
                         │
          保留各时刻的独立证据与相对时间
                         │
              目标时刻query分别选择性读取
```

pooled也有时间编码、独立目标query和目标节点自身局部特征；
聚合memory会重复到匹配的计算长度，但重复不恢复被平均掉的逐帧证据。
两组参数量、输入与训练配方匹配。不是“有future vs 没future”。

## 3. WhenAmI-B：把教师的当前节点知识转给因果学生

```text
固定A checkpoint
      ├→ 模型参数初始化学生（学生可训练）
      └→ 独立冻结教师

训练期教师：
真实past + current + future → 冻结A → C_t*
                                      │
                                    对齐loss ←───────────────┐
                                                            │
因果学生：                                                  │
真实past + current → 每帧编码 + 时间条件器 → C_t^S            │
                         │                   ├→ 辅助头 P ────┘
                         │                   │
                         └→ 当前z_t          └→ DiT(z_t,C_t^S)
                                                      │
                                                   当前动作
                                                      │
                                                  action loss
```

- B0：仅action loss；辅助头不通过教师loss训练。实际训练仍计算教师用于匹配计算/诊断。
- B1：action loss + 0.1 × cosine_distance(P(C_t^S), stopgrad(C_t*))。
- 对齐的是教师利用真实future形成的**当前节点条件**，不是future节点条件。
- 辅助头P不进入DiT；DiT直接使用学生当前condition，16个condition tokens。
- 部署不需要教师、真实future或辅助头输出；past与current都是合法输入。
- 与WoG Stage II的相似点：辅助预测训练共享表示；不同点：B保留历史和显式当前时间条件读出。

## 4. WhenAmI-C：显式预测future节点，再测试动作是否应该读它

```text
训练期教师：
真实past + current + future(t+4)
                 │
               冻结A50k
                 │
        C_t*           C_(t+4)*
          │               │
          └── 两个节点的对齐loss ──┐
                                  │
因果学生：                        │
真实past + current                │
          │                       │
        每帧编码                  │
          │                       │
   时间查询 [t, t+4]              │
   两个query都只读取≤t的真实观测   │
          │                       │
      C_t^S       predicted C_(t+4)^S
          └──────────┬────────────┘
                     │
          按C-aux/C-read选择动作条件
                     │
              DiT + 当前z_t
                     │
                  当前动作
                     │
                 action loss
```

### 两种读出

```text
C-aux:
   [ C_t^S | C_t^S ] → DiT → 当前动作
   16 tokens + 16 tokens
   future预测只参与辅助对齐，不直接进入动作读出

C-read（20k终点）:
   [ C_t^S | predicted C_(t+4)^S ] → DiT → 当前动作
   16 tokens + 16 tokens
   action loss也有通向预测future条件的直接梯度路径
```

两组相同的损失：

```text
L = L_action
  + 0.1 × mean(当前节点对齐, future节点对齐)
```

实际额外条件为`current + read_scale*(future-current)`；
C-aux的read_scale恒0，C-read前1000步从0升到1。
两组条件长度都为32，当前cognition读出相同。
C没有B的独立forecast_head；对齐的是直接预测的两组节点condition。

## 5. 最后一句话区分

```text
WoG-I：真实未来 → 一组动作相关未来条件
WoG-II：辅助预测这组未来条件，动作只读VLM cognition

A：真实前后文 → 每个时间点各自的动作条件
B：只看过去/当前，学习教师对当前节点的理解
C：只看过去/当前，显式预测未来节点，检验其直接读入动作的价值
```

截至本次图示：

- A50k：timeline相对pooled完整观测动作MSE低4.52%、隐藏节点低17.74%；
  这是特权DEV结果。
- 旧A10k→B：用户回传SIMPLER中有部分闭环收益，不是所有机器人全面超过WoG。
- 新A50k→B及C四组：20k训练/离线评测完成，本轮新增SIMPLER结果仍待回传。
- C不是动作条件世界模型，也没有执行后修订旧事件状态；这些不属于本图的已实现内容。

实现依据：

```text
whenami_timeline_joint/model.py / train.py / evaluate.py
whenami_timeline_b/model.py / train.py / data.py
whenami_timeline_c/model.py / train.py / data.py
WoG_16rank_probe/vla/vla.py
WoG_16rank_probe/action_model/action_model.py
https://arxiv.org/abs/2602.22010
```
