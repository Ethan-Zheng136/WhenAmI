# 主训练数据范围修正：OXE混合与两域专项训练必须区分

**最新：27域数据入口及320卡主训练配方已实现，提交说明以
`OXE27_320_SUBMIT.md`为准。27/27真实样本审计、76项测试通过；
全语料转换与修正后的32卡恢复验收仍待提交。下面“尚未实现”是历史说明。**

2026-10-02。用户指出原项目使用OXE，而最新联合训练入口只处理Bridge/Fractal。
核对结果：原Future Stage1和Past Stage2的config均为`oxe_magic_soup_plus`；
该混合名在当前参考实现中包含27个数据集，两个端点的dataset_statistics也覆盖27个。

## 当前入口的实际范围

- `inventory.py`只枚举Fractal 0.1.0和Bridge_orig 1.0.0。
- `raw_data.py`中的图像键处理目前只针对这两域；
  `RawWindows.batch`当前按域等概率抽样。
- 因此任务A是**两域完整训练分区打包**，不是原项目27域OXE混合准备。
- 任务B仍可用于验证8B模型的FSDP训练/保存恢复，但它不验证另外25域的
  数据standardization、相机/语言/动作语义。
- 现有主训练命令不能仅改一个DATA_MIX变量就变成27域训练。

此前选两域的依据是WoG论文附录B.1的SIMPLER配置：
Stage I用OXE，Stage II用Bridge/Fractal；这是具体基准的后训练配方，
不应被表述成整个WhenAmI项目的默认“全量OXE主训练”。

## 主训练目标范围

若要保持我们此前通用OXE训练路线，主训练应使用原来的
`oxe_magic_soup_plus` 27域混合，而不是只保留Bridge/Fractal。
“原项目选定的OXE混合”也不等于整个公开OXE集合的每一个数据集。

准备阶段需要：
1. 从原mixture表和dataset config构造输入，不在代码里只写两域；
2. 严格使用各数据集原standardization、主相机、语言和7D动作映射；
3. 冻结明确的数据版本与TFDS元数据引用的shard列表；
4. 延续原采样规则：mixture multiplier × pinned num_transitions，归一化，
   不能误用27域均匀采样或两个域50/50；
5. 延续既有episode分区，且对新增域建立可审计train/dev/test划分；
6. 逐域读取/窗口/动作语义检查，不能用两域通过代替全部27域已验证。

## 本轮只读核对

所有27域在本地都有dataset_info、对应端点统计及元数据引用的训练shard文件。
这是文件/配置清点，不是所有数据已经解码、清洗或转换成功。
发现`dobbe/0.0.1`目录同时存在256和1016两个物理shard布局；
当前dataset_info选择256个。全域准备必须依元数据选取，不能glob两套后混在一起。

记录：
`outputs/stage3_joint/oxe_scope_audit_20261002.json`

**当前状态：27域清点完成，27域联合训练数据入口尚未实现。**
本轮没有修改训练源码/正在排队的任务，没有删除任何数据，也没有假称OXE主训练已就绪。
任务A不要作为27域主训练前置来提交；若已经提交，保留产物作为两域数据，
不应将其误标为OXE full。任务B可继续做独立工程验收。
