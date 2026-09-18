# Priority-8 机制实验协议

## 目的与边界

本轮只做锁定 Adapter 主模型的推理期机制实验，不重新训练 ProteinMPNN、NA-MPNN 或 Adapter。实验仅使用 development 的三折验证样本；冻结的 86 个 legacy holdout 在本轮及模型/参数选择阶段不可读取。

正式预注册 seed 为 `20260917`。20 次 shuffle、rewire 的 3 次重复和 dropout 的 3 次重复是由该 seed 派生的固定扰动实例，不是额外训练 seed。

## 锁定模型与数据

- geometry：`G2`，radius `14.979730606 Å`；
- directional K：RNA→Protein 为 `8`，Protein→RNA 为 `12`；
- aggregation：`A0` mean；
- interaction：multiplicative；
- gate：scalar gate；
- edge encoder：两个方向分开；
- modality projector：启用；
- manifest/cache：`manifest_length_v1` 与 `cache/g3_noise0p0` 的 train+val development 数据；
- checkpoint：三个 development fold 的既有 `best.pt`，只加载并设为 eval；
- 输出：`I:\PR_PILOT_SCIENTIFIC\20260917\reports\mechanisms_priority8\`。

扰动不重算上游 prior hidden/logits；native mask、prior logits 和长度信息保持不变。每条记录逐行写入 JSONL，并以 fold/experiment done marker 支持断点续跑。

## 八组实验

1. `global_shuffle`：Protein 与 RNA partner token 各自 composition-preserving 全局置换；
2. `interface_shuffle`：仅在 interface 位点内置换；
3. `local_swap`：interface 内相邻位点交换；
4. `edge_rewiring`：方向内 degree-preserving edge swap，边数、两端 degree sequence 和 active mask 保持；
5. `coordinate_noise`：σ=`0/0.1/0.2/0.5/1.0 Å`；
6. `rigid_body`：RNA 相对 Protein 平移=`0.5/1/2/4 Å`，旋转=`5/10/20/40°`；
7. `edge_dropout`：p=`0.1/0.2/0.4/0.6`，每个 p 三个固定重复，并保留每个原 active target 至少一条边；
8. `single_site_mutation`：只扫描 interface partner 位点，RNA 使用 A/U/G/C，Protein 使用 20 种氨基酸。

coordinate noise 与 rigid-body 实验在原来选定的 directional edge pair 上重新计算 G2 几何，因此只改变 geometry feature，不因扰动重新选择拓扑；两个方向使用相同的扰动坐标视图。旧实现产生的 fold0/fold1 几何结果已移动到 `legacy_pre_fixed_geometry/` 作为审计记录，不与修复后的正式结果混合。

## 记录指标与审计

每个 complex/扰动实例记录 Protein/RNA 的 all、active、interface NLL、prior-normalized ratio、recovery、相对 native 的 ΔNLL、输出 KL 与 worst-direction ratio。每个实验汇总 complex-level paired bootstrap 10,000 次 95% CI，并记录距离/方向分层所需的逐条结果。

summary 中的 construction-level audit contract 明确区分：

- composition：除 single-site mutation 外保持；mutation 的 composition 改变是设计目标；
- degree：仅 edge_rewiring 报告保持，其他实验标为不适用；
- active mask：所有八组按构造保持；edge dropout 通过保留每个原 active target 的至少一条边实现。

缺少显式审计字段的旧 detail 行不会被伪装成独立重检；summary 会标明 `basis=construction_invariant`。

## 并行与内存

几何/rewiring 使用最多 8 个 CPU worker，按 complex 分片且输出由父进程按完成顺序流式写入；GPU Adapter 推理保持单进程，避免多个 CUDA context 竞争显存。mutation 与 geometry 变体按 batch 流式推理，checkpoint 和 prior 均不复制到 Git。

汇总阶段以流式方式合并 JSONL；complex bootstrap 只保留预注册的六个 interface 指标，避免把数 GB mutation 明细整体载入内存。

## 验收与交付

正式结果必须满足：三折均完成、`test_read=false`、`prior_retrained=false`、`adapter_retrained=false`；pytest、compileall、静态检查和 smoke test 通过。仓库只提交核心代码、测试、协议和小型汇总，不提交 checkpoint、cache、logs 或大规模 detail JSONL。
