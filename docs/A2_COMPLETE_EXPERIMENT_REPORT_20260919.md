# A2 实验总报告（2026-09-19）

本报告汇总当前独立分支已经实际执行的实验，并明确区分 development-CV、历史诊断 holdout 和最终新 blind。结果不把不同协议、不同数据划分或不同模型语义混合解释。

## 1. 最终锁定的 A2 协议

- 冻结 ProteinMPNN 与 NA-MPNN prior；本轮不重训 prior。
- development 使用当前重构后的 860 个 complex，3-fold grouped CV。
- G2 geometry，radius `14.979730606 Å`。
- `K_R→P=8`、`K_P→R=12`，mean aggregation。
- 显式选择矩阵 `M_ij=C+ΔC_ij`，矩阵大小 `20×4`。
- `ΔC` 由 sequence-free structural hidden 与 G2 geometry 生成；native partner token 只在最后矩阵索引处进入。
- A2 动态项使用共享 G2 edge encoder 和 `192→128→80` MLP，最后一层零初始化，确保训练初始状态等于 prior。
- 保留 learned scalar gates；不使用 entropy gate。
- prior-normalized loss：`0.5*(L_P/L_P^prior + L_R/L_R^prior)`。
- 每个正式结果使用一个 seed；A2 稳定性补充实验使用预先固定的 `20260919/20260920/20260921` 三个 development seed。
- 所有选择阶段 `test_read=false`；历史 215-complex 参考集只用于审计，不参与选择。

注意：A2 不是早期 reciprocal E0 的简单改名。A0 是锁定的 E0 baseline（separate directional edge encoder + modality projector）；A2 是新的显式 `C+ΔC` 矩阵模块，动态分支内部使用 shared G2 edge encoder。这是本轮有意的架构调整，已写入 protocol JSON。

## 2. 早期架构搜索与 RNA 稳定性实验

### C0–C4 Adapter V2 搜索

以下是同一旧 development-CV 协议下的平均 interface ratio（Adapter/prior，越低越好）：

| 组别 | Protein | RNA | 说明 |
|---|---:|---:|---|
| C0 | 0.9324 | 0.9779 | 基础 G2、K=8/12、multiplicative |
| C1 | 约 0.94 | 约 0.97 | 加入 prior-normalized/方向性改动 |
| C2 | 约 0.95 | 约 0.97 | 方向性 K 与相关模块组合 |
| C3 | 约 0.95 | 约 0.97 | 进一步结构条件化 |
| C4 | 约 0.96 | 约 0.97 | conservative scalar gate、hidden 等组合 |

C0–C4 在 development 上均有收益，但它们不是最终 A2 的同一概率语义；旧结果保留为审计记录。

### E0–E5 RNA residual 实验

| 组别 | RNA Q1–Q4 平均 ratio | Protein interface ratio | RNA better-complex fraction |
|---|---:|---:|---:|
| E0 | **0.98309** | **0.97279** | 0.67704 |
| E1 entropy gate | 0.99001 | 0.97708 | 0.56902 |
| E2 balanced sampling | 0.98407 | 0.97140 | 0.67400 |
| E3 balanced + entropy | 0.98556 | 0.97859 | 0.59431 |
| E4 learned gate × entropy | 0.98832 | 0.98037 | 0.72345 |
| E5 learned gate × clipped entropy | 0.98352 | 0.97280 | 0.68508 |

结论是 E0 在注册的 development 选择分数上略优于 E5，因此 E0 保留为早期主模型。E1/E3 的退化说明删除 learned scalar gate 后直接乘 entropy 可能放大 residual；这条线没有替换 A2。

## 3. B0–B3 条件比较与旧 holdout 诊断

这组实验使用旧协议，并在模型选择完成后读取 86-complex legacy holdout，仅作为诊断，不能与最终新 blind 混合。

- B0：absolute loss、共享 K=51、shared edge；旧 holdout 上 Protein interface 改善，RNA interface ratio 约 `1.040`，退化。
- B1：prior-normalized、共享 K=51；RNA 仍约 `1.038`，没有解决问题。
- B2：prior-normalized、方向性 K=8/12、shared edge；RNA 约 `1.184`，明显退化。
- B3：prior-normalized、方向性 K=8/12、separate edge；RNA 约 `1.064`，仍退化。

该结果推动了后续 RNA residual 诊断和 A2 显式矩阵设计；不能用它证明 A2 失败，因为它不是 A2 的新协议。

## 4. 新架构六条件旧 holdout

锁定的早期新架构在旧 86-complex holdout 上报告了：

| 条件 | Protein ratio | RNA ratio |
|---|---:|---:|
| all | 0.9942 | 1.0217 |
| active | 0.9893 | 1.0280 |
| interface | **0.9735** | 1.0289 |

这揭示了 development RNA 提升与旧 holdout RNA 退化之间的分布落差，直接促成了重新划分 development、冻结新 blind 和改用显式选择矩阵的方案。

## 4.1 回溯重划分后的 E0 215-complex 诊断

在 A2 之前，曾将 1,075 个合格候选按 fresh bilateral split 重划为 860 development + 215 blind，并在全部 860 development 上重训固定 E0。这个 215-complex blind 是回溯重划分，不是历史上从未接触过的数据，因此只作为诊断，不作为最终论文 blind。

E0 的 P1/P0 ratio 为：Protein all/active/interface `0.993589 / 0.986720 / 0.972615`，RNA `0.991382 / 0.990150 / 0.991463`。RNA interface bootstrap 95% CI 为 `[0.982880, 1.000237]`；RNA quartile 为 `1.097443 / 0.986604 / 0.971843 / 0.961957`，仍然呈现 easy-RNA 退化、hard-RNA 改善的形态。RNA native-vs-shuffle 未通过，native 仅约 41.4% 的 paired comparisons 更好。

这项诊断支持 Protein 侧收益较稳定，但不支持把早期 E0 描述成已经证明了 RNA partner identity 的模型。

## 5. A2 核心消融

在当前 860-complex development protocol、G2、K=8/12、prior-normalized loss 下：

| 变体 | Protein interface ratio | RNA interface ratio | RNA Q1–Q4 score | 作用 |
|---|---:|---:|---:|---|
| A0 | 约 0.9729 | **约 0.9704** | 约 0.9814 | 锁定 E0 baseline |
| A1 | 约 0.9914 | 约 0.9879 | 约 0.9900 | 只有固定 global `C` |
| A2 | **约 0.9715** | 约 0.9705 | **约 0.9799** | `C+ΔC_ij` |
| B1 | 约 0.9715 | 约 0.9703 | 约 0.9798 | 去掉 global `C`，只保留 `ΔC` |
| B2 | 约 0.9726 | 约 0.9718 | 约 0.9792 | 去掉 structural hidden |
| B3 | 约 0.9733 | 约 0.9716 | 约 0.9815 | G2 改为 G0 |

三折正式 A2 CV 的汇总约为：Protein interface ratio `0.9737±0.0021`，RNA interface ratio `0.9723±0.0021`；RNA quartile ratio 为 Q1 `1.0099`、Q2 `0.9887`、Q3 `0.9676`、Q4 `0.9534`。这说明 A2 主要改善较难的 RNA 区间，Q1 仍接近或略高于 1。

## 6. A2 三 seed 稳定性

固定 A2，不重新搜索架构，在 development 上使用 seed `20260919/20260920/20260921`：

- Protein interface ratio：`0.9737 ± 0.0021`。
- RNA interface ratio：`0.9723 ± 0.0021`。
- RNA Q1–Q4：`1.0099 / 0.9887 / 0.9676 / 0.9534`。
- Protein hard gate：3/3 通过。
- partner-use gate：两个 seed 全部通过，第三个 seed 为 3 个 fold 中 2 个通过。
- native-minus-shuffle interface NLL：Protein `-0.00237`，RNA `-0.00091`；development 上 native 优于 shuffle。

## 7. 八组机制实验

这八组均为锁定 A2 的 development-CV 推理期实验，不重训 prior 或 Adapter，`test_read=false`：

| 实验 | Protein interface ratio | RNA interface ratio | 观察 |
|---|---:|---:|---|
| native | 0.95992 | 0.99239 | 基准 |
| global shuffle | 0.95973 | 0.99262 | RNA 轻微变差 |
| interface shuffle | 0.95983 | 0.99217 | RNA 轻微变化 |
| local adjacent swap | 0.95991 | 0.99239 | 近似不变 |
| degree-preserving rewiring | 0.96199 | 0.99275 | Protein 变化更明显 |
| coordinate noise | 0.96038 | 0.99247 | 随噪声轻度变差 |
| relative rigid-body | 0.96075 | 0.99252 | 随相对扰动轻度变差 |
| edge dropout | 0.96074 | 0.99266 | dropout 后 correction 变弱 |
| single-site mutation scan | 0.95992 | 0.99236 | 机制扫描，不作选择 |

这组结果证明了扰动管线、mask、composition 和 degree 审计路径可运行，但 development 上 partner-specific gap 很小；不能仅凭这组实验宣称已经证明强 partner specificity。

## 8. 最终新 blind：P0–P3

从本机 RCSB 候选中先严格筛选，结构上合格 28 个；经 Protein30、RNA80、Rfam family 和 sequence hash 联合审计后，22 个排除，最终冻结 6 个 clean complex。P0–P3 使用同一 frozen cache、active mask 和结构输入：

- P0：frozen priors。
- P1：native A2。
- P2：token-off，理论上应回到 prior。
- P3：20 次 composition-preserving partner shuffle。

P1 相对 P0：

| subset | Protein ratio | RNA ratio |
|---|---:|---:|
| all | 0.9988 | 0.9549 |
| active | 0.9905 | 0.9524 |
| interface | **0.9740** | **0.9500** |

P1 interface recovery 相对 prior：Protein `+0.0071`，RNA `+0.0195`。RNA interface 的 10,000 次 complex bootstrap ratio CI 为 `[0.9167, 0.9741]`；Protein 为 `[0.9446, 1.0125]`。但 blind 只有 6 个 complex，CI 不能替代大样本验证。

P2 的 all/active/interface 指标与 prior 完全一致，说明 token-off 控制实现正确；P1 相对 P2 的 interface NLL 改善为 Protein `0.0641`、RNA `0.0911`。

关键负结果是 P3：

- Protein native-minus-shuffle interface NLL `+0.00162`，native 优于 shuffle 的比例 `2/6`。
- RNA native-minus-shuffle interface NLL `+0.03110`，native 优于 shuffle 的比例 `1/6`。

所以最终 blind 支持“A2 能改善这批样本的 interface prior NLL”，但不支持更强的“已经学习到正确 partner–geometry correspondence”结论。P3 在小样本上反而优于 native，是必须保留的限制与后续风险。

## 9. 额外调整与交付边界

本轮相对早期 E0 的额外调整：

1. 重构为 860-complex development protocol，避免继续沿用早期 991/旧 holdout 混合口径。
2. 引入 A2 显式 `20×4` 动态选择矩阵，并固定 `C+ΔC` 的 prior-preserving zero initialization。
3. 加入 sequence-free hidden contract 审计，禁止 native target token 进入 structural hidden。
4. A2 final refit 使用全部 860 development complex、固定 4 epoch；blind 只在模型锁定后读取一次。
5. 新增 RCSB 候选下载、严格筛选、blind refit 和 P0–P3 evaluator 工具。
6. 新增 checkpoint 原子写入、`best.pt/last.pt/final.pt` 约束和测试/编译/lint 验证。

明确未提交或未上传：权重、cache、日志、原始 CIF/mmCIF、Rfam 数据库、逐样本大 JSONL。仓库只包含核心代码、协议、测试和小型结果汇总。

## 10. 复现入口

- A2 协议：[EXPLICIT_SELECTION_MATRIX_PROTOCOL.md](EXPLICIT_SELECTION_MATRIX_PROTOCOL.md)
- A2 结果汇总：`results/explicit_selection_matrix_20260919/`
- 新 blind 汇总：`results/explicit_selection_matrix_20260919/blind_summary.json`
- blind 锁定记录：`results/explicit_selection_matrix_20260919/blind_lock.json`
- 训练脚本：`tools/run_explicit_selection_matrix.py`
- final refit：`tools/refit_explicit_selection_matrix.py`
- blind evaluator：`tools/evaluate_a2_blind.py`
- 机制实验：`tools/run_a2_evidence.py`
