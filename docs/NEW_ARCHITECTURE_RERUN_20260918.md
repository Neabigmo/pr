# 新架构重跑：六条件 holdout 与开发消融

## 协议

- frozen holdout：86 个 complex；只在架构搜索完成后读取。
- prior：复用冻结 ProteinMPNN/NA-MPNN，仅重新生成推理 cache，不重训。
- Adapter：复用 development-CV 已锁定的三个 fold `best.pt`，不重训。
- architecture：G2，R→P K=8，P→R K=12，radius=14.979730606，A0，multiplicative，scalar gate，separate directional edge encoder，modality projector。
- 六个条件是 teacher-forced 的 `Protein/RNA × all/active/interface`，不是自回归采样顺序。
- holdout 不参与架构、K、epoch 或 checkpoint 选择；所有正式搜索使用单一 seed `20260917`。

完整输出位于 `I:\PR_PILOT_SCIENTIFIC\20260918\new_architecture_rerun\`，主结果为 `reports\six_conditions\holdout_six_condition_summary.json`。

## 三个 fold 的 holdout 平均

| 条件 | Adapter NLL | Prior NLL | Adapter/Prior | Adapter recovery | Prior recovery |
|---|---:|---:|---:|---:|---:|
| Protein all | 2.130599 | 2.143110 | 0.994162 | 0.348249 | 0.345772 |
| Protein active | 2.227633 | 2.251683 | 0.989319 | 0.320695 | 0.314849 |
| Protein interface | 2.319108 | 2.382251 | 0.973495 | 0.288792 | 0.277358 |
| RNA all | 1.180160 | 1.155105 | 1.021691 | 0.484459 | 0.502037 |
| RNA active | 1.157697 | 1.126178 | 1.027987 | 0.493798 | 0.516844 |
| RNA interface | 1.182380 | 1.149210 | 1.028864 | 0.484246 | 0.503778 |

`Adapter/Prior < 1` 表示 Adapter NLL 更低。Protein 三种条件均改善；RNA 三种条件均退化。86-complex paired bootstrap 的 NLL 95% CI 也分别完全位于正侧（Protein）或负侧（RNA）。

## 开发集消融

下表来自已完成的新架构 development-CV 顺序搜索，`test_read=false`；它们不是 holdout 选择结果。

| 阶段 | 该阶段选中配置 | Protein ratio | RNA ratio | worst-direction |
|---|---|---:|---:|---:|
| geometry | G2 | 0.956309 | 0.973934 | 0.973934 |
| K | 8/12 | 0.956309 | 0.973934 | 0.973934 |
| radius | 14.9797 Å | 0.956333 | 0.973551 | 0.973551 |
| aggregation | A0 | 0.951340 | 0.972094 | 0.972094 |
| interaction | multiplicative | 0.958388 | 0.969439 | 0.969439 |
| residual | scalar gate | 0.960781 | 0.968299 | 0.968299 |
| edge encoder | separate | 0.959412 | 0.968083 | 0.968083 |

开发 CV 说明各环节在开发分布上有增量，尤其是 A0、multiplicative 和 scalar gate；separate edge encoder 的增量较小但方向一致。它不能抵消 holdout 上的 RNA 泛化退化。

## 解释边界

开发 CV 的 RNA interface ratio 为 0.968083，而 holdout 为 1.028864，存在约 6.1 个百分点的泛化落差；Protein 的落差约 1.4 个百分点。该结果支持“新架构在开发分布有效，但 RNA 方向对 frozen holdout 的分布变化不稳健”，不支持把开发集提升外推成普遍提升。

旧 B3 结果继续保留为 legacy protocol，不与本报告混合。新评估器为 `tools/report_new_architecture.py`，专门读取 scientific checkpoint 的 `metadata.spec`，避免旧 evaluator 按 legacy `config` 误加载新权重。
