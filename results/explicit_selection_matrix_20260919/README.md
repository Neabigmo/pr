# A2 compact results

本目录只保存可提交的小型汇总结果，不包含权重、cache、日志或原始结构数据。

- `protocol.json`：A0/A1/A2/B1/B2/B3 的当前 protocol。
- `cv_summary.json`：A0/A1/A2 当前 860-complex development CV。
- `a2_three_seed_summary.json`：固定 A2 的三 seed development 稳定性。
- `reports/ablation_B*/cv_summary.json`：B1/B2/B3 消融。
- `reports/a2_evidence/summary.json`：八组机制实验。
- `blind_summary.json`：最终新 blind 的 P0/P1/P2/P3 结果。
- `blind_lock.json`：blind manifest、development split 和 checkpoint hash 锁定记录。

早期回溯重划分的 E0 215-complex 诊断汇总位于
`results/retrospective_resplit_e0/blind_summary.json`；它不是当前 6-complex
untouched blind，也不参与 A2 选择。

最终新 blind 只有 6 个 clean complex：28 个结构合格候选中，22 个因 Protein30/RNA80/Rfam 或 sequence-hash 泄漏排除。该样本量适合 pilot/diagnostic，不足以支持大规模泛化结论。
