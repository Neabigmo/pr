# Adapter V2 grouped cross-validation

正式运行日期：2026-09-17  
结果目录：`results/adapter_v2_20260917/`

## 目的与边界

本轮只在 1,000 个 development complexes 上研究 Adapter V2 的结构性修复，不读取冻结的 final test。测试集没有参与训练、checkpoint 选择或消融 promotion gate。

C0 是旧 attention/residual 路径的匹配基线；为遵守本轮固定邻域协议，C0 也使用 R→P `K=8`、P→R `K=12`，因此它不是此前统一 `K=51` 的历史 B1 结果的复刻。C1–C4 每次只增加一个预先指定的改动：

| 实验 | 增量 |
| --- | --- |
| C0 | 旧路径、固定方向性 K |
| C1 | common-denominator null softmax + sequence-independent attention |
| C2 | C1 + partner-centered residual |
| C3 | C2 + modality-specific normalization/projector |
| C4 | C3 + conservative sigmoid gate，初始化为 0.1 |

所有实验保持 G2、A2、radius=14.357456 Å、R→P K=8、P→R K=12、batch size=16、prior frozen、prior-normalized loss 和相同优化配置。

## 交叉验证协议

- 3-fold grouped CV；每折约 667 train / 333 validation complexes。
- 分组使用 Protein P30、RNA R80、Rfam 的 bilateral connected components，并按 complex 数、Protein/RNA 长度和 interface pair 数做近似平衡。
- checkpoint gate 要求两个方向的 interface NLL/prior NLL 比例都 `<1`，并且两个方向 native interface NLL 都优于 token-off。
- 选择分数以两个方向中较差的 ratio 为主；gate 不满足时施加惩罚。
- 每折均保存 `best.pt` 和 `last.pt`，汇总只读取验证集选择的 best。

## 正式结果

| 实验 | 通过折数 | promotion | rP | rR | native−token-off P | native−token-off R |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| C0 | 2/3 | pass | 0.932438 | 0.977906 | -0.000669 | -0.003733 |
| C1 | 3/3 | pass | 0.938800 | 0.972893 | -0.061755 | -0.007695 |
| C2 | 2/3 | pass | 0.951615 | 0.975658 | -0.117834 | -0.026799 |
| C3 | 2/3 | pass | 0.952814 | 0.974760 | -0.115518 | -0.028121 |
| C4 | 3/3 | pass | 0.956920 | 0.967011 | -0.093956 | -0.033358 |

因此 C4 满足预设 promotion gate，可以进入下一阶段 reciprocal refinement。C4 的平均 learned gate 约为 Protein 0.11、RNA 0.43；其余逐折指标、null weight、delta RMS、距离分层和 prior-confidence 分层见 `results/adapter_v2_20260917/C4/cv_summary.json`。

## 重要限制

native−permutation 的平均差值在本轮大多接近 0（C4：Protein 0.000428，RNA -0.002745）。这意味着：

1. C4 通过的是预注册的 prior-improvement + token-off gate；
2. 当前结果尚不能宣称模型已经稳定利用了正确 partner 的 identity；
3. reciprocal refinement 必须继续保留 partner permutation 诊断，不能只报告 recovery 或 token-off 结果。

本轮没有读取 final test，因此上述结论是 development grouped-CV 结论，不是冻结测试集结论。

## 文件说明

- `results/adapter_v2_20260917/summary.json`：五组实验的完整汇总。
- `results/adapter_v2_20260917/C0`–`C4/cv_summary.json`：每组的逐折、平均指标和协议副本。
- `results/adapter_v2_20260917/folds.json`：实际使用的 development grouped fold 划分。
- 权重、逐 epoch `metrics.jsonl` 和缓存没有纳入 Git，保留在本机运行目录中。
