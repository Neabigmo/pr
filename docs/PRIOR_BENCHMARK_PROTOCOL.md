# Local structural-prior comparison protocol

本轮对比训练严格复用 prior 当前使用的冻结清单，不重新筛选、不重采样、不改动最终测试 ID：

- 清单目录：`workflow_20260905/manifests/round_20260905_exception_v2`
- Protein：`protein_train.tsv` 900 条、`protein_val.tsv` 100 条
- RNA：`rna_train.tsv` 900 条、`rna_val.tsv` 100 条
- 最终测试：`complex_test.tsv` 86 条；测试时分别只保留 Protein 或 RNA 单侧结构视图
- 最终测试不参与训练、早停选择、epoch/pass 选择或 refit

## 长度约束记录

声明的长度窗口是 Protein 40–2000 aa、RNA 10–500 nt。本轮按用户要求继续使用 prior 的原始冻结数据，因此这些窗口只记录、不追溯排除样本。审计得到：

- Protein pool/train/val：0 条越界，实际范围 40–1524 aa
- RNA pool：46 条越界，实际范围 1–545 nt
- RNA train：40 条越界，实际范围 1–545 nt
- RNA val：6 条越界，实际范围 7–367 nt
- complex_test：0 条 RNA 越界，实际范围 10–70 nt
- complex_train：9 条 RNA 超过 500 nt；complex_val：0 条越界

因此本轮结果的适用范围是“与 prior 完全相同的冻结清单上的比较”，不能表述为“所有训练样本均满足长度窗口”。后续若要严格执行窗口，必须新建实验版本并重新冻结，不能覆盖本轮清单。

## 方法矩阵与统一项

- Protein：本项目 Protein prior vs 锁定 SHA 的 ProteinMPNN
- RNA：本项目 RNA prior vs 锁定 SHA 的 NA-MPNN
- 两个官方实现均从随机初始化开始；上游源码不修改
- 统一使用相同冻结 train/validation ID、相同最终 86 条测试集、0.10 Å 坐标噪声、相同 seed 列表和 validation-only 选择原则
- 本项目 prior 保持原实现的单图单步语义：有效 batch 是 1 个可变长度稀疏图/optimizer step，没有合法的等价 `batch_size` 参数
- 官方 ProteinMPNN 和 NA-MPNN 使用各自上游可表达的 token batching；本地都设为 6000 token budget。该差异写入结果，不伪装成完全相同的 batch size
- ProteinMPNN 的输入长度上限设为 2000，以覆盖当前冻结池中最长 1524 aa 样本，避免因上游默认上限静默丢样本

## 外部 RNA 基线的结构兼容记录

RNA 的 900/100 行仍全部保留在转换 manifest 中；没有按长度、分辨率或实验方法二次筛选。由于锁定的 NA-MPNN 上游在加载时要求 RNA 每个保留残基的 12 个骨架原子都存在且 occupancy > 0.8，外部结构视图按该上游规则同步记录有效长度，避免辅助数组与结构张量错位：

- train：900 个 manifest ID，888 个含至少一个可表示残基；共同步移除 563 个不满足上游骨架条件的残基，12 个 ID 的有效长度为 0
- valid：100 个 manifest ID，100 个含至少一个可表示残基；共同步移除 50 个残基
- 这些是上游结构表示能力造成的有效 token 差异，不是本轮长度筛选，也不回写 prior 原始清单；12 个零有效长度 ID 会保留在输入记录并标记为 NA-MPNN 无可训练 token
- Protein baseline 没有同类结构裁剪；Protein 900/100 全部可转换

因此报告中同时给出 manifest-level 样本数与 baseline-effective token/sample 数，不能把 NA-MPNN 的 888/100 有效输入误报为新的训练集冻结结果。

所有本轮输出集中在 `prior_benchmark_local_20260906`，不写入旧的 workflow artifact 目录；远端训练不受影响。
