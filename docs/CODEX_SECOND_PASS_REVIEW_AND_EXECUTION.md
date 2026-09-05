# 给 Codex：PR mini-pilot v3 后续执行说明

> 这份文档替代早先的“第二轮审查待修改清单”。其中列出的 P0 代码问题已经由 v3 审计直接修入仓库。你现在的任务不是重新设计模型，而是**验证这些修复、构建真实数据、执行 GO/NO-GO gate，然后才开始训练**。

## 0. 首要原则

不要做以下事情：

- 不要根据 final 100 的表现改模型、cutoff、SPIR、loss、epoch 或数据筛选；
- 不要自行把 predicted structures 混进本 pilot；
- 不要把 ProteinMPNN/NA-MPNN 当成 partner-coupling 的因果对照，它们是 one-sided structural references；
- 不要改变 `C -> DeltaC -> alpha -> joint` 的参数 ownership；
- 不要把 canonical 6 A heavy-atom interface 与 8 A/capped PR message graph 混为一谈；
- 不要让 RNA base identity atoms 进入 structural-prior input；
- 不要把模型干预写成“生物学因果”。

任何必要的方案变更都应先停下来记录原因并开新 experiment version。

---

## 1. 先验证当前源码

在干净环境执行：

```bash
pip install -e '.[dev]'
python -m compileall -q src tests tools
pytest
python tools/audit_config_usage.py --config configs/pilot.yaml --repo-root .
pr-pilot --help
python tools/run_official_baselines.py --help
python tools/evaluate_official_baselines.py --help
python tools/run_pilot_experiments.py --help
python -m pr_pilot.evaluation.full_suite --help
python -m pr_pilot.evaluation.field_audit --help
```

要求：

```text
compileall             PASS
pytest                 PASS
config unknown/dead    0
all CLI smoke tests    PASS
```

若失败，先修源码，不得开始数据下载或 GPU 训练。

---

## 2. 验证 pinned official baselines

当前 `third_party/LOCK.json` 固定：

- ProteinMPNN: `dauparas/ProteinMPNN` @ locked SHA;
- NA-MPNN: `baker-laboratory/NA-MPNN` @ locked SHA.

执行：

```bash
python tools/preflight_official_baselines.py \
  --repo-root . \
  --third-party-root third_party/checkouts \
  --out artifacts/preflight/baselines.json
```

重点核实：

1. ProteinMPNN `training.py` 的实际 flags 与 wrapper 一致；
2. `path_for_training_data` 指向含 `list.csv + valid_clusters.txt + test_clusters.txt + pdb/` 的 prepared root；
3. NA-MPNN `na_run.py` 以单个 positional JSON 启动；
4. `BASE_FOLDER` 有尾 `/`；
5. NA inference 不含不存在的 `--rna_backbone_noise`；
6. NA PPM 的项目字母顺序为 `AUGC = DA,DT,DG,DC`。

不要修改 upstream checkout 来让 wrapper “跑通”。如果 upstream API 与预期不一致，应修改我们的 adapter/preflight 并记录。

---

## 3. 构建真实数据

严格按 `docs/DATA_PIPELINE.md` 执行。

必须保留：

- RCSB query JSON；
- download manifests + failure logs + SHA256；
- screening eligible/rejected tables；
- MMseqs2 / Infernal / Rfam 版本与日志；
- annotated candidate tables；
- frozen manifest metadata。

### 特别核查 v3 新规则

- complex Protein individual length <= 1000；
- complex RNA individual length <= 500；
- canonical interface = clean full-heavy-atom 6 A；
- final 100 先冻结；
- prior pools 再 purge final-test exact/P30/R80/Rfam；
- RNA extracted-chain view 不得来自 frozen 1,100 complex pool；
- Protein prior 900/100 必须 P30-disjoint；
- RNA prior 900/100 必须 R80 或 Rfam 任一共享都不能跨 split。

然后运行：

```bash
pr-pilot audit-data \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/data_audit
```

任何 error = NO-GO。

---

## 4. 用真实样本做 baseline smoke test

在完整训练前，先从 frozen manifests 中各取少量样本验证 converter 和 upstream reader。

要求：

- ProteinMPNN converter 后 canonical sequence 与 frozen manifest 逐字符相同；
- NA-MPNN converter 后 canonical RNA sequence 与 frozen manifest 逐字符相同；
- 真实 Protein sample 能进入 pinned ProteinMPNN parser/forward；
- 真实 RNA sample 能进入 pinned NA-MPNN parser/forward；
- 不允许某一方法转换失败后单独替换 ID。

将 smoke test 命令、stdout/stderr、样本 ID 和版本写入 `artifacts/preflight/`。

---

## 5. 主模型开发阶段

阶段顺序固定：

```text
Protein prior
-> RNA prior
-> Global C
-> DeltaC
-> Alpha
-> Joint
```

### 参数 ownership 必须检查

```text
Global C : C only
DeltaC   : interaction q + DeltaC only
Alpha    : relevance/tau only
Joint    : C frozen; context field + lambda + gradual pretrained encoder adaptation
```

每个阶段开始时导出 trainable-parameter report。若 ownership 与上面不一致，停止训练。

### Joint validation

checkpoint selection 不再使用 both-full-mask 单次 forward。

必须由：

```text
Protein conditional interface normalized NLL
+ RNA conditional interface normalized NLL
+ sequential teacher-forced joint normalized pseudo-NLL
```

组成。Sequential term 使用固定 validation subset 和 mixed / Protein-first / RNA-first 顺序。

---

## 6. full-1000 refit

不要把 best epoch 当作新的完整 schedule 长度。

正确含义：

```text
max schedule horizon = H
validation-selected best prefix = K
refit on full1000 = replay original schedule epochs 1..K under horizon H
```

检查 checkpoint 中：

```text
selected_epoch_count
schedule_horizon_epochs
schedule_progress_at_stop
```

Primary final report 只允许使用 full-1000 refit checkpoint。

---

## 7. scratch / controls 公平性

Scratch joint 必须：

```text
random initialization
all encoder blocks trainable from step 0
```

不能套 pretrained gradual-unfreezing。

Same-data controls 至少保留：

- dual structural prior only；
- global C；
- +DeltaC；
- +alpha；
- full joint；
- partner-blind；
- geometry-only capacity control；
- fixed empirical-PMI reference；
- C backbone-context control。

参数量、数据 IDs、训练 seed、development/final-refit规则都记录。

---

## 8. final 100 的执行纪律

打开 final 100 前冻结：

```text
resolved config
hypotheses.yaml
all checkpoint policies
SPIR settings
candidate budgets
analysis seed
baseline SHAs
manifest checksums
repository SHA
```

所有三个主 seed 做 core evaluation。

只有 `evaluation.analysis_seed` 做重量级：

- partner perturbations；
- order sensitivity；
- SPIR grid；
- candidate diversity；
- full interpretability/robustness battery。

Heavy ablation 默认 16 candidates/cell；primary sequence-design budget 64 candidates/complex。

不得因为 heavy battery 的结果改变主模型。

---

## 9. 统计报告

Primary confirmatory family 以 `configs/hypotheses.yaml` 为准，只对该 family 做 Holm。

统计单位：

```text
biological complex
```

不是 residue/token，也不是 seed×complex 当独立观测。

报告：

- complex-level paired effect；
- 10,000 paired bootstrap CI；
- Holm-adjusted primary p；
- seed stability；
- raw + adjusted values；
- secondary/exploratory analyses 分开。

---

## 10. 解释性报告

运行：

```bash
python -m pr_pilot.evaluation.field_audit \
  --config <resolved-config> \
  --checkpoint <final-refit> \
  --manifest manifests/pilot_v1/complex_dev.tsv \
  --out artifacts/interpretability/delta_c_drift
```

至少保存：

```text
C_stage1_anchor
DeltaC_mean
DeltaC_alpha_weighted_mean
C_eff_unweighted
C_eff_alpha_weighted
DeltaC mean-shift ratios
```

如果 mean DeltaC 明显不为零，正文必须把 raw C 称作 Stage-C global anchor，而不是最终全局平均 field。

PMI 来自 independent heavy-atom contact analysis，不来自 capped PR graph。

---

## 11. 不要临时“优化”的内容

本 pilot 暂不做：

- predicted-structure augmentation；
- family-aware replacement sampling；
- dynamic token packing；
- automatic PCGrad；
- alternative biological assembly enumeration。

这些是 full-scale 阶段的问题。

当前 loss 对 Protein/RNA、interface/non-interface 做平衡，但没有再对每一条 individual chain 等权。**不要在论文里写成 per-chain balanced loss。** 本 pilot 做 single-chain vs multi-chain stratified secondary analysis；若 full-scale 多链样本很多，再正式设计 per-chain balance。

---

## 12. 每轮执行后必须输出的汇报

Codex 每完成一个 milestone，都要更新一个短报告，包含：

```text
commit SHA
resolved config SHA256
manifest SHA256
软件/GPU版本
完成/失败样本数
训练阶段与 checkpoint
validation metric
已触发的异常或偏离
下一步 GO / NO-GO
```

禁止用“整体正常”“基本完成”代替可审计证据。
