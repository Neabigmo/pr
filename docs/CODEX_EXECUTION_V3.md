# 给 Codex：PR Mini-Pilot V3 权威执行规范

> 本文覆盖并取代旧的 `CODEX_SECOND_PASS_REVIEW_AND_EXECUTION.md` 中与执行相关的内容。
>
> **目标不是继续增加模型复杂度，而是严格执行已经冻结的科学合同。**
>
> 核心安全原则：`final100` 在训练、预算评估、checkpoint/controls 冻结完成前保持逻辑锁定。

---

# 0. 任务和不可改变的科学主线

固定 Protein backbone 与 RNA backbone，支持：

1. RNA 已知 -> Protein 设计；
2. Protein 已知 -> RNA 设计；
3. Protein/RNA 联合设计。

核心分解：

```text
sequence preference
=
intramolecular structural prior
+
local cross-molecular selection
```

DM-ICF：

```text
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

- `C`: 20x4 global AA-base compatibility anchor，小随机初始化；
- `DeltaC_ij`: rich PR geometry + Protein/RNA hidden states 产生的 contextual residual；
- `alpha_ij`: geometry-aware relational relevance；
- empirical PMI **不参与 primary C 初始化**。

任何实现、实验或文字修改均不得破坏这条主线。

---

# 1. Pilot 数据合同

目标固定为：

```text
Protein structural-prior pool  1000
RNA structural-prior pool      1000
Protein-RNA complexes          1100
```

Complex：

```text
1000 development
100 immutable final test
```

Development：

```text
900 train
100 validation
```

最终报告 checkpoint：

```text
900/100 只选 epoch/训练停止点
-> 从随机初始化重新开始
-> 完整 1000 development refit
-> 重放 development schedule 的相同前缀
-> 禁止 validation/final100 再次选择
```

`final100` 是 strict bilateral OOD，而不是 IID random test。

---

# 2. Final-test-before-prior-pool 的严格顺序

必须：

```text
experimental complex candidates
-> P30 + R80 + Rfam 联合 component
-> 先冻结 final100
-> 从剩余 complex 抽 1000 development
-> 用 final100 的 P30/R80/Rfam/exact sequence 反向 purge Protein/RNA prior candidates
-> 再抽 1000 Protein + 1000 RNA
```

任何顺序倒置都视为数据泄漏风险。

禁止：

- final100 P30 进入 Protein prior pool；
- final100 R80/Rfam 进入 RNA prior pool；
- exact sequence/mother sample 跨 split；
- multi-chain cluster label 作为一个拼接字符串比较；
- mutation/conformer family 被拆散后获得额外权重。

---

# 3. Canonical interface 与模型 PR graph 必须分开

## Canonical biological interface

由 coordinate screening 固定：

```text
full-heavy-atom Protein-RNA minimum distance <= 6.0 A
```

manifest 必须保存：

```text
protein_interface_residue_ids
rna_interface_residue_ids
canonical_interface_cutoff_angstrom
canonical_interface_definition
```

它用于：

- PI/PN/RI/RN loss 分组；
- interface/non-interface recovery/NLL；
- external baseline position mapping；
- final statistics。

## DM-ICF PR message graph

```text
8 A cutoff + max-neighbour cap + rich 5x12 atom-pair geometry
```

只用于 receptive field。

改变 `pr_cutoff_angstrom` / `pr_max_neighbors` **不得改变 canonical interface label**。

---

# 4. Primary 模型训练 ownership

严格按以下顺序：

```text
P -> R -> C -> Delta -> Alpha -> Joint
```

## P

Protein backbone -> AA structural prior。

## R

RNA sugar/phosphate/base-sugar backbone -> A/U/G/C structural prior。

禁止 native base-identity atoms 泄漏。

## C

只训练 global `C`：

- P/R prior frozen；
- DeltaC frozen/off；
- learned alpha frozen/off；
- lambda = 1；
- no hard-context mining；
- no label smoothing；
- experimental complexes only；
- C small-random init；
- 不使用 PMI 初始化。

## Delta

训练：

```text
q_ij interaction encoder + DeltaC
```

冻结：

```text
P/R priors + C + learned alpha
```

DeltaC final output exact zero-init。

## Alpha

Primary 版本只训练：

```text
relevance residual + learnable tau
```

冻结：

```text
P/R priors + C + q + DeltaC
```

这样 C / DeltaC / alpha 才有可辨识的阶段含义。

## Joint

允许：

- interaction/DeltaC/alpha/context decoder 继续协调；
- pretrained encoder gradual unfreezing；
- bounded lambda_P/lambda_R 学习。

**Primary global C 继续冻结。**

---

# 5. Scratch control 不能被 gradual unfreezing handicap

Scratch component 必须：

```text
same architecture
random initialization
all parameters trainable from step 0
```

不要从参数值猜 scratch/pretrained。

任何从 dual-prior checkpoint 开始的 control 必须显式：

```python
configure_stage(..., Stage.JOINT, scratch_joint=False)
```

真正 scratch：

```python
configure_stage(..., Stage.JOINT, scratch_joint=True)
```

---

# 6. H2 两个内部 fairness controls

## partner_blind

目的：回答“收益是不是仅来自 complex-task fine-tuning，而非 partner identity”。

正确路径：

```text
dual structural priors
-> partner-blind Joint adaptation
-> full1000 refit
```

不要强行让它经历 C/Delta/Alpha，因为 cross correction 被强制为 0 时这些阶段会出现无意义/断开的梯度。

## geometry_only

保留：

- partner structure/geometry；
- q/DeltaC/alpha extra capacity；

删除：

- 具体 partner AA/base identity。

路径：

```text
dual priors -> C -> Delta -> Alpha -> Joint
```

用于区分“partner identity coupling”与“只是多看了 partner geometry/多了参数”。

---

# 7. Loss 合同

绝不直接把所有 token loss 相加。

四组先分别 mean：

```text
Protein interface       PI
Protein non-interface   PN
RNA interface           RI
RNA non-interface       RN
```

跨 alphabet：

```text
Protein NLL / log(20)
RNA NLL / log(4)
```

Joint 再进行 P/R 等权组合。

一个 batch/step 对应一个 task，由 sampler 控制 task ratio；不要把三个巨大 task loss 同时硬加。

任何 refactor 必须保留这些不变量并有 unit test。

---

# 8. Joint validation 必须和真实 joint inference 同定义

禁止用“双侧一次 full-mask forward”选 Joint checkpoint，因为 unknown partner contribution=0，会退化成 structural prior 测试。

当前 primary checkpoint metric：

**deterministic teacher-forced sequential pseudo-NLL**：

- current/future design tokens unknown；
- 已经经过的 token 填 native 并变 known；
- 每一步记录 native log probability；
- 固定 mixed + Protein-first + RNA-first；
- PI/PN/RI/RN 等组归一；
- Protein/RNA 用 log20/log4 归一。

只在 frozen complex validation subset 上使用，不接触 final100。

---

# 9. Refit 不能压缩 schedule

如果 development：

```text
max_epochs = 150
best = epoch 20
```

refit 训练 20 epoch，但第 20 epoch 的 curriculum/unfreezing/LR 仍应对应：

```text
20 / 150 schedule progress
```

不是把 0->100% schedule 压缩到 20 epoch。

必须保存：

```text
selected_epoch_count
schedule_horizon_epochs
schedule_progress_at_stop
```

主模型、internal controls、C-context control 都遵守同一规则。

---

# 10. Official baseline 合同

Pinned upstream：

```text
ProteinMPNN
8907e6671bfbfc92303b5f79c4b5e6ce47cdef57

NA-MPNN
9fabc2482092b725e067969fba21297a806b6fda
```

## ProteinMPNN

- `path_for_training_data` 指 prepared root，不是 `/pdb` 子目录；
- 不传不存在的 cluster/seed flags；
- mixed_precision 传 explicit bool value；
- seed 由 repository-local seeded wrapper 注入，不修改 upstream checkout；
- 0.10 A backbone noise；
- development epoch 选择后 full1000 restart/refit。

## NA-MPNN

训练入口真实合同：

```text
python na_run.py config.json
```

不是 argparse training CLI。

final checkpoint：

```text
BASE_FOLDER/last.pt
```

RNA probability canonical order：

```text
A U G C
```

shared-token 情况对应：

```text
DA DT DG DC
```

绝不能再次写成 DA/DC/DG/DT。

External baselines 都是 **one-sided structural references**，不能单独证明跨分子 coupling。

---

# 11. 正式运行的安全状态机

## Phase A — code gate

```bash
python -m compileall -q src tests tools
python -m pytest -q
python tools/audit_config_usage.py --config configs/pilot.yaml
ruff check src tests tools --select E9,F63,F7,F82
```

必须全部通过。

## Phase B — real data

严格按 `docs/DATA_PIPELINE.md` / `RUNBOOK.md`：

```text
RCSB discover
-> concurrent deterministic download
-> coordinate screening
-> canonical interface IDs
-> RNA chain-view expansion if needed
-> joint MMseqs + Rfam
-> final100 first freeze
-> purge prior pools
-> 1000/1000/1000 freeze
-> leakage/interface/local-similarity audits
```

## Phase C — official baseline preflight

先 `--prepare-only`，再逐 seed：

```bash
python tools/preflight_official_baselines.py ...
```

必须 PASS 才允许 baseline GPU run。

## Phase D — primary training while final100 remains locked

只用：

```bash
python tools/run_primary_training_only.py ...
```

输出：

```text
PRIMARY_TRAINING_READY.json
```

这个脚本不得读取 `complex_test.tsv`。

## Phase E — internal-control training while final100 remains locked

```bash
python tools/run_internal_controls.py --phase train ...
```

输出：

```text
CONTROL_TRAINING_READY.json
```

不得 evaluate final100。

## Phase F — development-only profiling

```bash
python tools/profile_evaluation_budget.py \
  --manifest manifests/pilot_v1/complex_val.tsv ...
```

确认 GPU-hour budget 后才能继续。

## Phase G — freeze evaluation protocol

必须同时锁：

- base config SHA；
- final100 manifest SHA；
- runtime profile SHA；
- PRIMARY_TRAINING_READY SHA；
- CONTROL_TRAINING_READY SHA；
- B/C/E/F stage checkpoints SHA；
- partner-blind / geometry-only checkpoints SHA；
- analysis seed；
- Tier A/B candidate budget；
- H1-H4。

使用：

```bash
python tools/freeze_evaluation_protocol.py \
  --config configs/pilot.yaml \
  --test-manifest manifests/pilot_v1/complex_test.tsv \
  --runtime-profile artifacts/evaluation_budget_profile/summary.json \
  --training-ready artifacts/pilot_experiments/PRIMARY_TRAINING_READY.json \
  --control-training-ready artifacts/internal_controls/CONTROL_TRAINING_READY.json \
  --out artifacts/EVALUATION_PROTOCOL_LOCK.json
```

## Phase H — explicitly unlock final100

主模型：

```bash
python tools/run_primary_final_evaluation.py ...
```

H2 controls：

```bash
python tools/run_internal_controls.py --phase evaluate ...
```

B/C/E/F component：

```bash
python tools/evaluate_component_controls.py ...
```

Official one-sided baselines：

先在 lock 之后准备 final holdout view，再：

```bash
python tools/run_official_baseline_final_evaluation.py ...
```

所有 evaluator 必须只读 checkpoint，不训练、不选 epoch、不改阈值。

---

# 12. 统计：只保留 4 个 confirmatory hypotheses

## H1

Full DM-ICF vs dual structural priors。

Primary endpoint：canonical-interface normalized NLL。

## H2

Full DM-ICF 必须同时优于：

- partner-blind；
- geometry-only。

作为 intersection-union claim，H2 使用更保守的 component p-value。

## H3

Contextual field (`E_alpha`) vs `C_global_C`。

## H4

composition-preserving partner scramble：

```text
DeltaNLL = NLL_scramble - NLL_native > 0
```

四个 hypotheses 以 biological complex 为统计单位，先在 seed 内/跨 seed 汇总到 complex，再做 paired inference。

Holm **只校正 H1-H4**。

广泛 NLL/recovery/robustness 表属于 secondary/exploratory，不能悄悄扩大 primary family。

---

# 13. Tier A / Tier B

## Tier A：全部 3 seeds × 全部 final100

只跑支撑核心结论的轻量指标：

- conditional P/R per-position probabilities；
- canonical interface/non-interface；
- teacher-forced joint pseudo-NLL；
- partner scramble。

## Tier B：analysis_seed only

重型电池：

- counterfactual mutation；
- DeltaC/alpha；
- empirical heavy-atom PMI；
- coordinate noise；
- edge removal；
- partner hiding；
- geometry permutation；
- order sensitivity；
- SPIR；
- candidate diversity；
- calibration；
- dataset-shift diagnostics。

Candidate budget 必须在 protocol lock 前固定。

---

# 14. DeltaC mean-drift audit

在 final100 之前、development-only 运行：

```bash
python tools/audit_delta_c_drift.py ...
```

记录：

```text
mean_DeltaC
RMS_DeltaC
mean_shift_ratio
alpha-weighted mean
C_stage1
C_eff = C_stage1 + mean_DeltaC
```

如果 DeltaC population mean 很大：

- raw C 只能叫 `stage-1 global compatibility anchor`；
- 可以报告 C_eff；
- **不能看完 final100 后临时加 DeltaC norm penalty**。

---

# 15. 数据下载/资源 provenance

RCSB downloader：

- bounded concurrency；
- deterministic rank；
- retry/backoff；
- atomic `.part`；
- SHA256；
- failures.tsv；
- completion order 不影响 data order。

Rfam 保存：

- requested CURRENT URL；
- resolved URL；
- download timestamp；
- file size；
- SHA256。

content SHA 是主要 immutable reproduction identifier。

---

# 16. Codex 每次正式执行必须输出的 artifacts

至少：

```text
artifacts/data_audit/
artifacts/baselines/
artifacts/pilot_experiments/PRIMARY_TRAINING_READY.json
artifacts/internal_controls/CONTROL_TRAINING_READY.json
artifacts/pilot_experiments/development_audits/delta_c_drift/
artifacts/evaluation_budget_profile/
artifacts/EVALUATION_PROTOCOL_LOCK.json
artifacts/pilot_experiments/evaluation/
artifacts/component_controls/component_ladder_runs.tsv
artifacts/internal_controls/control_runs.tsv
artifacts/baseline_final100/
artifacts/statistics/confirmatory/
artifacts/statistics/exploratory/
```

再保存：

- exact git commit；
- frozen config SHA；
- manifest SHA；
- upstream baseline SHAs；
- environment versions；
- GPU；
- start/finish time；
- failed samples（不得静默丢弃）。

---

# 17. Codex 禁止事项

绝对不要：

- 训练时读取 final100；
- final100 出来后改 architecture/cutoff/noise/mask/loss/SPIR/candidate budget；
- final100 结果不好就删除 target；
- 重新挑 B/C/E/F 或 control checkpoint；
- 用 PMI 初始化 primary C；
- 把 ProteinMPNN/NA-MPNN 说成同信息条件 partner-aware baseline；
- 把 predictor confidence 说成 binding energy；
- 把 partner scramble/alpha ablation 写成 biological causality；
- 把 YAML 中不存在执行路径的字段写进 Methods；
- 把“代码写完”写成“实验做完”。

---

# 18. GO / NO-GO

只有以下全部为 YES 才允许长 GPU run：

- [ ] current commit compileall/pytest/config-audit/Ruff 全绿；
- [ ] exact 1000 Protein + 1000 RNA + 1000 dev + 100 final frozen；
- [ ] P30/R80/Rfam/exact/mother leakage = 0；
- [ ] canonical 6 A interface audit PASS；
- [ ] read-only local similarity audit archived；
- [ ] official baseline converters accept all frozen samples；
- [ ] official baseline preflight PASS；
- [ ] tiny end-to-end stage smoke PASS；
- [ ] no dead runtime-looking config keys；
- [ ] primary hypotheses H1-H4 frozen；
- [ ] final100 未用于任何选择。

只有以下全部为 YES 才允许打开 final100：

- [ ] PRIMARY_TRAINING_READY complete；
- [ ] CONTROL_TRAINING_READY complete；
- [ ] DeltaC development audit complete；
- [ ] evaluation runtime profile complete；
- [ ] evaluation GPU budget accepted；
- [ ] `EVALUATION_PROTOCOL_LOCK.json` exists；
- [ ] config/test/checkpoint hashes locked；
- [ ] no pending training/model-selection decision。

**如果任一项为 NO，停止，不要用“先跑看看”替代协议。**
