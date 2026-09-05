# PR Mini-Pilot Review 3 — 深度审计与修复记录

> 目的：记录第三轮广泛检查到底发现了什么、为什么是问题、具体怎样修，以及哪些风险仍然只能通过真实数据/真实 upstream smoke 来关闭。

---

# 结论先行

Review 3 后，**核心 DM-ICF 科学设计不需要推倒重来**。

真正需要修复的问题主要来自四类：

1. 设计语义与实现不完全一致；
2. official baseline wrapper 与真实 upstream contract 不完全一致；
3. final100 虽有原则隔离，但缺少足够强的代码级锁；
4. 实验过多时，计算预算与统计 primary family 会失控。

这些问题已经在 `review3-hardening` 分支逐项修复或加硬性 guard。

但仍有三类事项必须在真实实验前通过运行确认：

- 最新 branch CI；
- official upstream + prepared real data 的 CPU/tiny smoke；
- 真实 frozen manifests 的数据审计。

因此：

> **代码框架接近闭环，但真实 pilot 尚未执行。**

---

# A. 数据与划分审计

## A1. final test 冻结顺序

### 风险

如果先抽 1000 Protein/RNA prior pool，再冻结 final100 complex，会让 final100 homolog/family 通过 pretraining pool 泄漏。

### 修复

硬顺序：

```text
complex final100 first
-> complex development
-> purge final-test P30/R80/Rfam/exact from prior candidates
-> sample 1000 Protein / 1000 RNA
```

状态：**已修复**。

---

## A2. multi-chain cluster overlap

### 风险

`P30_A;P30_B` 如果作为一个字符串比较，与只含 `P30_A` 的样本会被误判为无重叠。

### 修复

所有 cluster/family 字段先拆成 constituent labels；任一共享 P30、R80 或 Rfam 就连成同一 connected component。

状态：**已修复并有 split tests**。

---

## A3. RNA strict split 不能在 Rfam 与 R80 二选一

### 风险

有 Rfam 时若不考虑 R80，可能漏掉 sequence-near RNA；有 R80 又不能替代 family-level isolation。

### 修复

两者取 union constraint：任一 Rfam/R80 重叠均不得跨 strict component。

状态：**已修复**。

---

## A4. canonical interface 不能由模型 graph 定义

### 风险

旧 runtime 使用 8 A + neighbor cap PR graph 决定 `interface=True`；这会使 interface NLL/recovery 的标签依赖我们自己的 receptive field。

### 修复

screening 阶段从 full-heavy-atom <= 6.0 A 固定 canonical interface residue IDs；runtime graph 只负责 message passing。

保存：

```text
protein_interface_residue_ids
rna_interface_residue_ids
canonical_interface_cutoff_angstrom
canonical_interface_definition
```

`load_complex_row()` 必须读取这些 IDs；缺列直接 fail，绝不 fallback 到 PR graph。

状态：**已修复**。

---

## A5. post-freeze local homology audit

### 风险

P30/R80 clustering 依赖 coverage/聚类定义，仍可能有局部 domain-level近同源不直观。

### 修复

新增只读 MMseqs local-search audit：final100 vs development，训练前报告 best identity/coverage。

如果训练前发现明显 near duplicate：version 成 `pilot_v2`，而不是 final metrics 后偷偷改 split。

状态：**代码已加入，待真实 manifest 执行**。

---

# B. 结构表示与模型合同审计

## B1. RNA node dimension mismatch

### 风险

adapter 实际 RNA node feature 为 20 维，旧 trainer 曾写 21，真实训练会在线性层直接崩溃。

### 修复

feature dimension 由 `gemmi_adapter.feature_dimensions()` 单一来源提供；model 不重复硬编码。

状态：**已修复并有 contract test**。

---

## B2. PR rich geometry 深度不足

### 风险

早期实现只覆盖较少 RNA atoms，未达到我们定义的 LigandMPNN-like rich interaction geometry。

### 修复

Protein：

```text
N / CA / C / O / virtual-CB
```

RNA：

```text
P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' O2' C1'
```

即 5 x 12 = 60 atom-pair distance families，再加 missing masks、双方 local-frame displacement 和 relative rotation。

状态：**已实现**。

---

## B3. native RNA base identity leakage

### 风险

N9/C8/N1 等天然 base atoms 会直接泄漏 A/U/G/C。

### 修复

structural prior 与 rich PR input 仅使用 sequence-neutral backbone/sugar set，不将 native identity-specific base atoms作为结构先验输入。

状态：**代码合同存在，仍需真实数据 smoke 验证 modified residues/atom naming coverage**。

---

# C. C / DeltaC / alpha 可解释性审计

## C1. C 不应由 empirical PMI 初始化

最终科学决定：

```text
C0 = small random
```

PMI 只做 post-hoc / fixed-statistical-control。

状态：**已实现**。

---

## C2. C 训练不应 hard-example weighting

### 风险

如果 Stage-C 后半程用 hard-context oversampling，C 就不再是 clean population anchor。

### 修复

Primary Stage C：

```text
hard_context_fraction_late = 0
```

状态：**已修复**。

---

## C3. Alpha stage 不应继续重写 DeltaC/q

### 风险

如果 Alpha stage 同时训练 q、DeltaC、alpha，三者会重新分配解释权。

### 修复

Primary Alpha：

```text
freeze C
freeze q
freeze DeltaC
freeze P/R priors
train relevance residual + tau only
```

状态：**已修复并加测试**。

---

## C4. Joint stage 不应让 C 漂移

### 风险

如果 final Joint low-LR 继续更新 C，Stage-C heatmap 与 final model 实际 global anchor 不再一致，C-vs-PMI 难解释。

### 修复

Primary Joint：

```text
C frozen
Delta/alpha/context heads adapt
pretrained encoders gradual-unfreeze
lambdas learn
```

状态：**已修复并有 C-frozen test**。

---

## C5. DeltaC population mean 可能重新定义 global rule

### 风险

即使 C frozen，DeltaC 可能在所有 edge 上有一个大致相同的 mean shift。

### 修复

不拍脑袋增加 penalty，而是在 final100 前做 development-only audit：

```text
mean_DeltaC
RMS_DeltaC
mean_shift_ratio
alpha-weighted mean_DeltaC
C_eff = C + mean_DeltaC
```

若 mean shift 大：raw C 只称 Stage-1 global anchor，另报告 C_eff。

状态：**audit 已实现，待真实 checkpoint**。

---

# D. 训练公平性审计

## D1. Scratch joint gradual-unfreezing handicap

### 风险

随机初始化 model 如果也 gradual-unfreeze，会冻结随机参数，人为做弱 scratch baseline。

### 修复

Scratch：all parameters trainable from step 0。

此外发现一个更隐蔽问题：dual-prior checkpoint 在未训练 Delta/alpha 时，这两个 head 也仍是 exact zero，不能仅凭零值推断 scratch。

因此 internal control 使用显式：

```python
scratch_joint=False
```

真正 scratch 使用显式：

```python
scratch_joint=True
```

并添加 explicit-pretrained-zero-head regression test。

状态：**已修复**。

---

## D2. partner-blind 不能强行走 C/Delta/Alpha

### 风险

partner-blind forward 强制 cross correction=0，C/Delta/alpha 不影响输出；这些阶段会没有科学意义甚至产生 detached loss。

### 修复

partner-blind：

```text
dual priors -> partner-blind Joint adaptation -> full1000 refit
```

geometry-only：

```text
dual priors -> C -> Delta -> Alpha -> Joint
```

并对 control loss 加 `requires_grad` fail-fast。

状态：**已修复**。

---

## D3. full1000 refit schedule 被压缩

### 风险

Development best=20/150，旧 refit 20 epoch 却将完整 curriculum/cosine/unfreezing 走到100%，等价于换训练算法。

### 修复

refit 运行 N epoch，但 horizon 保持 development max epochs。

记录：

```text
selected_epoch_count
schedule_horizon_epochs
schedule_progress_at_stop
```

状态：**主模型 + internal controls + C-context control 已修复**。

---

# E. Joint validation / inference 审计

## E1. simultaneous full-mask validation 不是真 joint metric

### 风险

两侧全部 unknown 时，interaction contribution=0，一次 forward 主要测 structural prior。

### 修复

Joint checkpoint selection 改为 deterministic teacher-forced sequential pseudo-NLL：

- current/future unknown；
- 已经过位置用 native token revealed；
- mixed + Protein-first + RNA-first；
- log20/log4 + PI/PN/RI/RN group balance。

状态：**已修复**。

---

## E2. final sampling battery 算力爆炸

### 风险

64 candidates x 多 SPIR x 多 order x 3 seeds x autoregressive full forwards，评估成本可能高于训练。

### 修复

Tier A/Tier B：

- Tier A：全部 primary seeds x final100，只跑核心；
- Tier B：analysis_seed only，ablation candidate budget=16；
- final100 前先用 development representative complexes profile sec/candidate/GPU memory；
- budget frozen 后才允许解锁 final100。

状态：**已实现安全 profiler 与 protocol lock**。

---

# F. Official baseline 审计

## F1. ProteinMPNN CLI / data-root mismatch

### 风险

早期 wrapper 传过 upstream 不支持参数，且 data root 曾指向错误层级。

### 修复

严格对照 pinned `training.py`：

- prepared root 直接传 `path_for_training_data`；
- 删除不存在参数；
- `mixed_precision True`；
- seed 用 local runpy wrapper 注入；
- 不修改 upstream checkout。

状态：**已修复；真实 prepared data preflight 尚待执行**。

---

## F2. ProteinMPNN worker nondeterministic no-arg NumPy seed

Pinned upstream `worker_init_fn` 使用 `np.random.seed()` 无参数。

### 修复

local wrapper 在不改 upstream 文件的情况下，按 seed/worker counter 替换该 no-arg 行为。

状态：**已实现**。

---

## F3. NA-MPNN training entrypoint

真实 pinned contract：

```text
python na_run.py config.json
```

### 修复

wrapper 只传一个 positional JSON，output 写 JSON 的 BASE_FOLDER；final checkpoint 读取 `BASE_FOLDER/last.pt`。

状态：**已修复**。

---

## F4. NA RNA A/U/G/C column order

### 风险

旧 exporter 可能按 DA/DC/DG/DT = A/C/G/U 读取，导致 U/C 对调。

### 修复

canonical project order：

```text
A U G C
```

shared token：

```text
DA DT DG DC
```

状态：**已修复并回归测试**。

---

# G. 统计设计审计

## G1. 不能把几十个实验都定义为 primary

### 风险

全部 Holm：极端保守；论文故事发散；还增加 post-hoc 解释空间。

### 修复

Primary confirmatory family 固定 H1-H4：

```text
H1 full vs dual priors
H2 full vs both partner-blind & geometry-only
H3 contextual field vs C-only
H4 partner scramble interface DeltaNLL
```

其余全部 secondary/exploratory/descriptive。

状态：**已实现 registry + confirmatory engine**。

---

## G2. residue 不能作为统计独立单位

### 修复

所有 primary inference：

```text
position -> complex
seed -> per-complex seed mean
complex -> paired statistical unit
```

H1-H4 one-sided directional paired tests + bootstrap CI；Holm only H1-H4。

状态：**已实现**。

---

## G3. H2 是“同时优于两个 controls”

### 修复

intersection-union：

```text
H2 p = max(p_vs_partner_blind, p_vs_geometry_only)
```

只有两个 component 都支持 full better，H2 才成立。

状态：**已实现**。

---

# H. Final100 protocol lock 审计

这是 Review3 最重要的工程强化之一。

过去原则上说“final100不能调参”，但仍可在某些脚本中训练完直接 evaluation。

现在 formal workflow 分开：

```text
primary train/refit (no final100)
controls train/refit (no final100)
DeltaC dev audit
runtime dev profile
freeze protocol lock
explicit final100 evaluation
```

`EVALUATION_PROTOCOL_LOCK.json` 锁定：

- config SHA；
- final100 manifest SHA；
- runtime profile SHA；
- PRIMARY_TRAINING_READY SHA；
- CONTROL_TRAINING_READY SHA；
- B/C/E/F checkpoint SHA；
- partner-blind / geometry-only checkpoint SHA；
- seeds；
- candidate budget；
- H1-H4。

状态：**已实现**。

Final evaluator 验证 hashes 后才执行，且不得训练/选择。

---

# I. 下载与 reproducibility 审计

RCSB：

- concurrent bounded threads；
- deterministic rank；
- retry/backoff；
- atomic `.part`；
- checksum；
- failures table；
- network completion order 不改变 manifest order。

Rfam：

- requested/resolved URL；
- UTC timestamp；
- bytes；
- SHA256；
- CURRENT 的 content SHA 作为主要 immutable identifier。

状态：**已实现，待真实网络执行**。

---

# J. 当前仍然存在的“真实运行才能关闭”的风险

这些不是当前代码里已知的科学 bug，而是不能靠静态审查完全证明的风险。

## J1. upstream real-data loader smoke

必须用真实 prepared Protein/RNA sample 验证：

- ProteinMPNN official loader；
- NA-MPNN PDB parser + preprocessed arrays；
- first train batch；
- first validation batch；
- checkpoint save/read。

CPU preflight 能证明 CLI/layout contract，但不能完全替代真实 loader smoke。

## J2. modified residue / unusual chain naming

shared canonical vocabulary 已统一，但真实 PDB 中可能出现意外 CCD、altloc、insertion code、multi-model情况。

原则：fail loudly + rejection table，不得 silent truncation。

## J3. strict component exact-100 feasibility

P30/R80/Rfam connected-component size分布可能使“正好100 final”难以组成。

当前代码会 fail 并要求扩大 candidate universe 或更换预先冻结 seed；绝不拆 component。

## J4. final compute budget

已有 profiler，但实际 autoregressive cost 取决于最终 length distribution。

必须 development-only profile 后再 lock budget。

## J5. mini-pilot statistical power

100 strict-OOD complexes 对很小 effect 未必有足够 power。

因此重点报告 effect size + CI，不把 `p>0.05` 自动解释为“机制不存在”。

---

# K. Review3 完成标准

代码框架可判定 Review3 完成，需要：

- [ ] review3 branch compileall PASS；
- [ ] pytest PASS；
- [ ] config dead-key audit PASS；
- [ ] Ruff correctness PASS；
- [ ] all hardened CLI `--help` smoke PASS；
- [ ] PR CI PASS；
- [ ] Review3 docs 与代码命令一致；
- [ ] 无 formal workflow 能在 protocol lock 前直接打开 final100。

真实 pilot 完成，需要额外：

- [ ] frozen real 1000/1000/1000+100 data；
- [ ] all data audits PASS；
- [ ] official baseline real-data preflight/smoke PASS；
- [ ] 3-seed primary full1000 refit；
- [ ] H2 controls full1000 refit；
- [ ] runtime profile + protocol lock；
- [ ] final100 Tier A/B；
- [ ] external baseline final100；
- [ ] H1-H4 statistics；
- [ ] all provenance archived。

---

# 最终科学判断

Review3 后最值得保留的不是代码数量，而是下面这个结构：

```text
strong single-molecule structural priors
+
clean global C anchor
+
contextual DeltaC
+
separately learned alpha relevance
+
strict joint coordination
+
locked bilateral OOD evaluation
```

这已经是一条足够清楚、能做消融、能做机制验证、也能被反驳的科学路线。

后续最重要的工作不是继续“设计更复杂模型”，而是让真实 1000/1000/1100 数据和真实 GPU 结果验证它。
