# PR Mini-Pilot 新工作区交接文档

**项目仓库**：https://github.com/Neabigmo/pr  
**默认分支**：`main`  
**交接基准提交**：`8cc6c0ffec0494c9d207de5ad7b12085912138c3`  
**交接时间**：2026-09-05  
**用途**：在新的 ChatGPT / Codex 工作区中无缝继续当前 Protein–RNA 联合设计 mini-pilot，不重新讨论已经冻结的科学主线，不重复踩已经发现的问题。

---

# 1. 新工作区先读这一段

这个项目当前**不是从零开始**，也**还没有完全进入正式 GPU 实验阶段**。

已经完成的是：

1. Protein–RNA 固定双骨架联合序列设计的核心科学方案已经基本定型；
2. DM-ICF 的主要代码框架、数据下载/筛选/聚类/冻结、训练、refit、evaluation、internal controls、官方 baseline adapter 等已经大面积实现；
3. 代码曾做过一轮较深自查，修掉了多链 split、预训练泄漏、RNA feature 维度、PR rich geometry、Node24 CI、loss 归一化等一批问题；
4. 仓库已有系统性的实验协议、runbook、自查文档和用户说明；
5. **第二轮更深入审阅又发现一组 P0 问题。它们已经被写成修复方案，但在本交接时点并没有全部落成代码。**

因此，新工作区接手后的第一任务不是继续扩展功能，也不是直接下载数据开 GPU，而是：

> **以 `docs/CODEX_SECOND_PASS_REVIEW_AND_EXECUTION.md` 为 P0 修复清单，将“设计、代码、baseline、validation、evaluation”重新对齐，完成测试后再进入真实数据和 GPU。**

不要把当前仓库描述成“已经完全完成”。更准确的状态是：

> **科学设计基本冻结；工程框架已经较完整；第二轮审计发现的最后一组关键合同问题仍待修复。**

---

# 2. 项目目标

研究目标是在**固定 Protein backbone + 固定 RNA backbone** 条件下，设计一侧或同时设计两侧序列。

三种原生任务：

\[
p(S_P\mid B_P,B_R,S_R)
\]

\[
p(S_R\mid B_P,B_R,S_P)
\]

\[
p(S_P,S_R\mid B_P,B_R)
\]

分别对应：

- RNA 已知 → Protein design；
- Protein 已知 → RNA design；
- Protein + RNA joint co-design。

这不是通用 sequence–structure diffusion，也不是单纯把 ProteinMPNN 和 RNA inverse folding 拼起来。

核心科学假设是：

\[
\boxed{
\text{sequence preference}
=
\text{intramolecular structural prior}
+
\text{cross-molecular selection}
}
\]

即：

> 模型先知道“这个骨架本身喜欢什么序列”，再学习“另一条链在这个局部界面上会怎样改变这种选择”。

---

# 3. 核心模型：DM-ICF

正式核心名称：

> **Dynamic Multiscale Interfacial Compatibility Field (DM-ICF)**  
> 动态多尺度界面相容场

跨分子项：

\[
\boxed{
\Gamma_{ij}=\alpha_{ij}(C+\Delta C_{ij})
}
\]

其中：

## 3.1 Structural prior

Protein：

\[
h_i^P=E_P(B_P)
\]

\[
z_i^{P,struct}=W_Ph_i^P+c_P
\]

RNA：

\[
h_j^R=E_R(B_R)
\]

\[
z_j^{R,struct}=W_Rh_j^R+c_R
\]

Protein 和 RNA 使用两套独立 encoder，不共享主干参数，但投影到统一 interaction hidden space。

## 3.2 Global compatibility matrix C

\[
C\in\mathbb{R}^{20\times4}
\]

含义：总体 AA–base compatibility contribution。

**最终决定：C 不用真实频率或 PMI 初始化。**

采用 small zero-centered random initialization：

\[
C^{(0)}_{ab}\sim \mathcal N(0,\sigma_C^2)
\]

PMI 完全保留为训练后的独立 biological validation。

注意：后续审阅已经指出，训练时不应机械 double-center C 的行和列，因为同一个 C 同时参与 Protein<-RNA 与 RNA<-Protein 两个方向，row/column main effect 可能有实际条件预测意义。更稳妥的做法是只固定真正不可辨识的 global scalar offset；double-centered C 可作为 post-hoc “pure interaction” 可视化。

这一点需要和早期论文稿同步更新。

## 3.3 Context residual Delta C

每条 PR edge：

\[
q_{ij}=G_{PR}(\Pi_Ph_i^P,\Pi_Rh_j^R,f_e(e_{ij}))
\]

\[
\Delta C_{ij}=g_\Delta(q_{ij})\in\mathbb R^{20\times4}
\]

最终输出层 zero-init，因此 Context stage 开始时：

\[
\Delta C_{ij}=0
\]

即严格从 global-C model 出发。

**最终决定：不对 \|Delta C\| 做强幅度正则。**

原因：局部特异识别可能必须大幅改写 global tendency。

## 3.4 Alpha / relational relevance

\[
s_{ij}=-d_{ij}^{eff}/\tau+\Delta s_{ij}
\]

\[
\alpha_{ij}=\operatorname{softmax}_j(s_{ij})
\]

其中：

- 距离提供几何初始 prior；
- \(\Delta s\) zero-init；
- \(\tau>0\) 可学习；
- alpha 不直接读取 AA/base identity；
- alpha 表达 “who matters”，C+DeltaC 表达 “what pairing is preferred”。

## 3.5 最终 logits

Protein：

\[
z_i^P(a)=z_i^{P,struct}(a)
+\lambda_P\sum_{j\in N_R(i)}
\alpha_{ij}^{P\leftarrow R}
[C(a,b_j)+\Delta C_{ij}(a,b_j)]
\]

RNA：

\[
z_j^R(b)=z_j^{R,struct}(b)
+\lambda_R\sum_{i\in N_P(j)}
\alpha_{ji}^{R\leftarrow P}
[C(a_i,b)+\Delta C_{ij}(a_i,b)]
\]

联合生成时，如果 partner token 还不知道，该 edge 的 token-specific interaction contribution 暂时为 0。

---

# 4. Rich Protein–RNA geometry

不能只用一个 residue–nucleotide 距离。

Protein sequence-neutral atoms：

- N
- CA
- C
- O
- virtual CB

RNA sequence-neutral atoms：

- P
- OP1
- OP2
- O5'
- C5'
- C4'
- O4'
- C3'
- O3'
- C2'
- O2'
- C1'

因此主 PR edge 当前目标是：

\[
5\times12=60
\]

组 atom-pair distances，每组 RBF encoding，并加入：

- missing-atom mask；
- Protein local-frame displacement；
- RNA local-frame displacement；
- relative frame rotation。

这部分理念接近 LigandMPNN / NA-MPNN 的 rich atomic context，而不是简单 C-alpha / C1' 距离。

RNA structural-prior 输入禁止天然碱基 identity atoms 泄漏，例如 N1/N9/base-ring atoms。

---

# 5. Pilot 实验规模

本 mini-pilot 的冻结目标：

- **1,000 个 Protein structures**；
- **1,000 个 RNA structures / experimental RNA chain views**；
- **1,100 个 experimental Protein–RNA complexes**。

复合物：

- 1,000 development；
- 100 immutable final test。

Development 内部：

- 900 train；
- 100 validation。

重要：最终主模型与 baseline 都不应只停在 900 样本。

规范流程：

1. 900/100 用于选择 epoch / hyperparameters；
2. 一旦所有设置冻结；
3. 从随机初始化重新开始；
4. 使用完整 1,000 development samples 进行 final refit；
5. final 100 只使用一次进行正式报告，不用于任何 tuning。

---

# 6. 两个官方小数据 baseline

## 6.1 Protein baseline

官方上游：

> https://github.com/dauparas/ProteinMPNN

主比较：

- 与本项目同一冻结的 1,000 Protein structures；
- random initialization；
- 不用 published pretrained weights 作为 primary baseline；
- development 900/100 选择训练长度；
- 完整 1,000 refit；
- final100 complex 上只看 Protein backbone-only structural reference。

Published pretrained ProteinMPNN 可以作为 secondary reference，但不能和小数据从头训练 baseline 混在一起。

## 6.2 RNA baseline

官方上游：

> https://github.com/baker-laboratory/NA-MPNN

当前项目将其 fixed-backbone RNA 路线称为 MPNN-fixbb / NA-MPNN baseline。

同样：

- 与本项目同一 1,000 RNA pool；
- random initialization；
- 900/100 development；
- full-1000 refit；
- final100 仅作为 RNA backbone-only structural reference。

**不要声称它与 DM-ICF 是“能力完全等价的直接 baseline”。**

ProteinMPNN / NA-MPNN 属于 external one-sided structural references；真正证明 partner coupling 的因果证据应主要来自内部 controls / ablations。

## 6.3 Upstream lock

`third_party/LOCK.template.json` 目前仍要求在正式实验前复制为 `third_party/LOCK.json` 并写死 immutable commit SHA。

正式实验前必须：

- 固定 ProteinMPNN SHA；
- 固定 NA-MPNN SHA；
- 记录 checkout HEAD；
- 保存实际执行命令；
- 保存输入 manifest SHA256。

不要在实验中途更新 upstream。

---

# 7. 数据策略

## 7.1 原则

- Pilot 的 1,100 complex 必须 experimental-only；
- predicted structures 暂不进入这个 pilot；
- 核糖体、剪接体和极大 RNP assemblages 排除；
- 其他 Protein–RNA 类型尽量保留，避免数据集过早专业化；
- 所有 exact construct、chain、mother sample、cluster 信息都要保留。

## 7.2 RNA 1000 条的现实处理

如果严格的 standalone RNA-only PDB 不够 1000：

允许从实验 complex 中抽取 RNA chain view，训练 RNA structural prior 时完全隐藏 Protein partner。

这仍属于：

\[
B_R\rightarrow S_R
\]

但这些 RNA chain view 必须和 final test 一起执行 R80 + Rfam leakage purge。

## 7.3 Strict split

Protein：

- exact sequence；
- P30 strict cluster。

RNA：

- exact sequence；
- R80；
- Rfam family。

Complex split 时，只要任一 constituent chain 的 P30 / R80 / Rfam 有重叠，就必须属于同一个 connected component。

最终 100 先冻结，然后反向 purge：

- Protein prior pool 中的 final-test exact sequence / P30 neighbours；
- RNA prior pool 中的 final-test exact sequence / R80 / Rfam neighbours。

---

# 8. Loss 设计：这是硬规则

Protein 与 RNA alphabet 大小不同：

\[
|A_P|=20,\qquad |A_R|=4
\]

不能把所有 token CE 直接 sum。

基本 semantic groups：

- Protein interface；
- Protein non-interface；
- RNA interface；
- RNA non-interface。

先分别求 mean：

\[
\ell_{P,I},\ell_{P,N},\ell_{R,I},\ell_{R,N}
\]

跨 polymer 组合时使用 entropy normalization：

\[
\tilde \ell_{P,g}=\ell_{P,g}/\log20
\]

\[
\tilde \ell_{R,g}=\ell_{R,g}/\log4
\]

Joint 在四组均存在时：

\[
L_J=\frac14(
\tilde\ell_{P,I}+\tilde\ell_{P,N}+
\tilde\ell_{R,I}+\tilde\ell_{R,N})
\]

主原则：

> **先组内平均，再跨组组合；绝不能让 token 数量或 alphabet 大小自然决定 gradient voice。**

一个 minibatch / sample step 只执行一种 task，不直接把 P-cond + R-cond + Joint 三个大 loss 生硬相加。

Joint task ratio：

\[
2:2:1\rightarrow1:1:1
\]

---

# 9. 训练阶段

目标阶段：

1. Protein structural prior；
2. RNA structural prior；
3. Global C；
4. Delta C；
5. Alpha / relevance；
6. Joint coordination。

早期设计目标是逐层解释：

\[
C\rightarrow\Delta C\rightarrow\alpha
\]

但第二轮审计发现当前代码在 Stage Alpha / Stage Joint 仍有解释权混杂问题，见后面的 P0。

训练 trick：

- coordinate noise 主配置 0.10 Å；
- 消融 0 / 0.05 / 0.10 / 0.20 Å；
- variable mask 约 10–100%；
- random + local/spatial patch masking；
- small wrong-token corruption；
- 5% intra edge dropout；
- 5% PR edge dropout；
- light DropPath（深 encoder 时）；
- structural-prior label smoothing 0.05；
- DeltaC output zero-init；
- Alpha residual zero-init；
- AdamW；
- warmup + cosine；
- grad clip 1.0；
- BF16 when supported；
- gradual unfreezing 只适用于 pretrained model，不适用于 scratch control。

---

# 10. Joint inference 与 SPIR

Joint proposal 采用 mixed random-order autoregressive decoding，例如：

```text
P23 -> R8 -> P51 -> R11 -> ...
```

早期 token 可能在 partner 尚未知时生成。

因此第一轮结束后执行：

> **SPIR = Single-Pass Interface Reconciliation**

逻辑：

1. 两条 sequence 已完整；
2. non-interface 固定；
3. 只重新开放低 confidence 的约 20–40% interface positions；
4. 进行一次双向 reconciliation cycle；
5. 一半候选 Protein-first，一半 RNA-first；
6. refinement temperature 约 0.3–0.7；
7. 一般只做 1 cycle，repeated refinement 仅作消融。

需要测：

- no SPIR；
- one-pass SPIR；
- repeated refinement；
- mixed order；
- Protein-first；
- RNA-first；
- sequence diversity 是否塌缩。

---

# 11. 最终 100 complex 的主要评价

最终 statistical unit 应是 **complex**，不是 residue。

不能把数万个 residues 当成数万个独立样本制造显著性。

## Primary scientific questions 建议只保留 4–5 个

1. Structural pretraining 是否提高数据效率？
2. Global C 是否提供 structural prior 之外的 partner information？
3. DeltaC / alpha 是否在 global C 基础上进一步提高 interface modelling？
4. Full DM-ICF 是否表现出明确 partner dependence？
5. Joint design + SPIR 是否在不显著损伤 diversity 的情况下改善界面一致性？

只对这些 primary hypotheses 做主要 multiple-testing correction（例如 Holm）。

其他大量检查视为 secondary/exploratory，避免 30 多个 test 混成一个主假设集合。

## 必做分析

- raw NLL；
- entropy-normalized NLL；
- overall / interface / non-interface recovery；
- partner scramble DeltaNLL；
- counterfactual partner mutation；
- local KL vs distance；
- learned C vs empirical PMI；
- shuffled-partner PMI null；
- DeltaC magnitude；
- alpha entropy / effective neighbours / spatial map；
- coordinate noise robustness；
- PR-edge removal；
- partner hiding；
- decode-order sensitivity；
- SPIR ablation；
- candidate diversity / collapse；
- 10/25/50/100% data efficiency；
- three random seeds；
- paired complex-level bootstrap（建议 10,000）；
- primary comparison Holm correction。

---

# 12. Internal fairness controls

已设计的核心内部 controls：

## Partner-blind

跨分子 token-specific correction 强制为 0。

回答：

> full model 的提升是否真的来自 partner sequence information？

## Geometry-only capacity control

保留跨分子 geometry / q / DeltaC / alpha 的额外网络容量，但去掉具体 AA/base identity coupling。

回答：

> 提升是否只是因为看到了另一条链结构、或者单纯参数更多？

## C backbone-only-context control

Global C 学习时，减少/移除目标链已知 sequence context 对 C 的影响。

回答：

> C 是否真的近似“structure prior 之外的 cross-molecular correction”，还是严重依赖 target-chain sequence context？

## Fixed empirical-PMI control

PMI 不用于主模型初始化；可以单独作为固定 statistical potential control。

回答：

> 一个简单经验统计矩阵究竟能做到什么程度？

---

# 13. 当前仓库已经比较可靠的部分

截至本交接点，以下内容已经有较完整代码/合同或经过前一轮本地检查：

- 数据 RCSB discovery / download 框架；
- Rfam 下载与 annotation 框架；
- structure screening；
- P30/R80/Rfam strict freezing；
- final-test-first purge；
- multi-chain component split；
- RNA sequence-neutral geometry；
- Gemmi adapter；
- 5×12 PR rich geometry；
- C / DeltaC / alpha 核心算子；
- DeltaC zero init；
- distance-prior alpha init；
- edge / PR-edge dropout；
- mask curriculum / full mask；
- partner-token dropout；
- balanced Protein/RNA/interface loss；
- BF16 / AdamW / warmup cosine；
- six-stage trainer 框架；
- development/final-refit 基本框架；
- joint sampler / SPIR 框架；
- partner scramble / counterfactual / PMI 等 evaluation 框架；
- internal fairness controls 框架；
- Node24 GitHub Actions workflow；
- 手动 current-head audit workflow。

曾经对当时的源码快照执行过：

```text
python -m compileall -q src tests tools
python -m pytest -q
ruff check src tests tools --select E9,F63,F7,F82
```

并通过。

**但这不能替代第二轮 P0 修复后的重新测试。**

---

# 14. 当前未完成、必须先修的 P0

这一节是新工作区最重要的执行列表。

完整展开版见：

> `docs/CODEX_SECOND_PASS_REVIEW_AND_EXECUTION.md`

## P0-1 官方 baseline runner 与 pinned upstream CLI 不匹配

ProteinMPNN：

- training data root 语义要按官方源码修；
- 删除 upstream 不支持的 args；
- seed 需要通过最小 wrapper 注入，不改 upstream；
- mixed_precision 的 bool CLI 调用需要按官方实际 parser；
- 保存 actual command + SHA + manifest SHA。

NA-MPNN：

- pinned training entry 实际以 positional JSON 为入口，不能按任意 argparse CLI 调；
- 每个 seed 生成独立 JSON；
- BASE_FOLDER / TOTAL_STEPS / checkpoint path 必须按 pinned upstream 真行为；
- 增加 CPU preflight。

## P0-2 NA-MPNN RNA probability alphabet mapping

本项目 alphabet：

```text
A U G C
```

NA-MPNN 内部输出必须明确映射到这一顺序。

之前审计指出存在 U/C 对调风险。

必须写人工 PPM 单测，保证 A/U/G/C 的 native probability 和 argmax token 绝无列交换。

## P0-3 full-1000 refit schedule 必须重放 development schedule 前缀

不能把：

```text
best epoch = 20 / max 150
```

在 refit 中压缩成“20 epoch 内完整跑完 0→100% curriculum / unfreezing / cosine”。

正确：

- `selected_epoch_count = 20`
- `schedule_horizon_epochs = 150`
- refit 只重放 development schedule 的前 20/150。

要记录：

- selected epoch count；
- schedule horizon；
- schedule progress at stop。

## P0-4 Scratch control 不能 gradual unfreeze

Scratch joint baseline 随机初始化后必须从 epoch 0 全参数可训练。

Gradual unfreezing 是保护 pretrained representations 的 transfer-learning trick，不能拿来限制随机初始化 baseline。

## P0-5 C -> DeltaC -> alpha 阶段要进一步干净化

Primary：

- Stage C：只训 C；hard-context mining 先关掉；
- Stage Delta：训 q + DeltaC；
- Stage Alpha：冻结 C/q/DeltaC，只训 alpha/tau；
- Stage Joint：可放 structural prior / Delta / alpha 小 LR 协调；**Global C primary 版本继续冻结。**

原因：保持可解释性与后续 C-vs-PMI 意义。

## P0-6 Canonical biological interface 与 model PR graph 分离

当前 screening 用 full-heavy-atom 6 Å，runtime interface 可能来自 8 Å + neighbour cap 的模型 PR graph。

必须分开：

- canonical interface = full heavy-atom 6 Å；
- PR graph = 8 Å + cap，只负责 receptive field；
- PI/RI loss、interface recovery/NLL、baseline mapping 一律使用 canonical biological interface。

建议在 manifest 存：

```text
protein_interface_residue_ids
rna_interface_residue_ids
```

并在 load_complex 后覆盖 canonical mask。

单测：改变 PR cutoff/cap 不得改变 canonical interface label。

## P0-7 Joint validation 不能继续用一次双方 full-mask forward

双方都 unknown 时，DM-ICF 的 token-specific cross correction 基本不能发挥作用。

因此它不能代表真正 joint co-design checkpoint quality。

推荐 deterministic teacher-forced sequential pseudo-NLL：

- fixed mixed order；
- Protein-first；
- RNA-first；
- 按真实 autoregressive known/unknown state逐 token评分 native probability；
- 多 order 平均。

Checkpoint selection 必须更接近真实 joint inference。

## P0-8 最终 evaluation 预算与 hypothesis hierarchy

Final100 不要从第一天就把全部 expensive combinations 全开。

先定义：

- core final battery；
- expensive secondary battery；
- exploratory battery。

同时冻结 primary hypotheses，避免过度 multiple testing。

---

# 15. 还需要同步修的文档漂移

P0 修完后必须同步：

- `README.md`
- `RUNBOOK.md`
- `docs/EXPERIMENT_SPEC.md`
- `docs/SELF_AUDIT.md`
- manuscript / Methods 中 C gauge 与 training-stage 描述

早期文档可能还保留已经被后续审阅推翻或微调的内容，例如：

- C train-time double-centering；
- Joint 中 C 是否更新；
- interface 是否由 PR graph 定义；
- scratch 是否 gradual-unfreeze；
- baseline 命令。

新工作区必须以**最新代码 + 最新审计决定**为 source of truth，而不是任选一份旧文档。

---

# 16. 新工作区的推荐执行顺序

## Phase 0：只修合同，不跑正式 GPU

1. checkout `Neabigmo/pr main`；
2. 阅读：
   - 本文档；
   - `docs/CODEX_SECOND_PASS_REVIEW_AND_EXECUTION.md`；
   - `docs/PROJECT_STATUS_FOR_YIHENG.md`；
   - `docs/SELF_AUDIT.md`；
3. 修 P0-1 ～ P0-8；
4. 每一项都补测试；
5. 重新跑：

```bash
python -m compileall -q src tests tools
python -m pytest -q
ruff check src tests tools --select E9,F63,F7,F82
```

6. 所有关键 CLI 跑 `--help` / dry-run；
7. baseline preflight 必须通过；
8. 提交一个明确的 P0-closure commit。

## Phase 1：synthetic / micro smoke

不要马上上 1000+1000+1100。

先构造：

- 5–10 Protein；
- 5–10 RNA；
- 5–10 complexes。

完整走：

```text
download/adapter
-> manifest
-> prior train
-> C
-> DeltaC
-> alpha
-> joint
-> refit
-> sample
-> SPIR
-> evaluate
-> compare
```

目标不是性能，是证明整个 pipeline 不断裂。

## Phase 2：真实数据构建

建议 oversample candidates，例如：

- Protein 下载 3k–5k 候选；
- RNA 下载/抽链 3k–5k 候选；
- Complex 下载 3k–5k 候选；

再筛到足够形成 strict 1000 / 1000 / 1100。

执行：

```text
RCSB discovery
-> download
-> screen
-> canonical residue audit
-> P30/R80 clustering
-> Rfam
-> final100-first strict split
-> purge prior pools
-> freeze manifests
-> audit
```

## Phase 3：CPU data GO/NO-GO

必须满足：

- exact frozen counts；
- no test leakage；
- no pretraining leakage；
- canonical interface coverage 合理；
- feature missingness 可接受；
- PR graph coverage 合理；
- Protein/RNA family distribution 可解释；
- final100 全部 experimental；
- baseline data conversion 与 manifest sequence 逐条一致。

不满足就 NO-GO。

## Phase 4：GPU mini-pilot

三个 seeds，推荐至少：

```text
20260905
20260906
20260907
```

顺序：

1. ProteinMPNN dev；
2. ProteinMPNN full-1000 refit；
3. NA-MPNN dev；
4. NA-MPNN full-1000 refit；
5. DM-ICF Protein prior；
6. RNA prior；
7. C；
8. DeltaC；
9. Alpha；
10. Joint；
11. full-1000 final refit；
12. internal controls；
13. final100 core battery；
14. paired stats；
15. only then expensive secondary battery。

---

# 17. GO / NO-GO 标准

正式 GPU 前，以下任一不满足都应 NO-GO：

- upstream baseline preflight 失败；
- NA-MPNN alphabet mapping 未有单测；
- refit schedule 与 development 不同定义；
- scratch baseline 被 pretrained-specific 冻结策略限制；
- canonical interface 与 model graph 混用；
- Joint checkpoint metric 不含 sequential partner-aware validation；
- final-test leakage 不为 0；
- prior-pool leakage 不为 0；
- 任何配置中的 primary trick 只有 YAML 而无执行代码；
- synthetic end-to-end pipeline 不能完整跑完。

---

# 18. 论文主线不要改变

在新工作区中，不要因为工程修复重新把论文主线变复杂。

核心仍然是：

> **结构先验约束 + 动态跨分子互选。**

三层知识：

1. structural prior；
2. global C；
3. context-dependent DeltaC + neighbour relevance alpha。

成熟领域 trick（mask curriculum、zero-init、coordinate noise、gradual unfreezing、discriminative LR 等）是训练工具，不要包装成主要创新。

主论文最值得强调的 scientific tests：

- pretraining data efficiency；
- C independently recapitulates empirical enrichment；
- DeltaC captures local geometric deviations；
- partner scramble/counterfactual proves partner dependence；
- joint design / SPIR improves practical co-design stability。

---

# 19. 关键“不要做”列表

1. 不要用 final100 调任何超参数。
2. 不要用 final100 决定是否开启 PCGrad。
3. 不要把 PMI 用来初始化主模型 C。
4. 不要把 C 或 C+DeltaC 称为 binding energy。
5. 不要把 alpha 称为物理因果重要性。
6. 不要把 predicted structure 和 experimental structure 混成一个证据层级。
7. 不要让 ProteinMPNN/NA-MPNN 看额外 partner 信息。
8. 不要因为 baseline 跑不通就偷偷修改 upstream 源码而不记录。
9. 不要对 scratch control 使用保护 pretrained weights 的 gradual unfreezing。
10. 不要把 model PR graph 当成 biological interface 的唯一真值。
11. 不要用 residue-level sample size 做统计显著性。
12. 不要把 30 多个 exploratory test 全部声明成 primary hypotheses。
13. 不要为了“更高级”继续给 DM-ICF 加大 Transformer/diffusion 模块。

---

# 20. 当前仓库重要文件

优先查看：

```text
README.md
RUNBOOK.md
configs/pilot.yaml
third_party/LOCK.template.json

docs/EXPERIMENT_SPEC.md
docs/DATA_PIPELINE.md
docs/SELF_AUDIT.md
docs/CODEX_SECOND_PASS_REVIEW_AND_EXECUTION.md
docs/PROJECT_STATUS_FOR_YIHENG.md

src/pr_pilot/model/dmicf.py
src/pr_pilot/runtime/gemmi_adapter.py
src/pr_pilot/data/manifest.py
src/pr_pilot/training/engine.py
src/pr_pilot/training/stages.py
src/pr_pilot/training/losses.py
src/pr_pilot/inference/sampler.py
src/pr_pilot/evaluation/

tools/run_official_baselines.py
tools/evaluate_official_baselines.py
tools/run_final_refit.py
tools/run_component_ladder.py
tools/compare_runs.py
```

注意：文件是否存在与内容是否完全正确是两回事。P0-closure 前不能仅凭“目录看起来很完整”判断项目完成。

---

# 21. 给新的 Codex/ChatGPT 工作区的首条提示词

建议直接把下面这段作为新工作区第一条任务：

```text
你正在接手 GitHub 仓库 https://github.com/Neabigmo/pr 的 Protein–RNA joint design mini-pilot。

先不要增加新模型功能，不要启动正式 GPU 训练。请以仓库 main 最新 HEAD 为准，先阅读：
1. docs/PR_MINIPILOT_NEW_WORKSPACE_HANDOFF.md
2. docs/CODEX_SECOND_PASS_REVIEW_AND_EXECUTION.md
3. docs/SELF_AUDIT.md
4. docs/EXPERIMENT_SPEC.md
5. configs/pilot.yaml

核心科学设计不得随意改变：
sequence preference = structural prior + cross-molecular selection；
DM-ICF = alpha_ij (C + DeltaC_ij)；
Protein/RNA structural priors 独立预训练；C 小随机初始化；PMI 只做 post-hoc validation；最终支持双侧 conditional design 和 joint design + SPIR。

当前最重要的工作是关闭交接文档列出的 P0：
- official ProteinMPNN / NA-MPNN runner 与 pinned upstream CLI 合同；
- NA-MPNN AUGC 概率映射；
- development/refit schedule 一致；
- scratch control 公平性；
- C→DeltaC→alpha 阶段解耦；
- canonical heavy-atom biological interface 与 model PR graph 分离；
- joint checkpoint validation 改成 deterministic sequential pseudo-NLL；
- final100 core/secondary hypothesis hierarchy 与计算预算。

要求：
- 每修一个 P0 都补 regression/unit test；
- 不许用文档注释代替代码实现；
- 不许改 upstream baseline 源码而不记录；
- 修完后运行 compileall、pytest、Ruff correctness、CLI smoke test、official baseline preflight；
- 给出逐条 P0 closure 证据；
- 只有全部通过才允许进入真实数据下载/筛选与 GPU。
```

---

# 22. 给未来自己的项目判断

如果 mini-pilot 成功，最有价值的结论不是“1000 条数据就打败所有 SOTA”。

真正希望回答的是：

1. 两个 structural priors 能否在极小 complex 数据规模下提供稳定基础？
2. C 是否能从随机初始化中自发学出和 empirical AA–base enrichment 有一致性的结构？
3. DeltaC 是否真的学习 geometry-specific deviation，而不是无意义噪声？
4. Alpha 是否能在多个邻居间形成合理、稳定、非纯距离的 relevance？
5. Partner scramble / counterfactual 是否证明模型的 interface prediction 真正依赖另一条链？
6. Joint decoding 是否能在不依赖巨大迭代过程的情况下完成双侧设计？
7. SPIR 是否能修正 early unknown-partner decisions，同时不造成 sequence collapse？

如果这些机制问题成立，即使 mini-pilot 的绝对 recovery 尚未压过所有大规模 pretrained 模型，这个小实验也已经达到了最重要目的：

> **证明我们的建模分解与训练方案在小数据上是可学习、可解释、可检验的，然后再合理放大数据规模。**

---

# 23. 最终交接结论

当前最准确的项目状态：

```text
科学问题：清楚
核心模型：基本冻结
数据设计：清楚
主要模块代码：大面积实现
前一轮代码合同测试：曾通过
第二轮深审 P0：已明确，但尚未全部代码关闭
真实 1000/1000/1100 数据冻结：尚未正式完成
正式 GPU mini-pilot：尚未开始
最终论文结果：尚未产生
```

因此，新工作区的正确目标不是重新设计论文，而是：

\[
\boxed{
\text{关闭 P0}
\rightarrow
\text{synthetic end-to-end smoke}
\rightarrow
\text{真实数据冻结}
\rightarrow
\text{CPU audit}
\rightarrow
\text{GPU mini-pilot}
\rightarrow
\text{final100 evaluation}
}
\]

这条顺序不要打乱。
