<!-- REVIEW4-ROUTING-2026-09-05 -->
> **Review-4 routing notice (2026-09-05).** See [INDEX_REVIEW4.md](INDEX_REVIEW4.md) and the Review-4 runbook for updated contracts, known incompatibilities and outstanding integration gates. The original text below is preserved as historical context; it is not a claim that the new protocol or real experiments have passed.

# 给 Codex：PR Mini-Pilot 第二轮审阅后的修复与执行计划

> 目标：不要继续堆功能。先把当前 mini-pilot 从“逻辑完整的研究代码”收敛成“真正可运行、定义一致、可公平比较、可写进论文的方法与实验系统”。
>
> **硬规则：P0 项全部关闭前，不启动正式 GPU 训练，不冻结最终论文结果。**

---

# 0. 你现在面对的项目是什么

项目目标是固定骨架条件下的 Protein–RNA sequence design / co-design。核心分解为：

```text
sequence preference
=
intramolecular structural prior
+
local cross-molecular selection
```

DM-ICF 的跨分子项为：

```text
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

其中：

- `C`: 20×4 全局 AA–base compatibility anchor，小随机初始化；
- `DeltaC_ij`: 由 Protein/RNA 结构隐状态与 PR 几何生成的上下文残差；
- `alpha_ij`: 邻域内关系重要性；
- Protein 与 RNA structural prior 分开训练；
- 后续依次训练 `C -> DeltaC -> alpha -> joint`；
- 最终采用 mixed-order autoregressive decoding + SPIR。

数据目标：

- 1,000 Protein structural-prior structures；
- 1,000 RNA structural-prior structures；
- 1,100 experimental Protein–RNA complexes；
- 其中 1,000 development、100 immutable final test；
- development 内部 900/100 用于选择 epoch/超参数，最后在完整 1,000 上 refit。

当前 repo 已有模型、数据筛选、聚类/Rfam、训练、refit、evaluation、internal controls、official baseline adapters 等代码，但第二轮审阅发现下列问题。

---

# 1. P0：必须在任何正式训练前修复

## P0-1. 官方 baseline runner 与 pinned upstream CLI 不匹配

这是当前最明确的“会真正跑崩”的问题。

### ProteinMPNN

当前 `tools/run_official_baselines.py` 不能直接照现状运行。

Pinned 官方 `training/training.py` 的 CLI 只接受其官方参数；当前 runner 中存在以下风险/错误：

1. `--path_for_training_data` 应指向 prepared ProteinMPNN 根目录，而不是其 `pdb/` 子目录。官方 loader 自己会在该 root 下寻找：
   - `list.csv`
   - `valid_clusters.txt`
   - `test_clusters.txt`
   - `pdb/...`
2. 不要传官方脚本不存在的：
   - `--path_for_training_clusters`
   - `--path_for_valid_clusters`
   - `--path_for_test_clusters`
   - `--seed`
3. 官方 `--mixed_precision` 是 `type=bool`，调用时必须给值，例如：
   ```text
   --mixed_precision True
   ```
   不能只传裸 flag。
4. 官方训练脚本没有 seed 参数。为了保留官方源码不修改，同时实现可重复 seed，新增一个最小 wrapper，例如：
   ```text
   tools/run_seeded_upstream.py
   ```
   它只负责：
   - `random.seed(seed)`
   - `numpy.random.seed(seed)`
   - `torch.manual_seed(seed)` / CUDA seed
   - 设置 `sys.argv`
   - `runpy.run_path(upstream_script, run_name="__main__")`
   不改 upstream checkout 文件。
5. 每次 run 保存最终“实际执行命令 + upstream SHA + seed + prepared manifest SHA”。

### NA-MPNN

Pinned `na_run.py` 的入口不是 argparse CLI，而是：

```python
JSON = sys.argv[1]
```

因此当前类似：

```text
python na_run.py --path_for_outputs ... --model_input_json ... --seed ...
```

是错误的。

修复方式：

1. 对每个 seed 创建一份独立 JSON 配置副本；
2. 在 JSON 中写入该 run 的 `BASE_FOLDER`；
3. 执行：
   ```text
   python tools/run_seeded_upstream.py --seed ... --script third_party/.../na_run.py -- <config.json>
   ```
4. `NA-MPNN` checkpoint 按 pinned 版本真实行为读取：优先明确的 `BASE_FOLDER/last.pt`，不要假设存在 `model_weights/*.pt`。
5. development 与 full-1000 refit 都按上述方式执行。

### baseline preflight

新增一个无需 GPU 的 preflight：

```text
python tools/preflight_official_baselines.py
```

要求至少检查：

- LOCK SHA 与 checkout HEAD 一致；
- ProteinMPNN training data root 结构正确；
- ProteinMPNN command 不包含 unsupported args；
- NA-MPNN command 只有 positional JSON；
- NA-MPNN JSON 的 `BASE_FOLDER`、`TOTAL_STEPS`、train/valid CSV 均存在；
- 所有 full-refit checkpoint 解析规则与 pinned upstream 源码一致。

---

## P0-2. NA-MPNN RNA 概率列顺序错误

当前项目统一 RNA alphabet 是：

```text
A U G C
```

但 `evaluate_official_baselines.py` 当前从 NA-MPNN 取列时使用了类似：

```text
DA DC DG DT
```

这实际对应：

```text
A C G U
```

会把 **U 与 C 对调**。

必须改为与本项目 `AUGC` 完全一致：

```text
DA DT DG DC
```

如果使用 RNA token 名，则为：

```text
A U G C
```

新增强制单元测试：构造一个人工 PPM，使 A/U/G/C 每列各自明显最大，确认最终 native probability 和 predicted token 没有 C/U swap。

同时删除 NA-MPNN inference 中 pinned `inference/run.py` 不支持的参数，例如当前 `--rna_backbone_noise`，除非通过源码确认该参数真实存在。

---

## P0-3. full-1000 refit 的 schedule 目前与 development 不是同一个 schedule

当前 development：

```text
progress = epoch / (max_epochs - 1)
LR cosine horizon = max_epochs * len(train)
```

但 refit 当前把：

```text
progress = epoch / (selected_epochs - 1)
LR horizon = selected_epochs * len(full1000)
```

这会把整个 curriculum / gradual unfreezing / cosine decay / joint task schedule **压缩到较短的 selected epoch 数里**。

例如 development 在第 20/150 epoch 取得最佳模型时，只走了约 13% 的 unfreezing/curriculum；refit 如果训练 20 epoch，却会走完整个 0→100% schedule。

这不是“按选中的 epoch 原样 refit”，而是改变了训练算法。

### 修复

给每个 stage 明确定义：

```text
schedule_horizon_epochs = development max_epochs
```

refit 虽然只运行 `selected_epoch_count` 个 epoch，但：

```text
progress = epoch / (schedule_horizon_epochs - 1)
LR horizon = schedule_horizon_epochs * len(full1000_manifest)
```

也就是说 refit 重放 development schedule 的前缀，而不是把 schedule 压缩到结束。

checkpoint 中记录：

- `selected_epoch_count`
- `schedule_horizon_epochs`
- `schedule_progress_at_stop`

新增测试：同一个 stage 在 development epoch `k` 与 refit epoch `k` 的 curriculum fraction、joint task ratio、unfreezing depth、LR normalized scale 应一致（允许因为 900/1000 steps 带来的 epoch 内细微差异，但 epoch-level schedule 必须一致）。

---

## P0-4. scratch joint control 当前被 gradual unfreezing 人为削弱

component ladder 的 scratch joint 从随机初始化开始，但使用了为“预训练 encoder 微调”设计的 gradual unfreezing。

这意味着随机初始化的多数 encoder layer 在早期被冻结，属于不公平 handicap。

### 修复

`scratch joint` 必须：

- 从随机初始化；
- **所有模型参数从 epoch 0 就可训练**；
- 同样的 optimizer family、loss、data、mask、noise 与总预算；
- 不使用 pretrained-specific gradual unfreezing。

实现一个明确模式：

```text
joint_unfreezing_mode:
  pretrained: gradual
  scratch: all_trainable_from_start
```

不要用一个 boolean 让语义含糊。

component ladder A 必须使用 `scratch` 模式。

---

## P0-5. `C -> DeltaC -> alpha` 的阶段可解释性还不够干净

当前 Alpha stage 会继续训练：

- interaction encoder `q_ij`
- `DeltaC`
- relevance/alpha

这会使 `DeltaC` 与 `alpha` 同时重新分配解释权，削弱“先学 contextual compatibility，再学谁更重要”的阶段解释。

### 主实验改成更干净的版本

建议：

### Stage C

只训练 `C`。

并把 primary 配置中的：

```text
hard_context_fraction_late
```

设为 0。

原因：Global C 如果后半程专门过采样“结构先验最困难的位置”，它就不再是一个干净的 population/global anchor，而是 hard-example-weighted anchor。

hard-context mining 可以保留为后续 optimization ablation，不进入 primary C。

### Stage Delta

训练：

- interaction encoder `q_ij`
- `DeltaC`

冻结：

- P/R priors
- C
- alpha relevance

### Stage Alpha

Primary 版本冻结：

- `C`
- interaction encoder
- `DeltaC`
- P/R priors

只训练：

- alpha/relevance score residual
- tau（若 tau 被定义为 alpha 模块的一部分）

这样才能把：

```text
C = global preference
DeltaC = context correction
alpha = neighbour relevance
```

真正拆开。

### Stage Joint

可以重新允许 Delta/alpha 与 structural prior 小 LR 协调，但：

**Global C 在 primary 模型中必须继续冻结。**

当前 joint 给 C 一个低 LR 会导致最终 C 与 Stage-C global anchor 漂移，后续 `C vs PMI` 的解释变得不干净。

主实验将 `C` 视为固定 anchor，joint 不再更新它。

---

## P0-6. interface 定义不能依赖模型自己的 PR graph

目前存在两个 interface 定义：

1. 数据 screening：全重原子 Protein–RNA 接触，6 Å；
2. runtime：PR graph 8 Å + neighbour cap + 只基于模型允许的 Protein/RNA 原子集合，然后用 retained PR edges 标 interface。

如果训练、评估、external baseline mapping 使用第 2 种，那么“interface recovery/NLL”本身会依赖 DM-ICF 的图构建方式，不够公平。

### 修复原则

**canonical biological interface 与 model message-passing graph 分离。**

- canonical interface：screening 时由 full heavy-atom 6 Å 定义；
- PR graph：仍用 8 Å + neighbour cap，作为模型 receptive field；
- loss 的 PI/RI、evaluation 的 interface/non-interface、baseline position mapping 全部使用 canonical 6 Å mask。

### 实现建议

在 screening 输出 complex row 时保存：

```text
protein_interface_residue_ids
rna_interface_residue_ids
```

用 JSON array，residue ID 与 runtime `residue_ids` 采用同一格式。

`load_complex_row()` 读取 manifest 后覆盖 `PolymerGraph.interface`：

```text
interface = residue_id in canonical_interface_set
```

PR graph 仍独立构建。

新增测试：改变 `pr_cutoff_angstrom` 或 `pr_max_neighbors` 时，canonical interface mask 不得变化。

---

## P0-7. Joint stage 的 validation metric 与真实 joint inference 不一致

当前 joint validation 有一个 full-mask forward：Protein/RNA 的非 fixed token 都设成 unknown。

但 DM-ICF correction 的逻辑是：未知 partner token contribution = 0。

因此在“双方全 unknown 的一次 forward”里，cross-molecular correction 基本不参与；这个 joint validation 项实际上主要在测 structural prior，而不是 mixed-order co-design。

### 修复

Joint checkpoint selection 必须与实际 joint generation 更一致。

推荐使用 **deterministic teacher-forced sequential pseudo-NLL**：

- 每个 validation complex 使用固定 3 个 order；
- 至少包含：
  - 1 mixed deterministic order
  - Protein-first
  - RNA-first
- 按 order 逐 token：
  - 当前 token 设 unknown；
  - 已经“解码”的位置填 native token 并设 known；
  - 记录 native log probability；
- 得到 order-averaged normalized pseudo-NLL。

这是 validation-only teacher forcing，不是 final candidate generation。

若算力太高，可只在 complex_val 的固定子集上做 sequential metric，同时保留便宜的 conditional P/R metric；但 checkpoint rule 必须预先固定。

不要继续用“一次全 mask forward”冒充 joint quality。

---

# 2. P1：强烈建议在正式 final-100 前完成

## P1-1. Final evaluation 计算量目前过大

当前 `full_suite` 对每个 complex 做多组：

- 64 candidates；
- 0/1/repeated SPIR；
- mixed / Protein-first / RNA-first；
- with/without SPIR；
- 3 seeds；
- 每个 candidate 又是逐 token autoregressive full forward。

在长度较大的 complex 上会产生数千万级 model forward，mini-pilot 很可能被 evaluation 而不是 training 拖死。

### 改成分层预算

#### Tier A：所有 3 seeds × 全 100 complexes

只跑论文 primary/core：

- conditional Protein/RNA NLL + recovery；
- canonical interface/non-interface；
- teacher-forced joint pseudo-NLL；
- calibration；
- primary internal controls；
- C seed stability。

#### Tier B：完整 mechanistic/robustness battery

只用预先指定的 `analysis_seed = 20260905` 跑全 100，或若仍过重则使用预先冻结的 30–50 complex diagnostic subset。

不能根据结果挑 subset。

#### Candidate budget

- primary mixed-order + 1-pass SPIR：64；
- order/SPIR ablation：16；
- repeated-SPIR：16；
- 若需要结构预测 evaluator，再从 64 中按预注册规则取固定数量，不能按 final-test recovery 选。

### 先做 runtime profiling gate

正式跑 final100 前：

```text
10 complexes
× representative length bins
× one primary candidate budget
```

记录：

- sec/candidate；
- GPU memory；
- estimated GPU-hours for full battery。

若预算不可接受，调整的是 **预注册 evaluation budget**，不是模型。

### 可选工程优化

优先缓存不会随 decoding token 改变的：

- backbone encodings hp/hr；
- q_ij；
- C + DeltaC；
- alpha structural scores。

不要每生成一个 token 都重新计算完整 backbone encoder。

---

## P1-2. 不要把 30 多个实验都叫 primary hypothesis

当前 battery 很丰富，这是优点；但如果 20–30 个项目都标 `primary=true`，Holm 会极端保守，而且论文故事会散。

### 重新分级

建议 primary confirmatory hypotheses 最多 4–5 个：

**H1.** Full DM-ICF 优于 dual structural prior，primary endpoint = interface normalized NLL。

**H2.** Full DM-ICF 优于 partner-blind / geometry-only capacity control，证明收益来自 partner identity coupling，而不仅仅是参数量或 partner geometry。

**H3.** Contextual field（C + DeltaC + alpha）相对 C-only 有增益，可用预注册的 component comparison 表达。

**H4.** Partner scramble 对 interface NLL 的破坏显著高于 matched control，作为 interaction dependence 的 model-intervention evidence。

可选 **H5.** learned alpha top-edge removal 比 distance-matched lower-alpha edge removal 造成更大 NLL degradation。

Holm 只在这些 primary comparisons 内做。

其余：

- noise robustness；
- repeated SPIR；
- PMI stratification；
- DeltaC case studies；
- decoding-order sensitivity；
- data efficiency；
- calibration；
- NMR sensitivity；

标为 secondary / exploratory，并用 BH 或只报告 CI/effect size。

不要使用“causal biological evidence”措辞。`alpha_edge_removal`、partner scramble 等应写成：

```text
model-interventional / mechanistic sensitivity evidence
```

---

## P1-3. 增加 DeltaC mean-drift audit

即使 C 冻结，DeltaC 也可能学出一个几乎处处相同的矩阵，从而实际上重新定义 global compatibility。

不建议现在临时加一个拍脑袋 regularizer。

先增加 audit：在 complex train/val 上统计：

```text
mean_DeltaC = E_edges[DeltaC_ij]
RMS_DeltaC
mean_shift_ratio = ||mean_DeltaC||_F / RMS_DeltaC
```

同时报告 alpha-weighted 版本。

如果 mean shift 很大：

- 论文里把 raw C 称为 `stage-1 global compatibility anchor`；
- post-hoc 可报告 `C_eff = C + mean_DeltaC`；
- 不要把 raw C 强行说成 final population-average interaction matrix。

不要看 final100 后再决定是否加 penalty。

---

## P1-4. 做 config-key coverage audit，彻底消灭“假开关”

当前 YAML 仍有一些 key 只存在配置文件、没有真正控制执行，例如需要重点检查：

- `preferred_resolution_angstrom`
- `require_interface_contact`
- `group_conformers`
- `group_mutation_series`
- 若干 loss 的 `normalize_* / equalize_*`
- 部分 stage 声明项
- 部分 fairness declarative flags

有些功能代码里确实是 hard-coded true，这不一定错，但 YAML 不应给人“可关闭”的假象。

实现一个测试/脚本：

```text
python tools/audit_config_usage.py --config configs/pilot.yaml
```

每个 leaf key 必须属于三类之一：

1. `runtime_consumed`
2. `declarative_assertion`
3. `deprecated/remove`

CI 中不允许出现第四类 `unknown/dead`。

对于 hard-coded scientific invariant，最好从 YAML 删除开关，直接写进 implementation contract；不要制造“看起来可调但实际上不生效”的参数。

---

## P1-5. 文档已明显落后于代码，需要统一

重点：

### README

当前 README 仍写着类似“remaining local adapter is the next implementation step”，但 Gemmi adapter、data pipeline、baseline converters 已经存在。

必须重写成真实状态。

### RUNBOOK

当前 RUNBOOK 仍保留旧路径/旧工具名，例如手工 `prepare_proteinmpnn.py`、旧的 eligible table 流程，以及把 full-development refit 写成 optional。

应以：

```text
docs/DATA_PIPELINE.md
+ tools/run_official_baselines.py
+ tools/run_pilot_experiments.py
```

为唯一真实执行入口。

修复后，把旧命令删掉，不要同时保留两个互相冲突的 runbook。

---

# 3. P2：数据冻结时一起完善

## P2-1. NMR 不要和 X-ray/cryo-EM 静默混在一起

RNA 数据中 NMR 很重要，因此不建议为了整洁直接删除。

但 primary data audit 必须：

- 分别统计 X-ray / cryo-EM / NMR；
- NMR 固定使用 model 1，不随机 model；
- final100 中报告 method composition；
- 增加 `exclude_NMR` 的 sensitivity table（不重新训练也可先做 evaluation stratification）。

如果 NMR 占比非常高，再决定是否单独版本化数据集；不能 final100 出结果后再删。

---

## P2-2. 再加一层 post-freeze leakage audit

P30 + R80 + Rfam 已经不错，但为了防局部 domain homology 漏过 80% coverage clustering，freeze 后再跑一次只读 audit：

- Protein dev vs final100：MMseqs local search，记录最高 identity/coverage；
- RNA dev vs final100：R80 + Rfam 之外再记录最佳 local similarity；
- 只报告，不在看到 final metrics 后更改 split。

如果发现明显近同源，应在 **训练前** version 成 `pilot_v2`，而不是训练后偷偷修。

---

## P2-3. Rfam `CURRENT` 需要可复现元数据

现有 SHA256 很重要，再补：

- resolved release/version；
- download date；
- URL；
- `Rfam.cm.gz` / `Rfam.clanin` SHA256。

最好把 release 元数据写入 manifest，而不是只写 `CURRENT`。

---

## P2-4. RCSB 下载器并发与断点续传

当前顺序下载 5k/3k/6k structures 在中国网络环境会很慢。

不改变科学逻辑的前提下：

- ThreadPool 并发 8–16；
- existing non-zero + checksum valid 则 skip；
- `.part` 文件可恢复/清理；
- retry/backoff；
- failures.tsv 永远保留；
- 最终 download manifest 按 deterministic candidate order 排序，而不是完成顺序。

这属于工程优化，可以在 P0 完成后做。

---

# 4. 推荐的最终训练语义

修复后的 primary stage 应明确成：

```text
Stage P
  train Protein structural prior

Stage R
  load P checkpoint
  train RNA structural prior only

Stage C
  freeze both priors
  train global C only
  fixed distance alpha
  no hard-context mining

Stage Delta
  freeze priors + C
  train q/interactions + DeltaC
  fixed distance alpha

Stage Alpha
  freeze priors + C + q + DeltaC
  train relevance residual + tau only

Stage Joint
  C remains frozen
  DeltaC / alpha / sequence-context heads adapt
  pretrained encoders gradual-unfreeze at low LR
  bounded lambda_P/lambda_R learned
```

Scratch control：

```text
same full architecture
random initialization
all parameters trainable from step 0
no pretrained gradual-unfreezing handicap
```

---

# 5. 修复后的 primary evaluation story

论文最重要的主线不要被 30 个实验冲散：

```text
1. structural priors work
2. adding global C helps interface selection
3. DeltaC adds context dependence
4. alpha identifies useful neighbours
5. full joint model improves over partner-blind / geometry-only controls
6. partner perturbations actually change predictions in local, structure-dependent ways
```

External ProteinMPNN / NA-MPNN：

- 作为 one-sided structural references；
- 同一 source pool、同样 full-1000 refit rule；
- 不拿它们单独证明 cross-partner coupling；
- probability semantics 分开写清楚。

---

# 6. Codex 的执行顺序

严格按以下顺序工作。

## Phase A — P0 code closure

1. 修 official baseline train runner；
2. 修 external baseline evaluator；
3. 修 NA-MPNN RNA column order；
4. 修 refit schedule replay；
5. 修 scratch control all-trainable；
6. 重构 C/Delta/Alpha/JOIN 冻结关系；
7. canonical heavy-atom interface 独立于 PR graph；
8. joint validation 改为 teacher-forced sequential pseudo-NLL；
9. 补单元测试与 smoke tests。

完成后输出：

```text
artifacts/review2/P0_FIX_REPORT.md
```

必须逐条说明：旧行为、风险、修改、测试。

## Phase B — scientific hygiene

1. primary/secondary/exploratory hypothesis registry；
2. DeltaC mean-drift audit；
3. config usage audit；
4. evaluation budget 分层 + runtime profiler；
5. README/RUNBOOK/SELF_AUDIT 同步。

输出：

```text
artifacts/review2/SCIENTIFIC_PROTOCOL_V2.md
```

## Phase C — tiny synthetic smoke experiment

不要马上下载完整数据。

先用极小 synthetic/已有 fixture 做：

```text
2 Protein
2 RNA
2 complexes
```

从 P → R → C → Delta → Alpha → Joint → refit → evaluate 完整走一次。

目标不是看指标，而是确认：

- stage checkpoint 可串联；
- final refit 可串联；
- official baseline preflight 通过；
- final evaluator 能生成所有 required schema；
- control modes 不崩；
- joint teacher-forced validation 不崩。

输出：

```text
artifacts/review2/END_TO_END_SMOKE_REPORT.md
```

## Phase D — 真实数据 pipeline

只有 A–C 全通过后，按 `docs/DATA_PIPELINE.md`：

```text
RCSB discover
-> oversized download
-> screen
-> RNA chain-view augmentation
-> joint MMseqs + Rfam
-> freeze final100 first
-> purge prior pools
-> freeze 1000/1000/1000
-> audit-data
```

## Phase E — 训练前 GO/NO-GO

Codex 必须生成：

```text
artifacts/GO_NO_GO.md
```

只有以下全部为 YES 才允许用户启动正式 GPU：

- [ ] P0 全修复并测试
- [ ] current HEAD pytest/compile/Ruff correctness 通过
- [ ] official baseline preflight 通过
- [ ] exact 1000/1000/1000 + final100 frozen
- [ ] P30/R80/Rfam/exact leakage = 0
- [ ] canonical interface 与 PR graph 解耦
- [ ] end-to-end smoke 通过
- [ ] full evaluation GPU-hour estimate 可接受
- [ ] primary hypotheses frozen
- [ ] config usage audit 无 dead keys
- [ ] final configuration 已 SHA256 frozen

---

# 7. 不允许 Codex 做的事

- 不要因为某个 final100 指标不好就改模型；
- 不要边跑 final100 边改 SPIR/candidate count/cutoff；
- 不要为了凑 1000 放宽 QC 而不版本化；
- 不要把 ProteinMPNN / NA-MPNN 的 one-sided task 写成与 DM-ICF 完全同信息条件；
- 不要把 model intervention 写成 biological causality；
- 不要把 YAML 中未执行的字段写进方法；
- 不要增加新的“大模型 trick”来掩盖基础合同问题；
- 不要在 P0 未结束前启动长时间 A800/RTX PRO 6000 训练。

---

# 8. 最终交付物

修复完成后至少交付：

```text
artifacts/review2/P0_FIX_REPORT.md
artifacts/review2/SCIENTIFIC_PROTOCOL_V2.md
artifacts/review2/END_TO_END_SMOKE_REPORT.md
artifacts/GO_NO_GO.md
artifacts/FROZEN_FINAL_CONFIGURATION.yaml
```

并更新：

```text
README.md
RUNBOOK.md
docs/SELF_AUDIT.md
docs/DATA_PIPELINE.md（如接口字段发生变化）
```

最后给用户一个不超过 2 页的执行摘要：

```text
已修什么
还剩什么
现在能不能开始数据下载
现在能不能开始正式 GPU 训练
下一条应执行的命令
```

**核心原则：现在的目标不是再增加复杂度，而是让每一个已经写下来的科学主张，都有唯一、真实、可复现的执行路径。**
