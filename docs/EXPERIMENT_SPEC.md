# PR Mini-Pilot — Frozen Experimental Specification

## 0. Purpose

This repository implements a **small but complete, falsifiable pilot** for fixed-backbone protein/RNA inverse folding and joint protein–RNA co-design. It is intentionally small enough to run before the full dataset is ready, but it must exercise every algorithmic component that will later exist in the full study.

The pilot has four non-negotiable goals:

1. reproduce a small ProteinMPNN protein inverse-folding baseline on exactly 1,000 frozen protein structures;
2. reproduce a small MPNN-fixbb/NA-MPNN RNA inverse-folding baseline on exactly 1,000 frozen RNA structures;
3. train the full DM-ICF pipeline **from scratch** using the same 1,000 protein structures, the same 1,000 RNA structures, and 1,000 protein–RNA complexes;
4. evaluate all frozen models on 100 completely held-out experimental protein–RNA complexes using a deliberately broad battery of sequence, interface, partner-dependence, robustness, calibration, ablation, interpretability and inference tests.

No part of the held-out 100 may be used for model selection, early stopping, threshold selection, architecture selection, noise selection, SPIR selection or plotting choices.

---

## 1. Frozen sample counts

### 1.1 Protein structural-prior pool

- exactly 1,000 experimental protein structures;
- sampled reproducibly from the eligible protein corpus using `pilot_seed`;
- no RNA/DNA/ligand partner is required for the protein-only view;
- native side-chain atoms are never used as model inputs;
- ProteinMPNN baseline and DM-ICF protein prior receive **the identical manifest**.

Internal development split:

- 900 train;
- 100 validation.

After all hyperparameters are frozen, an optional final-refit mode may train on all 1,000 structures. The selected hyperparameters must not change after this refit.

### 1.2 RNA structural-prior pool

- exactly 1,000 experimental RNA structures;
- same sampling and manifest-freezing principles as protein;
- only sequence-neutral sugar–phosphate geometry is exposed to our DM-ICF RNA prior;
- native base-ring atoms/N1/N9 must not leak nucleotide identity into our model view;
- MPNN-fixbb/NA-MPNN baseline and DM-ICF RNA prior receive **the identical RNA manifest** to the extent compatible with each upstream baseline’s required preprocessing.

Internal development split:

- 900 train;
- 100 validation.

### 1.3 Protein–RNA complex pool

Start from exactly 1,100 eligible **experimental** protein–RNA complexes after QC and biological-assembly parsing.

Freeze once:

- `complex_dev_1000.tsv`: 1,000 complexes;
- `complex_test_100.tsv`: 100 complexes.

The 100 test complexes are immutable. They are not available to any training script except through evaluation-only dataloaders protected by `assert_test_only=True`.

Development split:

- 900 complex train;
- 100 complex validation.

Optional final-refit:

- retrain the frozen configuration on all 1,000 development complexes;
- evaluate exactly once on the 100 test complexes.

---

## 2. Leakage and sampling requirements

Even though this is a pilot, it must rehearse the full study’s leakage controls.

### 2.1 Required grouping fields

Every record should carry, when available:

- protein exact-sequence hash;
- protein P90/P40/P30 cluster IDs;
- RNA exact-sequence hash;
- RNA R90/R80 cluster IDs;
- Rfam family/clan;
- PDB/assembly/mother-sample identifier;
- structure release date;
- biological source/type;
- interface fingerprint or at minimum a contact-map hash.

### 2.2 Test-set construction

The final 100 should be selected before training and should preferentially satisfy bilateral novelty:

- no protein P30 cluster overlap with complex development data;
- no RNA Rfam family overlap where family labels are available;
- no exact protein or RNA sequence overlap;
- all conformers/mutation-series/near-duplicate assemblies stay in one split;
- test is experimental structure only.

For the tiny pilot, if strict bilateral isolation cannot produce 100 samples, **do not silently relax criteria**. Write the relaxation to `artifacts/data_audit/split_relaxations.json` and report the exact number satisfying each strict criterion.

### 2.3 Single-molecule pretraining leakage audit

The protein/RNA 1,000 pools should be sampled after the complex test set is frozen. A strict mode removes test-neighbour protein/RNA structures from prior pretraining. A pragmatic mode may retain broad pretraining but must report nearest-neighbour overlap. Main pilot conclusions about cross-molecular generalization should use strict mode whenever feasible.

---

## 3. Baseline A — ProteinMPNN on 1,000 proteins

Upstream: `https://github.com/dauparas/ProteinMPNN`

The repository does not copy or silently modify ProteinMPNN. `third_party/proteinmpnn/README.md` records the pinned upstream commit and exact commands.

Required workflow:

1. convert the frozen protein manifest to upstream ProteinMPNN training format;
2. train from random initialization using only the 900 protein-train structures;
3. choose checkpoint only on the 100 protein-validation structures;
4. optionally refit on all 1,000 with the frozen configuration;
5. export a standardized prediction file understood by our evaluator.

Required metrics:

- token NLL;
- perplexity;
- sequence recovery;
- confidence/calibration;
- recovery by length bin;
- recovery by structural environment when annotations are available;
- coordinate-noise robustness at 0, 0.05, 0.10 and 0.20 Å.

This baseline answers only: **can a small protein-only fixed-backbone model learn a useful structural prior from 1,000 proteins?**

---

## 4. Baseline B — MPNN-fixbb / NA-MPNN on 1,000 RNAs

Upstream: `https://github.com/baker-laboratory/NA-MPNN`

The OpenKnot literature uses “MPNN-fixbb” for fixed-backbone MPNN RNA designs; the public NA-MPNN repository is the reproducible upstream implementation used by this pilot wrapper.

Required workflow mirrors ProteinMPNN:

1. freeze 900/100 RNA train/validation;
2. convert structures to upstream format;
3. train from random initialization;
4. checkpoint on RNA validation only;
5. export standardized logits/predictions when technically available.

Required RNA metrics:

- 4-class token NLL;
- normalized NLL (`NLL/log(4)`);
- sequence recovery;
- recovery by paired/unpaired status if geometry-derived annotations exist;
- recovery by RNA length bin;
- nucleotide composition drift;
- coordinate-noise robustness;
- leakage audit demonstrating that the DM-ICF RNA input view itself cannot directly read native A/U/G/C identity.

---

## 5. DM-ICF model to implement in the pilot

Core hypothesis:

`sequence preference = intramolecular structural prior + local cross-molecular selection`

### 5.1 Protein prior

`hP = E_P(B_P)`

`zP_struct = W_P hP + b_P`

Input is sequence-neutral N/CA/C/O geometry plus virtual CB and geometric graph features.

### 5.2 RNA prior

`hR = E_R(B_R)`

`zR_struct = W_R hR + b_R`

Input is sequence-neutral sugar–phosphate geometry only. Native base atoms that leak A/U/G/C are prohibited.

### 5.3 Rich PR edge representation

For each PR edge `(i,j)`:

`q_ij = G_PR(Pi_P hP_i, Pi_R hR_j, f_e(e_ij))`

`e_ij` must include more than one reference distance. The first implementation should expose:

- multi-atom protein-backbone to RNA sugar/phosphate RBF distances;
- displacement in protein local frame;
- displacement in RNA local frame;
- relative frame rotation;
- chain/type masks;
- missing-atom masks.

A distance-only ablation must be supported through config.

### 5.4 Global compatibility matrix C

`C ∈ R^(20×4)`

Final decision for this project:

- **small random zero-centred initialization**;
- never initialize from empirical frequencies or PMI;
- double-center/gauge-fix C for interpretability;
- learn C first while both structural priors are frozen;
- empirical PMI is post-hoc validation only.

### 5.5 Contextual residual

`DeltaC_ij = W_delta q_ij + b_delta`, shape `20×4`.

Rules:

- final projection zero-initialized;
- no explicit Frobenius penalty forcing `DeltaC` to be small;
- ordinary network weight decay is allowed;
- double-centering is applied for identifiability.

### 5.6 Relational relevance alpha

`score_ij = -d_eff_ij/tau + DeltaScore_ij`

`DeltaScore_ij = W_alpha q_ij + b_alpha`

- `DeltaScore` final projection zero-initialized;
- alpha is normalized separately for Protein←RNA and RNA←Protein neighbourhoods;
- alpha reads geometry/structural hidden states but not partner token identity directly;
- a weak early entropy regularizer may be used and must anneal to zero.

### 5.7 Final logits

Protein direction:

`Delta zP_i(a) = lambda_P * sum_j alpha_ij * [C(a,b_j)+DeltaC_ij(a,b_j)]`

`zP_i = zP_struct_i + Delta zP_i`

RNA direction is symmetric.

If a partner token is unknown during joint decoding, that edge contributes zero interaction correction until the partner token becomes known.

---

## 6. DM-ICF staged training — none may be skipped

The pilot must run all stages even if each stage is intentionally short.

### Stage P — protein prior

Data: protein train 900.

Train: protein encoder/head only.

Validation: protein val 100.

### Stage R — RNA prior

Data: RNA train 900.

Train: RNA encoder/head only.

Validation: RNA val 100.

### Stage C — global compatibility

Data: complex train 900 only.

Freeze: both prior encoders and prior heads.

Train: C and only explicitly configured scalar interface gains.

Tasks sampled 1:1:

- RNA known → predict masked protein interface positions;
- protein known → predict masked RNA interface positions.

No partner-scramble training loss. Scramble is evaluation-only.

### Stage Delta — contextual residual

Freeze: priors and C.

Train: PR geometry encoder + DeltaC head.

DeltaC output begins at exactly zero.

### Stage Alpha — relational relevance

Freeze: priors and C.

Train: PR geometry encoder + DeltaC + alpha head.

Use distance-prior initialization, weak temporary entropy regularization, 5% PR edge dropout and configurable 10–20% partner-token dropout.

### Stage Joint — final coordination

Tasks:

- protein conditional;
- RNA conditional;
- joint masked design.

Task schedule:

- start 2:2:1;
- finish 1:1:1.

Gradual unfreezing:

- heads/interactions first;
- top prior layers next;
- lower prior layers last;
- discriminative LR / layer-wise LR decay.

C may be unfrozen only at a very small learning rate.

---

## 7. Loss implementation — strict requirements

This is a critical implementation area. The code must fail loudly if group normalization is accidentally replaced with a raw token sum.

Raw groups per sample:

- protein interface `P,I`;
- protein non-interface `P,N`;
- RNA interface `R,I`;
- RNA non-interface `R,N`.

Within each non-empty group: mean CE over tokens.

When combining protein and RNA, normalize alphabet scales:

- protein: `CE/log(20)`;
- RNA: `CE/log(4)`.

For a joint sample with all four groups available:

`L_joint = 0.25*(L_PI + L_PN + L_RI + L_RN)`.

If a group is empty, renormalize over the remaining valid groups.

A minibatch contains one task type; task balance is controlled by the sampler, not by summing three giant losses.

Predicted structures are outside this initial pilot and must not silently enter these manifests.

The training logger must record:

- each raw group NLL;
- each normalized group NLL;
- total loss;
- regularizer contributions separately;
- gradient norms by major module;
- task-gradient cosine similarities during joint coordination.

PCGrad is **off by default**. It may only be activated in an explicit ablation after sustained gradient conflicts are demonstrated.

---

## 8. Required augmentation

Configurable and logged:

- coordinate Gaussian noise: main `sigma=0.10 Å`;
- ablations: 0, 0.05, 0.10, 0.20 Å;
- random + local/spatial patch masking;
- variable mask fraction covering approximately 10–100%;
- light mask curriculum;
- 5% spatial edge dropout;
- 5% PR edge dropout after PR learning starts;
- structural-prior label smoothing 0.05;
- no label smoothing in global C stage.

---

## 9. Joint inference and SPIR

Initial joint decoding uses a mixed random order over protein and RNA designable positions.

After both sequences are complete, run **Single-Pass Interface Reconciliation (SPIR)**:

1. freeze all non-interface positions;
2. estimate interface uncertainty using entropy or top1–top2 margin;
3. reopen only the least-confident 20–40% interface positions;
4. update protein conditioned on complete RNA, then RNA conditioned on updated protein;
5. reverse the direction for half the generated candidates;
6. use lower temperature, default search range 0.3–0.7;
7. execute only one cycle in the default model.

Required ablation: no SPIR vs one-pass SPIR vs repeated refinement.

---

## 10. Final 100-complex evaluation — mandatory battery

Every metric must be paired by target whenever possible. Report mean/median, per-target distributions and uncertainty intervals; never report only a global pooled number.

### 10.1 Core sequence prediction

For each method/mode where applicable:

- protein NLL;
- protein normalized NLL;
- protein recovery;
- RNA NLL;
- RNA normalized NLL;
- RNA recovery;
- interface and non-interface separately;
- per-complex macro averages;
- per-token micro averages (secondary, clearly labelled).

### 10.2 Conditional design tests

RNA→Protein:

- DM-ICF vs protein structural prior;
- DM-ICF vs ProteinMPNN where the input definitions are fair;
- interface-specific gains.

Protein→RNA:

- DM-ICF vs RNA structural prior;
- DM-ICF vs MPNN-fixbb/NA-MPNN where fair.

### 10.3 Partner-scramble test

For each native complex:

- preserve backbones;
- scramble partner sequence using composition-preserving and family-matched variants when possible;
- recompute target NLL.

Report:

`DeltaNLL = NLL_scrambled - NLL_native`

separately for interface/non-interface and both directions.

A genuine partner-aware model should show a much larger effect at the interface.

### 10.4 Local counterfactual partner mutation

For contacting partner tokens, substitute each alternative identity while keeping geometry fixed.

Measure:

- KL divergence of target-token distribution before/after mutation;
- KL vs spatial distance;
- contact vs far-control effect;
- directional symmetry (RNA→Protein and Protein→RNA).

### 10.5 C vs empirical PMI

PMI is computed **after training** from an experimental structure set and never used for initialization.

Report:

- learned C heatmap;
- empirical PMI heatmap;
- Pearson correlation;
- Spearman correlation;
- bootstrap CI;
- seed-to-seed C stability;
- shuffled-pair null distribution.

Where data permit, stratify empirical enrichment by base/sugar/phosphate-facing geometry.

### 10.6 DeltaC analysis

Report:

- `||DeltaC_ij||_F / ||C||_F` distribution;
- relation to geometry/contact class;
- examples where contextual correction reverses global preference;
- seed stability;
- effect of replacing rich geometry with distance-only geometry.

### 10.7 Alpha analysis

Report:

- entropy and effective neighbour count `exp(H(alpha))`;
- relation between alpha and distance;
- cases where learned relevance prefers a non-nearest neighbour;
- edge-dropout robustness;
- alpha maps for representative interfaces.

Alpha is a learned relevance coefficient, not causal physical importance.

### 10.8 Robustness

Re-evaluate with coordinate perturbations at 0, 0.05, 0.10, 0.20 and optionally 0.30 Å.

Additional stress tests:

- remove 5/10/20% PR edges;
- hide 10/20/40% known partner tokens in conditional mode;
- perturb decoding order;
- change candidate generation temperature;
- missing-atom masks where supported.

### 10.9 Joint decoding stability

For each complex generate multiple candidates and orders.

Measure:

- recovery/NLL distribution across orders;
- pairwise sequence identity across generated candidates;
- order-sensitivity variance;
- protein-first vs RNA-first directional bias;
- SPIR improvement and diversity retention.

### 10.10 Calibration

For both alphabets:

- expected calibration error;
- Brier score;
- reliability bins;
- confidence vs correctness at interface and non-interface separately.

### 10.11 Sequence diversity / collapse audit

Measure:

- amino-acid composition;
- RNA composition;
- protein net-charge proxy;
- generated pairwise sequence identity;
- entropy per position;
- fraction of near-duplicate candidates;
- enrichment of Lys/Arg at interfaces;
- comparison to native composition.

### 10.12 Ablation ladder

At minimum:

A. scratch joint model, no prior pretraining;

B. + protein/RNA structural priors;

C. + global C;

D. + contextual DeltaC;

E. + learned alpha = full DM-ICF before final joint coordination;

F. + final joint coordination;

G. + SPIR.

Additional ablations:

- random C init vs zero C init;
- rich PR geometry vs distance only;
- no coordinate noise;
- no edge dropout;
- no partner-token dropout;
- no gradual unfreezing;
- fixed-distance alpha vs learned alpha;
- no double-centering;
- repeated refinement vs SPIR.

### 10.13 Data-efficiency pilot inside the 1,000 complexes

Train fixed configurations on nested subsets of complex train:

- 10%;
- 25%;
- 50%;
- 100%.

Compare scratch vs dual-prior initialization. This is the direct pilot test of whether single-molecule structural priors reduce complex-data requirements.

### 10.14 Optional external structure-consistency evaluation

This repository provides adapters/config contracts for Boltz/AlphaFold3/other independent predictors, but external predictors are not required for unit tests and are never used as ground-truth training labels in this pilot.

When run, report protein backbone agreement, RNA backbone agreement, interface geometry/contact recovery and predictor confidence separately. Do not collapse them into a fake “binding energy”.

---

## 11. Statistical analysis on the held-out 100

Required defaults:

- paired bootstrap over complexes, not residues, 10,000 resamples;
- paired permutation or Wilcoxon signed-rank as a secondary non-parametric test when appropriate;
- effect sizes and confidence intervals, not p-values alone;
- Holm correction for the small pre-registered primary comparison family;
- Benjamini–Hochberg for exploratory metric families;
- seed variation reported separately from target variation.

Primary pre-registered comparisons:

1. full DM-ICF vs structural-prior-only at protein interface NLL;
2. full DM-ICF vs structural-prior-only at RNA interface NLL;
3. full DM-ICF partner-scramble DeltaNLL at interface vs non-interface;
4. contextual DeltaC model vs global-C model;
5. SPIR vs no-SPIR joint interface NLL/recovery.

No test-set metric may be used to decide whether a model component remains in the final model.

---

## 12. Reproducibility artifacts

Every run must persist:

- git commit;
- full resolved config;
- random seeds;
- exact manifest checksums;
- upstream baseline commit SHAs;
- environment lock information;
- hostname/GPU/PyTorch/CUDA versions;
- epoch/step logs;
- selected checkpoint reason;
- test-run command;
- prediction files;
- metric JSON/CSV;
- statistical-analysis outputs.

Expected top-level artifacts:

```text
artifacts/
  manifests/
  data_audit/
  checkpoints/
  logs/
  predictions/
  metrics/
  statistics/
  figures/
  interpretability/
  external_structure/
```

---

## 13. Hard failure conditions

The pipeline must stop, not warn-and-continue, when:

- a test complex appears in any training manifest;
- an exact protein/RNA test sequence appears in a supposedly strict development manifest;
- native RNA base identity can be inferred because prohibited base atoms entered the DM-ICF RNA input view;
- a loss group is empty and the code divides by zero instead of renormalizing;
- a non-finite loss/gradient occurs;
- C/DeltaC/alpha tensor shapes deviate from contract;
- baseline and DM-ICF manifests differ when the experiment claims a fair comparison;
- a checkpoint is selected using final test metrics;
- predicted structures silently enter the experimental-only pilot.

---

## 14. Definition of “pilot complete”

The pilot is complete only when all of the following exist:

1. frozen manifests and leakage report;
2. trained ProteinMPNN baseline;
3. trained MPNN-fixbb/NA-MPNN RNA baseline;
4. DM-ICF checkpoints for P, R, C, Delta, Alpha and Joint stages;
5. joint sampler + SPIR outputs;
6. all mandatory ablations or explicit machine-readable `NOT_RUN` records with reason;
7. full 100-complex metric table;
8. paired statistical report;
9. C/PMI, DeltaC and alpha interpretability outputs;
10. robustness and decoding-order reports;
11. exact reproduction commands in `RUNBOOK.md`.

A partially trained model is **not** a completed pilot.
