# Implementation Contract v3 — exact tensors, stage ownership and leakage boundaries

This document is the current bridge between the scientific design and executable code. Earlier contracts that defined `interface` from the PR message graph, double-centered C during training, or allowed C/DeltaC to co-adapt in Alpha/Joint are superseded.

## 1. Polymer tensors

Protein graph:

```text
node_x          float [NP, FP] sequence-neutral
edge_index      long  [2, EPP]
edge_x          float [EPP, FPP]
sequence        long  [NP] 0..19, supervision/context token only
interface       bool  [NP] canonical biological interface label
valid           bool  [NP]
fixed           bool  [NP]
reference_xyz   float [NP,3]
chain_index     long  [NP]
residue_ids     metadata
```

Protein structural geometry may use N/CA/C/O, virtual CB, local frames, torsions, edge distances and missingness. Native side-chain identity is not an encoder feature.

RNA graph:

```text
node_x          float [NR, FR] sequence-neutral
edge_index      long  [2, ERR]
edge_x          float [ERR, FRR]
sequence        long  [NR] 0..3 in project order A,U,G,C
interface       bool  [NR] canonical biological interface label
valid/fixed/reference_xyz/chain_index/residue_ids
```

Allowed RNA structural atoms:

```text
P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' O2' C1'
```

Native base-ring identity atoms are prohibited from structural-prior inputs.

## 2. PR message graph

Sparse PR tensors:

```text
protein_index        [EPR]
rna_index            [EPR]
edge_features        [EPR, FPR]
effective_distance   [EPR]
```

Primary rich geometry uses:

- Protein N/CA/C/O/virtual-CB;
- 12 RNA sugar/phosphate atoms;
- all 5×12 pair distance RBFs + explicit presence masks;
- displacement in both local frames;
- relative frame rotation.

Default message receptive field:

```text
cutoff = 8 A
max partner neighbours = 12 per side, unioned
```

This graph is **not** the reporting-interface definition.

## 3. Canonical interface versus design-time refinement region

### Canonical interface

`graph.interface` is computed from original, unaugmented full-heavy-atom Protein/RNA coordinates at the frozen 6-A contact threshold.

It is used for:

- PI/RI training-loss grouping;
- interface/non-interface NLL and recovery;
- external baseline position labels;
- confirmatory interface endpoints.

It must not change when:

- PR graph cutoff changes;
- PR max-neighbour cap changes;
- training coordinate noise changes.

### Design-time/SPIR region

Inference cannot require native Protein side chains or native RNA base atoms. SPIR therefore derives reopen-eligible positions from nodes participating in the sequence-neutral PR message graph supplied by the target backbones.

Never use the canonical native-heavy-atom interface mask to give the generative sampler extra information.

## 4. Model decomposition

```text
q_ij = G_PR(Pi_P hP_i + Pi_R hR_j + f_e(e_ij))
C in R^(20x4)
DeltaC_ij in R^(20x4)
score_ij = -d_eff/tau + residual_alpha(q_ij)
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

Protein correction:

```text
Delta zP_i(a) = lambda_P sum_j alpha^P_ij [C(a,b_j) + DeltaC_ij(a,b_j)]
```

RNA is symmetric. Unknown partner tokens contribute zero through the `known` mask.

## 5. C / DeltaC gauge

Training removes only the single global all-entry scalar offset:

```text
X <- X - mean(X)
```

This retains potentially meaningful row/column main effects in the shared bidirectional 20×4 potential.

**Double-centering is post-hoc only** for interaction-only visualization/comparison:

```text
X_interaction = X - row_mean - column_mean + grand_mean
```

Do not apply double-centering in the forward path.

`C` starts from small zero-centred random values. Empirical frequencies/PMI never initialize the primary C.

`DeltaC` output projection is exactly zero-initialized. No explicit Frobenius penalty is applied in the primary pilot.

## 6. Alpha semantics

```text
score_ij = -d_eff/tau + DeltaScore_ij
```

`DeltaScore` starts at zero, so initial alpha is a distance prior. Protein←RNA and RNA←Protein use the same edge score but normalize over their respective neighbourhoods.

Alpha reads structural hidden states and geometry, not partner token identity directly.

Interpretation: learned relation relevance within the model. It is not a physical force, binding energy or proof of biological causality.

## 7. Strict stage ownership

### Protein prior

Trainable:

```text
protein encoder + token-context decoder + head
```

Forbidden: RNA geometry/sequence, PR edges, native AA as structural node feature.

### RNA prior

Trainable:

```text
RNA encoder + token-context decoder + head
```

Forbidden: Protein context, PR edges, native base-ring identity atoms.

### Global C

Trainable:

```text
C only
```

Fixed:

```text
both priors
DeltaC = 0
learned alpha residual disabled
lambda_P = lambda_R = 1
```

### DeltaC

Trainable:

```text
G_PR + DeltaC head
```

Fixed:

```text
priors
C
learned alpha residual
lambda = 1
```

### Alpha

Trainable:

```text
relevance score head + tau
```

Fixed:

```text
priors
C
G_PR
DeltaC
lambda = 1
```

This is intentionally stricter than earlier drafts: Alpha should answer “who matters?” without also redefining the contextual matrix.

### Joint

Trainable:

```text
G_PR / DeltaC / alpha
bounded lambda_P/lambda_R
token-context decoders + heads
pretrained encoders under gradual output-to-input unfreezing
```

Frozen:

```text
Stage-C global C anchor
```

Scratch-joint control is different: random-initialized encoders are all trainable from step 0.

## 8. Loss contract

For every sample/task, construct semantic group means before combination:

```text
PI = mean CE(masked Protein interface)
PN = mean CE(masked Protein non-interface)
RI = mean CE(masked RNA interface)
RN = mean CE(masked RNA non-interface)
```

Normalize alphabet scales:

```text
Protein group / log(20)
RNA group / log(4)
```

Then average available interface/non-interface groups within each polymer. Joint loss gives Protein and RNA equal polymer weight.

Never pool all tokens and sum raw CE across chains/polymers.

### Current pilot limitation

This is polymer/interface group-balanced, **not additionally per-individual-chain balanced**. Do not describe it as a per-chain-balanced objective. Report single-chain versus multi-chain performance as a secondary stratification. Reconsider explicit per-chain balancing before full-scale training if multi-chain examples become common.

## 9. Corruption/augmentation contract

Primary:

- variable 10–100% corruption with explicit non-zero full-mask probability;
- random + local/spatial patch modes;
- wrong-token corruption;
- interface-upweighted sampling;
- 0.10-A Gaussian coordinate noise during training only;
- 5% intra-edge dropout;
- 5% PR-edge dropout once contextual interaction learning begins;
- light DropPath;
- Protein/RNA prior label smoothing 0.05;
- Stage-C prediction uses no label smoothing.

Canonical interface labels are computed from clean coordinates even when training geometry is augmented.

## 10. Validation and refit contract

Prior and conditional stages select by the registered validation normalized NLL.

Joint selection combines:

```text
Protein conditional canonical-interface normalized NLL
RNA conditional canonical-interface normalized NLL
sequential teacher-forced joint normalized pseudo-NLL
```

Sequential joint validation scores a token while unknown, then reveals its native token only to later positions. Fixed mixed/Protein-first/RNA-first orders are used on a deterministic validation subset.

Full-1,000 refit interprets selected epoch K as a **prefix of the original max-epoch schedule horizon H**. Curriculum, cosine LR and unfreezing progress remain `epoch/H`; they are not compressed into K.

## 11. External baseline contract

Official repositories are pinned in `third_party/LOCK.json`.

ProteinMPNN and NA-MPNN use exactly the same frozen single-polymer IDs as our corresponding priors, first 900/100 development then full-1,000 refit.

They are one-sided structural references and do not by themselves prove partner-coupling value. Causal/mechanistic evidence comes from internal same-data controls and model interventions.

NA-MPNN standardized project probability order is:

```text
A U G C
DA DT DG DC   # under shared-token mode
```

## 12. Final-test/statistics contract

- final 100 are immutable and never tune the method;
- statistical unit = biological complex;
- three primary training seeds receive core evaluation;
- heavyweight generative/mechanistic battery is limited to the predeclared analysis seed;
- primary hypotheses are frozen in `configs/hypotheses.yaml` and Holm-corrected as one family;
- remaining robustness/interpretability analyses are secondary/exploratory;
- use “model-interventional sensitivity”, not biological-causality language, for in-silico perturbations.
