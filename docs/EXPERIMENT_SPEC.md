# PR Mini-Pilot — Frozen Experimental Specification v3

## 0. Purpose

This pilot is a small but complete rehearsal of fixed-backbone Protein/RNA inverse folding and Protein–RNA co-design. It is designed to falsify the central hypothesis before the full data programme is launched:

```text
sequence preference
= intramolecular structural prior
+ local cross-molecular selection
```

The cross-molecular selection field is:

```text
Gamma_ij = alpha_ij (C + DeltaC_ij)
```

The pilot must exercise the complete data, training, inference, control and statistical pipeline. A partially run model demo is not a completed pilot.

## 1. Frozen scale

### Protein structural-prior pool

Exactly 1,000 experimental Protein structures:

```text
900 development train
100 P30-component-disjoint validation
```

ProteinMPNN and our Protein prior receive identical frozen IDs. Native side-chain identity is never a structural-prior feature in our model.

### RNA structural-prior pool

Exactly 1,000 experimental RNA structural views:

```text
900 development train
100 validation disjoint under R80 OR Rfam connected components
```

NA-MPNN and our RNA prior receive identical frozen IDs. Our RNA structural view exposes sugar/phosphate geometry but no base-ring identity atoms.

Standalone RNA is preferred. RNA chains extracted from other screened experimental Protein–RNA complexes are allowed with the Protein partner removed, but an extracted view sourced from any of the frozen 1,100 downstream complexes is excluded.

### Protein–RNA complex pool

Exactly 1,100 eligible experimental complexes after screening:

```text
1000 development
  900 train
  100 strict bilateral validation
100 immutable final holdout
```

Complex components are linked by any shared Protein P30, RNA R80 or Rfam family. Final 100 are frozen before the two prior pools are sampled.

## 2. Non-negotiable leakage controls

Final test must have no development overlap by:

- exact Protein sequence;
- exact RNA sequence;
- mother sample;
- any constituent Protein P30;
- any constituent RNA R80;
- any constituent Rfam family.

The two prior pools are then purged against final-test exact sequence/family neighbours.

RNA complex-chain fallback views sourced from the entire frozen 1,100 downstream complex pool are additionally removed, preventing exact downstream RNA-backbone pre-exposure.

If strict component sizes make exactly 100 final complexes impossible, enlarge the eligible candidate universe or create a new explicitly relaxed pilot version. Never split a homologous component silently.

## 3. Structural screening

Primary limits:

```text
Protein single prior       30..1000 residues
RNA single prior            5..500 nt
complex Protein            30..1000 residues
complex RNA                 5..500 nt
complex total              <=1000 tokens
resolution                 <=4.0 A when applicable
```

NMR without a conventional resolution is allowed and recorded as a distinct method stratum.

Ribosome/spliceosome-like assemblies are excluded from the primary pilot. Complex samples must show real Protein–RNA heavy-atom contact and at least three contact pairs under the 6-A screening definition.

## 4. Two interface concepts

### Canonical supervised/reporting interface

- full heavy-atom Protein/RNA contact;
- 6 A;
- computed on clean, unaugmented experimental coordinates;
- used for PI/RI loss groups, interface recovery/NLL, baseline position labels and confirmatory endpoints.

### DM-ICF PR message graph

- sequence-neutral Protein N/CA/C/O/virtual-CB against RNA sugar/phosphate geometry;
- default 8-A cutoff;
- max 12 neighbours per side, unioned;
- used as the cross-molecular receptive field.

The PR graph must never redefine the reported interface. Conversely, SPIR/design-time position selection must use the sequence-neutral PR graph and not native side-chain/base contact labels.

## 5. Official one-sided references

### ProteinMPNN

Pinned official `dauparas/ProteinMPNN` commit from `third_party/LOCK.json`.

Protocol per seed:

1. exact 900/100 frozen Protein development split;
2. random-initialized upstream training;
3. select epoch using Protein validation only;
4. restart from random initialization;
5. train on all 1,000 Protein structures for the selected development epoch count;
6. final one-sided evaluation uses Protein backbone only.

### NA-MPNN / MPNN-fixbb RNA reference

Pinned official `baker-laboratory/NA-MPNN` commit.

Same 900/100 -> selected epoch/pass -> full-1,000 refit logic. Final one-sided RNA evaluation sees RNA only.

Standardized project probability order is `AUGC`. Under NA-MPNN shared tokens, columns map `AUGC = DA,DT,DG,DC`.

These baselines answer how strong standard one-sided structural design is on the same frozen polymer pools. They do not receive partner identity and are not the sole controls for cross-partner coupling.

## 6. DM-ICF model

Protein and RNA encoders are independent and sequence-neutral. Their hidden dimensions are aligned for interaction projection.

For each PR edge:

```text
q_ij = G_PR(Pi_P hP_i + Pi_R hR_j + f_e(e_ij))
```

Primary rich `e_ij` contains 5×12 atom-pair RBF distances, explicit missing masks, two local-frame displacements and relative local-frame rotation.

Global compatibility:

```text
C in R^(20x4)
```

- small random zero-centred initialization;
- no empirical frequency/PMI initialization;
- only global scalar centering during forward;
- row/column main effects retained;
- double-centering reserved for post-hoc interaction-only views.

Context residual:

```text
DeltaC_ij in R^(20x4)
```

- generated from q_ij;
- exact zero-initialized output head;
- no explicit magnitude penalty in the primary pilot.

Relevance:

```text
score_ij = -d_eff/tau + DeltaScore_ij
alpha = neighbourhood_softmax(score)
```

The learned score residual starts at zero. Protein←RNA and RNA←Protein normalize separately.

Final directional correction uses known partner tokens. An unknown partner contributes zero until it becomes known in sequential joint decoding.

## 7. Mandatory staged training

### P — Protein prior

Train Protein encoder/context decoder/head on the 900 Protein development train set. Select on P30-disjoint validation normalized NLL.

### R — RNA prior

Train RNA encoder/context decoder/head. Select on R80/Rfam-disjoint validation normalized NLL.

### C — global compatibility

Complex train only.

Trainable:

```text
C only
```

Frozen/disabled:

```text
Protein/RNA priors
DeltaC
learned alpha residual
lambda_P = lambda_R = 1
```

Bidirectional conditional interface prediction supervises the same C. No empirical PMI is supplied.

### DeltaC — contextual compatibility

Trainable:

```text
G_PR + DeltaC head
```

Frozen: priors, C, learned-alpha residual, lambdas.

### Alpha — relation relevance

Trainable:

```text
alpha relevance head + tau only
```

Frozen: priors, C, G_PR and DeltaC. This stage isolates “which neighbour matters?” from “what compatibility matrix is present?”.

### Joint — final coordination

C remains frozen as the Stage-C anchor. Context field, bounded lambdas, token-context heads and progressively released pretrained encoder blocks may adapt.

Task curriculum begins approximately 2:2:1 Protein-conditional : RNA-conditional : joint and transitions to 1:1:1.

Scratch-joint control differs deliberately: all random-initialized encoder layers are trainable from step zero.

## 8. Loss

Per-sample semantic groups:

```text
PI PN RI RN
```

Each non-empty group is a mean CE. Protein groups divide by `log(20)` and RNA groups by `log(4)` before cross-polymer combination. Interface/non-interface groups are balanced within a polymer; Protein/RNA receive equal weight in the joint task.

The pilot is not additionally per-chain balanced. This must not be misreported as a chain-balanced loss.

## 9. Augmentation

Primary training uses:

- coordinate Gaussian noise `sigma=0.10 A`;
- variable masking about 10–100%;
- explicit full-mask probability;
- random plus local/spatial patch corruption;
- wrong-token corruption;
- interface-upweighted corruption;
- 5% intra-edge dropout;
- 5% PR-edge dropout in contextual stages;
- light DropPath;
- 0.05 label smoothing for structural priors, none for Stage C.

Canonical interface labels are computed from the clean structure and cannot change due to coordinate augmentation.

## 10. Checkpoint selection and final refit

Joint checkpoint selection combines:

```text
Protein conditional canonical-interface normalized NLL
RNA conditional canonical-interface normalized NLL
sequential teacher-forced joint normalized pseudo-NLL
```

The sequential metric uses fixed mixed, Protein-first and RNA-first orders on a deterministic validation subset. Each target is scored while unknown, then its native token is revealed to subsequent positions.

A development best epoch K is a **prefix of the original schedule horizon H**. Full-1,000 refit restarts from scratch and replays epochs 1..K under the same H-based curriculum/cosine/unfreezing schedule. It must not compress a complete H-epoch schedule into K epochs.

Only full-1,000 refit checkpoints support the primary final report.

## 11. Joint inference and SPIR

Primary joint generation uses mixed Protein/RNA autoregressive order. Protein-first and RNA-first are order controls.

When partner tokens become available, their DM-ICF contribution becomes available immediately.

SPIR is a single low-temperature interface reconciliation pass. Reopen-eligible positions are selected from the sequence-neutral PR graph, ranked by current model uncertainty. Repeated SPIR is an ablation, not the primary method.

Primary generation budget: 64 candidates/complex. Heavy order/SPIR ablation cells: 16 candidates/complex unless changed before final-test opening.

## 12. Internal fairness controls

Required same-data controls:

- scratch joint;
- dual structural priors only;
- + global C;
- + DeltaC;
- + alpha;
- full joint;
- partner-blind;
- geometry-only capacity control;
- C backbone-context control;
- fixed empirical-PMI reference.

The component ladder must use compatible data IDs/seeds and the same development-selection/full-refit discipline.

## 13. Final 100 battery

Every primary seed receives the inexpensive core final evaluation. The predeclared `analysis_seed` receives heavyweight mechanistic/generative analyses.

Core/secondary analyses include:

- Protein/RNA conditional NLL/recovery;
- canonical interface/non-interface decomposition;
- sequential joint metrics;
- partner scrambling;
- partner counterfactual mutation and local KL response;
- independent heavy-atom C-vs-PMI comparison;
- DeltaC context/magnitude and mean-drift audit;
- alpha entropy/effective-neighbour/top-edge removal;
- coordinate noise, PR edge removal and partner hiding;
- order sensitivity;
- SPIR ablation;
- calibration and candidate diversity;
- 10/25/50/100% **nested** complex-data efficiency;
- strict OOD covariate-shift report.

## 14. Confirmatory statistics

`configs/hypotheses.yaml` is frozen before final-test metrics are inspected.

Primary Holm family contains only the predeclared H1–H4/H2 subtests there. Other analyses are secondary/exploratory.

Statistical unit is the biological complex. Seeds and residues are not treated as independent test samples.

For in-silico scrambling/mutation/edge-removal, conclusions are phrased as **model-interventional sensitivity**, not biological causality.

## 15. Interpretation

C and C+DeltaC are learned conditional sequence-compatibility contributions, not thermodynamic binding energies.

Because DeltaC can develop a non-zero population mean even with C frozen, report the Stage-C anchor C, mean DeltaC, alpha-weighted mean DeltaC and an effective `C_eff` when appropriate.

Empirical PMI is computed after training from independent full-heavy-atom experimental contacts; it never initializes the primary C.

## 16. Explicitly deferred from the primary pilot

- predicted-structure augmentation;
- family-aware replacement sampling;
- dynamic token packing;
- automatic PCGrad;
- enumeration of alternative biological assemblies;
- a new per-chain-balanced objective.

These are full-scale follow-up questions, not missing requirements for the controlled mini-pilot.
