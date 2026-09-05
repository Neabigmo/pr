# Implementation Contract — exact tensors, adapters and training invariants

This document is the bridge between the scientific specification and executable local data code. Any Codex/agent implementation must conform to these contracts.

## 1. Structure sample object

A single complex loader should return a dictionary or dataclass with the following logical fields.

### Protein

```text
protein.node_x          float32 [NP, FP]
protein.edge_index      int64   [2, EPP]
protein.edge_x          float32 [EPP, FPP]
protein.sequence        int64   [NP] values 0..19
protein.interface       bool    [NP]
protein.valid           bool    [NP]
protein.residue_id      metadata length NP
```

Protein node features must be sequence-neutral in DM-ICF structural-prior mode. Native amino-acid identity is a label, not an input.

Allowed geometry includes N/CA/C/O, virtual CB, local frames, backbone torsions and missing-atom masks.

### RNA

```text
rna.node_x              float32 [NR, FR]
rna.edge_index          int64   [2, ERR]
rna.edge_x              float32 [ERR, FRR]
rna.sequence            int64   [NR] values 0..3 in A,U,G,C order chosen globally
rna.interface           bool    [NR]
rna.valid               bool    [NR]
rna.nucleotide_id       metadata length NR
```

Native base-ring identity atoms are prohibited from `rna.node_x` and `rna.edge_x`.

Allowed atom view:

```text
P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' O2' C1'
```

The exact atom vocabulary is implemented in `pr_pilot.data.features`.

### Protein–RNA sparse edges

```text
pr.protein_index        int64   [EPR]
pr.rna_index            int64   [EPR]
pr.edge_features        float32 [EPR, FPR]
pr.effective_distance   float32 [EPR]
```

Primary rich `edge_features` should contain the following information, with explicit missing-value masks:

1. RBF-expanded distances between sequence-neutral protein atoms and sequence-neutral RNA atoms;
2. protein-frame displacement of RNA reference geometry;
3. RNA-frame displacement of protein reference geometry;
4. relative local-frame rotation representation;
5. edge/contact masks and chain metadata.

Distance-only ablation must be constructible by a config switch without changing train/test manifests.

## 2. Interface definition

The exact graph cutoff is a model hyperparameter, not a biological truth. Data audit must record contact statistics at several thresholds before final selection.

Recommended pilot default:

- PR graph radial cutoff: 8 Å on sequence-neutral reference geometry;
- cap: 12 partner neighbours per target node;
- interface label: position participates in at least one valid PR edge under the frozen graph definition.

For empirical chemistry/PMI analysis, a separate heavy-atom contact definition may be used and must be written into the result metadata. Do not conflate the neural graph cutoff with the biochemical contact cutoff.

## 3. Batch corruption object

Every training batch must expose explicit masks:

```text
protein.masked          bool [NP]
protein.known           bool [NP]
rna.masked              bool [NR]
rna.known               bool [NR]
```

Invariant:

```text
known == ~masked
```

for designable positions, except fixed/padded positions which must have an explicit `valid/designable/fixed` mask.

Masking generator supports:

- independent random masks;
- contiguous sequence patches;
- 3D spatial patches;
- full mask;
- interface-upweighted mask;
- wrong-token corruption;
- partner-token dropout.

Every corruption realization must be seeded and optionally reproducible from `sample_id + epoch + global_seed`.

## 4. Stage-specific forward views

### protein_prior

Inputs:

- protein structural geometry only.

Forbidden:

- RNA geometry;
- RNA sequence;
- PR edges;
- native protein sequence as node feature.

Output:

```text
protein_struct_logits [NP,20]
```

### rna_prior

Inputs:

- RNA sugar–phosphate geometry only.

Forbidden:

- protein context;
- PR edges;
- native base-ring atoms;
- native RNA sequence as node feature.

Output:

```text
rna_struct_logits [NR,4]
```

### global_c

Inputs:

- frozen `hP`, frozen `hR`;
- PR topology and simple distance weighting;
- known partner sequence;
- target-side interface mask.

Trainable:

```text
C [20,4]
lambda_P
lambda_R
```

Frozen/disabled:

```text
DeltaC = 0
learned alpha disabled
prior encoders frozen
```

### delta_c

Trainable:

```text
G_PR
DeltaC projection
```

Frozen:

```text
C
protein prior
RNA prior
alpha residual
```

Step-zero invariant:

```text
DeltaC_ij == 0 for every edge
```

### alpha

Trainable:

```text
G_PR
DeltaC
alpha residual score
```

Step-zero alpha invariant:

```text
alpha == neighbourhood_softmax(-distance/tau)
```

### joint

Full model can be unfrozen gradually. The run metadata must record exactly when each parameter family became trainable.

## 5. C and DeltaC gauge fixing

Both use double centering:

```text
X[a,b] <- X[a,b]
          - mean_b X[a,b]
          - mean_a X[a,b]
          + mean_ab X[a,b]
```

Reason: additive row/column constants are poorly identifiable in conditional logits and make heatmap interpretation unstable.

This operation is not a magnitude penalty.

## 6. Alpha semantics

`alpha` is a geometric/relation relevance coefficient.

It answers:

> among multiple partner neighbours, which edges contribute more strongly to the local compatibility correction?

It is **not** claimed to be:

- a physical force;
- a binding energy;
- a causal importance score.

Alpha reads structural hidden states and PR geometry; partner token identity acts through `C + DeltaC`, not through alpha.

## 7. Loss implementation

Per-sample raw semantic groups:

```text
PI = mean CE over masked protein interface
PN = mean CE over masked protein non-interface
RI = mean CE over masked RNA interface
RN = mean CE over masked RNA non-interface
```

Normalized:

```text
PI /= log(20)
PN /= log(20)
RI /= log(4)
RN /= log(4)
```

Protein task:

```text
LP = mean(non-empty PI, PN)
```

RNA task:

```text
LR = mean(non-empty RI, RN)
```

Joint:

```text
LJ = 0.5 * (LP + LR)
```

A missing semantic group is omitted and remaining weights renormalize. Never fill an absent group with zero, because that would artificially lower loss.

## 8. Validation selection

Checkpoint metrics are pre-specified by stage and never use the final 100 test complexes.

Recommended joint composite:

```text
mean(
  protein_interface_NLL/log(20),
  RNA_interface_NLL/log(4),
  joint_normalized_NLL
)
```

Recovery is reported but not the main early-stopping metric.

## 9. Baseline fairness contract

ProteinMPNN and DM-ICF protein prior:

- identical frozen protein sample IDs;
- both from random initialization in the primary small-data comparison;
- same 900/100 split;
- coordinate noise experiments explicitly matched when possible;
- no published ProteinMPNN weights in the primary from-scratch baseline.

MPNN-fixbb/NA-MPNN and DM-ICF RNA prior:

- identical frozen RNA sample IDs to the maximum technically valid extent;
- both from random initialization in primary comparison;
- same 900/100 split;
- no hidden extra RNA training corpus.

If upstream preprocessing excludes a frozen sample that our model accepts, the comparison must produce both:

1. **intersection benchmark** — exactly shared usable samples;
2. **intended-pool report** — documents why samples were lost.

Never silently replace excluded samples with new random ones for only one method.

## 10. Test-time prohibitions

After the final 100 are unblinded:

Forbidden:

- architecture changes;
- mask schedule changes;
- cutoff/neighbour changes;
- coordinate noise selection;
- SPIR fraction selection;
- new checkpoint selection;
- choosing a favourable seed;
- deleting difficult complexes because a metric looks poor.

Allowed:

- running the pre-registered metric battery;
- correcting a genuine code bug if the correction is documented and all affected methods/metrics are rerun;
- reporting prespecified subgroup analyses.

## 11. Required tests before GPU training

Unit tests must cover:

- C shape and initialization;
- DeltaC exact zero initialization;
- double-centering;
- alpha neighbourhood normalization;
- distance-prior alpha at initialization;
- unknown partner -> zero interaction correction;
- RNA base-atom leakage guard;
- balanced four-group loss under extreme group-size imbalance;
- no NaN on absent PI/PN/RI/RN groups;
- deterministic manifest sampling;
- test leakage detection;
- task/stage contract violations;
- fixed-site masks;
- SPIR never modifies non-interface positions.

## 12. External predictors

External structure predictors are optional evaluators. Adapter contracts should consume generated FASTA/sequences plus target complex metadata and output a standardized table containing:

```text
sample_id
candidate_id
predictor
predictor_version
seed
protein_backbone_metric
rna_backbone_metric
interface_metric
confidence_metric
raw_output_path
```

Do not train DM-ICF against these values in the primary mini-pilot.
