# PR Mini-Pilot

A strict small-data rehearsal of fixed-backbone Protein/RNA inverse folding and Protein-RNA co-design.

The goal is not to make a toy demo. The pilot is deliberately small in sample count but complete in scientific logic:

- **1,000 Protein structures** for the Protein structural prior and an official ProteinMPNN-from-scratch reference;
- **1,000 RNA structures/views** for the RNA structural prior and an official NA-MPNN/MPNN-fixbb-from-scratch reference;
- **1,100 experimental Protein-RNA complexes**: 1,000 development + **100 immutable strict final-test complexes**.

Within the 1,000-complex development pool, 900/100 is used only for model/epoch selection. Reported final checkpoints are restarted and refit on the entire 1,000 development complexes using the **same development schedule prefix**, then the final 100 are opened.

## Scientific model

The working hypothesis is

```text
sequence preference
=
intramolecular structural prior
+
local cross-molecular selection
```

The cross-molecular module is the Dynamic Multiscale Interfacial Compatibility Field (DM-ICF):

```text
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

- `C`: global learned 20x4 amino-acid/nucleotide compatibility anchor; **small-random initialized**;
- `DeltaC_ij`: context-dependent 20x4 residual generated from Protein/RNA hidden states and rich PR geometry;
- `alpha_ij`: geometry-aware neighbourhood relevance.

Empirical AA/base PMI is **never used to initialize the primary C**. It is an independent post-hoc/statistical reference.

## Primary stage semantics

```text
P       Protein structural prior
R       RNA structural prior
C       C only; priors/Delta/alpha frozen
Delta   q(interaction) + DeltaC; C/priors/alpha frozen
Alpha   relevance residual + tau only; q/DeltaC/C/priors frozen
Joint   contextual heads + low-LR pretrained coordination; global C remains frozen
```

Scratch joint control is different by design: the same full architecture starts randomly and **all parameters are trainable from step 0**.

## Two different notions that must never be confused

### Canonical biological interface

Frozen during coordinate screening from **full-heavy-atom Protein-RNA contacts <= 6 A**. These residue IDs define interface/non-interface loss and evaluation for every model.

### DM-ICF PR message graph

A richer **8 A + neighbour-cap** graph used only as the model receptive field. Changing this graph cannot relabel the canonical interface.

## Strict split policy

The final 100 are frozen first as whole connected components under:

- Protein P30;
- RNA R80;
- Rfam family;
- exact sequence/mother-sample identity.

Final-test P30/R80/Rfam/exact neighbours are then purged from the 1,000-Protein and 1,000-RNA structural-prior candidate pools before those pools are sampled.

This is a **bilateral strict-OOD test**, not an IID random test.

## Official external references

`third_party/LOCK.json` freezes:

- `dauparas/ProteinMPNN` at an immutable commit;
- `baker-laboratory/NA-MPNN` at an immutable commit.

Primary external references are trained from random initialization on the exact frozen 1,000 single-molecule source pools. They remain **one-sided structural references**: ProteinMPNN does not see RNA partner identity, and the RNA-only NA-MPNN reference does not see Protein partner identity. Cross-partner claims therefore rely on same-data internal controls and interventions, not these external baselines alone.

## Confirmatory claims are deliberately small

Only four hypotheses enter the primary Holm family:

1. **H1** — Full DM-ICF improves canonical-interface normalized NLL over dual structural priors.
2. **H2** — Full DM-ICF improves over both partner-blind and geometry-only capacity controls.
3. **H3** — Contextual field (`C + DeltaC + alpha`) improves over global-C-only.
4. **H4** — Composition-preserving partner scrambling increases interface NLL.

The larger robustness/interpretability/sampling battery is secondary or exploratory.

## Compute-tiered final evaluation

- **Tier A**: all primary seeds x all 100 final complexes; only core metrics needed for the main claims.
- **Tier B**: expensive counterfactual/field/robustness/order/SPIR battery on the predeclared `analysis_seed` only, using the smaller predeclared ablation candidate budget.
- A development-only runtime profiler estimates final GPU cost **before** final100 is inspected.

## Repository map

```text
configs/pilot.yaml
  executable defaults + declarative frozen protocol

docs/DATA_PIPELINE.md
  RCSB/Rfam -> screen -> annotate -> freeze -> audit

docs/CODEX_EXECUTION_V3.md
  exact execution instructions for an agent

docs/PROJECT_STATUS_FOR_YIHENG.md
  user-facing project status and interpretation

docs/REVIEW3_FINAL_AUDIT.md
  review findings, fixes, remaining deliberate limitations

src/pr_pilot/data/
  discovery/download/screening/clustering/freeze/frozen audits
src/pr_pilot/model/dmicf.py
  structural priors + C + DeltaC + alpha
src/pr_pilot/training/
  corruption, balanced loss, stage ownership, controls, refit
src/pr_pilot/inference/
  mixed/directional decoding + SPIR
src/pr_pilot/evaluation/
  Tier A, full Tier B, robustness, empirical contacts, DeltaC drift,
  runtime profiling, broad descriptive comparisons, confirmatory H1-H4 statistics

tools/
  official baseline prep/train/eval/preflight
  pilot orchestration, controls, audits, profiling and statistics

tests/
  scientific/data/runtime/inference/review3 contracts
```

## Current execution gate

Do **not** start long GPU training merely because the repository imports.

Before formal runs, all of the following must pass:

```bash
python -m compileall -q src tests tools
python -m pytest -q
python tools/audit_config_usage.py --config configs/pilot.yaml
ruff check src tests tools --select E9,F63,F7,F82
```

After real data are frozen, additionally run:

```bash
pr-pilot audit-data --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 --out artifacts/data_audit

python tools/audit_frozen_complexes.py --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 --out artifacts/data_audit/canonical_interface

python tools/post_freeze_similarity_audit.py \
  --manifest-root manifests/pilot_v1 --out artifacts/data_audit/local_similarity
```

Then prepare official baselines and run their CPU preflight before GPU training.

The authoritative execution order is in **`RUNBOOK.md`**.

## Scientific non-negotiables

- no final100 feedback into architecture, cutoffs, epochs, SPIR settings, candidate budget or plotting-driven model selection;
- no native Protein side-chain or RNA base-identity atom leakage into our structural priors;
- PI/PN/RI/RN losses are mean-balanced and Protein/RNA cross-polymer combination is normalized by `log(20)` / `log(4)`;
- `DeltaC` starts exactly at zero; learned alpha residual starts exactly at the distance prior;
- primary global C is randomly initialized, trained as a clean anchor and frozen thereafter;
- canonical interface labels are independent of the DM-ICF PR graph;
- predicted structures are excluded from this mini-pilot;
- predictor confidence is never called binding energy;
- model intervention is never called biological causality;
- a partially executed pipeline is never reported as a completed experiment.
