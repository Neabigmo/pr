# PR Mini-Pilot — DM-ICF

A small-data but end-to-end pilot for fixed-backbone Protein/RNA inverse folding and joint Protein–RNA co-design.

The scientific question is deliberately narrow:

```text
sequence preference
= intramolecular structural prior
+ local cross-molecular selection
```

The cross-molecular term is the **Dynamic Multiscale Interfacial Compatibility Field (DM-ICF)**

```text
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

with a random-initialized global 20×4 AA/base compatibility anchor `C`, a geometry-conditioned full 20×4 residual `DeltaC_ij`, and a learned neighbour relevance term `alpha_ij`.

## Frozen pilot scale

- 1,000 Protein structural-prior structures;
- 1,000 RNA structural-prior structures;
- 1,100 experimental Protein–RNA complexes;
  - 1,000 development complexes;
  - 100 immutable strict final complexes.

The 900/100 development splits select hyperparameters and **schedule prefix lengths**. Primary final checkpoints are retrained from scratch on all 1,000 frozen development structures while replaying the same selected schedule prefix. The final 100 never participate in model selection.

## Official external references

- `dauparas/ProteinMPNN` — Protein fixed-backbone reference;
- `baker-laboratory/NA-MPNN` — RNA fixed-backbone reference.

Both are pinned by immutable SHA in `third_party/LOCK.json`, trained from random initialization on the exact frozen single-polymer IDs used by our priors, selected on 900/100 development data, and refit on the full 1,000. They are **one-sided structural references**: they do not receive partner identity and therefore are not the causal controls for DM-ICF partner coupling.

The internal causal/mechanistic controls are structural-prior only, global-C only, DeltaC, alpha, partner-blind, geometry-only capacity, partner scrambling and counterfactual perturbations.

## Data pipeline

The repository now contains the real data path; there is no missing local adapter placeholder:

```text
RCSB discovery
-> deterministic oversized download
-> coordinate QC and biological-assembly contact screening
-> RNA-chain-view augmentation for RNA prior candidates when needed
-> joint Protein/RNA clustering with MMseqs2
-> Rfam cmscan annotation
-> final-test-first strict freezing
-> purge final-test families from both prior pools
-> cluster-disjoint 900/100 prior validation splits
-> manifest audit
```

Read `docs/DATA_PIPELINE.md` before downloading data.

### Two different geometric concepts are intentionally separated

**Canonical biological interface**
- full heavy-atom Protein/RNA contacts;
- 6 Å in the pilot;
- used for interface NLL/recovery, loss grouping and baseline position labels;
- independent of model graph hyperparameters.

**DM-ICF PR message graph**
- sequence-neutral 5 Protein × 12 RNA atom geometry;
- default 8 Å cutoff and maximum 12 neighbours per side;
- used only as the interaction receptive field.

Changing the model PR cutoff/cap must not redefine the reported interface.

## Training stages

Primary parameter ownership is strict:

```text
P prior   : Protein encoder/decoder/head
R prior   : RNA encoder/decoder/head
Global C  : C only; priors frozen; fixed distance relevance
DeltaC    : interaction encoder + DeltaC only; C/priors/alpha frozen
Alpha     : relevance/tau only; C/DeltaC/q/priors frozen
Joint     : context field + bounded lambdas + gradual encoder adaptation; C frozen
```

This separation is intentional so `C`, contextual correction and relevance can be interpreted and ablated independently.

Joint checkpoint selection uses conditional Protein/RNA interface NLL plus a deterministic sequential teacher-forced joint pseudo-NLL. It does **not** use a single both-sides-unknown forward pass as a proxy for joint decoding.

## Inference

- conditional Protein design given RNA;
- conditional RNA design given Protein;
- mixed Protein/RNA autoregressive joint design;
- Protein-first/RNA-first order controls;
- one-pass SPIR interface reconciliation, with repeated-SPIR as an ablation.

Primary design candidate budget is 64. Heavy order/SPIR ablation cells use a smaller predeclared 16-candidate budget and are run on the predeclared analysis seed to keep the final battery computationally tractable.

## Confirmatory statistics

`configs/hypotheses.yaml` defines a small primary family before final-100 evaluation. Holm correction is applied only to that family. The many robustness, PMI, DeltaC, alpha, decoding and calibration analyses are secondary/exploratory rather than being mislabeled as dozens of independent primary hypotheses.

Statistical unit = **biological complex**, not residue/token.

Use “model-interventional sensitivity” rather than biological causality for partner scramble, edge removal and counterfactual in-silico interventions.

## Repository map

```text
configs/
  pilot.yaml
  hypotheses.yaml

docs/
  DATA_PIPELINE.md
  SELF_AUDIT.md
  AUDIT_V3_RESOLUTION.md
  CODEX_SECOND_PASS_REVIEW_AND_EXECUTION.md
  PROJECT_STATUS_FOR_YIHENG.md

src/pr_pilot/
  data/                  download, screening, clustering, strict manifests
  runtime/               Gemmi tensor adapter and canonical interface contract
  model/                 DM-ICF implementation
  training/              corruption, losses, stages, trainer, full-1000 refit
  inference/             joint sampling and SPIR
  evaluation/            core metrics, robustness, statistics, field audits

tools/
  run_official_baselines.py
  evaluate_official_baselines.py
  preflight_official_baselines.py
  audit_config_usage.py
  run_pilot_experiments.py

RUNBOOK.md                canonical execution order
```

## CPU-only gate before GPU work

```bash
pip install -e '.[dev]'
python -m compileall -q src tests tools
pytest
python tools/audit_config_usage.py --config configs/pilot.yaml --repo-root .
pr-pilot --help
```

After `third_party/LOCK.json` is populated with reviewed immutable SHAs:

```bash
python tools/preflight_official_baselines.py --repo-root .
```

No GPU training should start until `pr-pilot audit-data` passes on frozen manifests and the official-baseline converters/preflight accept every frozen single-polymer sample.

## Scientific non-negotiables

- no final-100 feedback into cutoffs, checkpoints, SPIR, loss weights or architecture;
- no RNA base identity atoms in the structural-prior input;
- no empirical PMI initialization of primary `C`;
- `DeltaC` output starts exactly at zero;
- learned-alpha residual starts exactly at zero over a distance prior;
- Protein/RNA and interface/non-interface loss groups are normalized before combination;
- primary `C` is frozen after Stage C;
- predicted structures do not enter this experimental-only pilot;
- external structure predictors are orthogonal evaluators, not teachers;
- a configuration flag without executable semantics is not allowed to masquerade as an implemented method.
