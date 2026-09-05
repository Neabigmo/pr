<!-- REVIEW4-ROUTING-2026-09-05 -->
> **Review-4 routing notice (2026-09-05).** See [docs/INDEX_REVIEW4.md](docs/INDEX_REVIEW4.md) and the Review-4 runbook for updated contracts, known incompatibilities and outstanding integration gates. The original text below is preserved as historical context; it is not a claim that the new protocol or real experiments have passed.

# PR Mini-Pilot

A strict, reproducible pilot for fixed-backbone Protein/RNA inverse folding and joint Protein–RNA sequence design.

## Frozen scientific question

The pilot tests whether Protein–RNA co-design benefits from an explicit decomposition

```text
sequence preference
= intramolecular structural prior
+ cross-molecular selection
```

with the DM-ICF field

```text
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

- `C`: randomly initialized learned 20x4 global AA/base compatibility anchor;
- `DeltaC_ij`: geometry/context-dependent 20x4 residual;
- `alpha_ij`: learned multi-neighbour relevance.

Empirical AA/base PMI never initializes the primary model. It is an independent post-hoc validation target.

## Pilot scale

- 1,000 frozen Protein structures for Protein structural-prior training;
- 1,000 frozen RNA structures for RNA structural-prior training;
- 1,100 experimental Protein–RNA complexes;
  - 1,000 development complexes;
  - 100 immutable strict bilateral-OOD final-test complexes.

Development uses 900/100 train/validation only to select epoch counts and fixed settings. Final reported checkpoints restart from random initialization and refit on the full 1,000 Protein, 1,000 RNA and 1,000 complex-development pools. Final 100 never tune the model.

## Current implementation status

Implemented:

- RCSB discovery/download and coordinate screening;
- canonical residue normalization;
- MMseqs/Rfam annotation and leakage-safe freezing;
- Gemmi Protein/RNA/complex tensor adapter;
- model-independent canonical interface labels from full-heavy-atom 6 A contacts;
- rich 5 x 12 Protein–RNA atom-pair geometry;
- Protein and RNA structural priors;
- Global C, contextual DeltaC and relevance alpha;
- six-stage training, schedule-preserving full-1,000 refit and true scratch controls;
- mixed-order joint generation and SPIR;
- partner-blind/geometry-only/statistical-potential controls;
- official ProteinMPNN and NA-MPNN input adapters;
- immutable upstream lock and CPU baseline preflight;
- final-100 conditional/joint/mechanistic/robustness evaluation;
- complex-level bootstrap/statistics;
- five confirmatory hypotheses plus secondary/exploratory registry;
- Node-24-compatible CI/manual audit.

Not yet claimed as completed science:

- the real 1,000/1,000/1,100 manifests have not yet been downloaded/frozen in this repository;
- no reported GPU training result exists yet;
- final-100 metrics therefore do not exist yet.

## External baselines

Primary external references are trained from random initialization on the exact frozen single-molecule pools:

- official `dauparas/ProteinMPNN`, locked in `third_party/LOCK.json`;
- official `baker-laboratory/NA-MPNN` fixed-backbone RNA route, locked in the same file.

They are **one-sided structural references**: ProteinMPNN does not receive RNA identity, and RNA NA-MPNN does not receive Protein identity. Cross-partner mechanism claims are tested with same-data internal controls, not by pretending the external tasks have identical information.

Before any baseline GPU job run:

```bash
python tools/preflight_official_baselines.py --clone --out artifacts/preflight/baselines.json
```

## Primary DM-ICF training semantics

```text
Protein prior
-> RNA prior
-> Global C only
-> interaction/q + DeltaC
-> alpha/relevance only
-> joint coordination
```

Primary joint coordination keeps `C` frozen. Scratch joint controls have all parameters trainable from step 0. Full-1,000 refit replays the selected prefix of the development schedule instead of compressing the curriculum/unfreezing/cosine schedule.

## Interface definition

Two concepts are deliberately separate:

1. **canonical biological interface**: any Protein/RNA residue pair with full-heavy-atom distance <= 6 A; used for interface loss/metrics/baseline mapping;
2. **DM-ICF PR graph**: 8 A cutoff + neighbour cap + sequence-neutral atom geometry; used only as the model receptive field.

Changing the PR graph cutoff/cap must not change interface labels.

## Confirmatory hypotheses

Only five hypotheses are primary and share Holm correction:

1. full DM-ICF vs dual structural prior on interface normalized NLL;
2. full DM-ICF vs partner-blind/geometry-only controls;
3. contextual field vs C-only;
4. partner-scramble interface degradation;
5. high-alpha edge removal vs distance-matched lower-alpha removal.

All other robustness, calibration, PMI, DeltaC, order, SPIR and case-study analyses are secondary or exploratory.

## Recommended execution order

Read `RUNBOOK.md`, then:

```bash
pip install -e '.[dev]'
pytest
python -m compileall -q src tests tools
ruff check src tests tools --select E9,F63,F7,F82
python tools/audit_config_usage.py --config configs/pilot.yaml
python tools/preflight_official_baselines.py
```

Data download/freeze comes next. Long GPU training is allowed only after the GO/NO-GO gates in `RUNBOOK.md` pass.

## Repository map

- `configs/pilot.yaml` — primary frozen defaults;
- `src/pr_pilot/data/` — discovery/screening/clustering/freezing;
- `src/pr_pilot/runtime/` — manifest-backed Gemmi tensors and canonical interface;
- `src/pr_pilot/model/` — structural priors + DM-ICF;
- `src/pr_pilot/training/` — six-stage trainer, controls, refit, loss audits;
- `src/pr_pilot/inference/` — joint decoder + SPIR;
- `src/pr_pilot/evaluation/` — final-100 battery/statistics;
- `tools/` — data/baseline/orchestration/preflight utilities;
- `tests/` — scientific-contract regression tests;
- `third_party/LOCK.json` — immutable external baseline SHAs.

The code is a pilot framework, not a fabricated result package: absence of real frozen data or completed training is reported explicitly rather than silently replaced by toy outputs.
