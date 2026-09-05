# PR Mini-Pilot

A strict, reproducible small-scale pilot for fixed-backbone protein/RNA inverse folding and joint protein–RNA co-design.

The pilot is deliberately designed to be **small in data volume but complete in scientific logic**. It rehearses the full future study using:

- **1,000 frozen protein structures**;
- **1,000 frozen RNA structures**;
- **1,100 experimental protein–RNA complexes**, split once into 1,000 development complexes and 100 untouched final test complexes.

## Primary experiments

### Baseline A — ProteinMPNN

Official upstream: `dauparas/ProteinMPNN`.

Train from random initialization on the same frozen protein pool used for our protein structural prior.

### Baseline B — MPNN-fixbb / NA-MPNN RNA

Official reproducible upstream: `baker-laboratory/NA-MPNN`.

Train from random initialization on the same frozen RNA pool used for our RNA structural prior.

### Our method — DM-ICF

Core hypothesis:

```text
sequence preference
=
intramolecular structural prior
+
local cross-molecular selection
```

The Dynamic Multiscale Interfacial Compatibility Field is:

```text
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

where:

- `C`: global learned 20×4 amino-acid/nucleotide compatibility matrix, **small-random initialized**;
- `DeltaC_ij`: context-dependent 20×4 residual generated from rich protein–RNA local geometry;
- `alpha_ij`: geometry-aware multi-neighbour relevance coefficient.

Empirical AA/base PMI is **not** used to initialize `C`; it is reserved for independent post-hoc biological validation.

## Mandatory training stages

None may be skipped:

```text
P: protein structural prior
R: RNA structural prior
C: global compatibility matrix
Delta: contextual residual
Alpha: relational relevance
Joint: low-LR final coordination
Inference: mixed random-order decoding + SPIR
```

## Final 100-complex battery

The held-out 100 are used only after configuration freeze. Mandatory tests include:

- Protein/RNA NLL and recovery;
- interface vs non-interface metrics;
- conditional Protein and RNA design;
- partner-scramble DeltaNLL;
- local counterfactual partner mutation and KL-distance response;
- learned `C` vs empirical PMI;
- `DeltaC` geometry dependence;
- `alpha` entropy/effective-neighbour analysis;
- coordinate-noise robustness;
- PR-edge removal robustness;
- partner-token hiding robustness;
- decoding-order sensitivity;
- SPIR ablation;
- calibration;
- sequence-collapse/composition audit;
- full ablation ladder;
- data-efficiency experiment at 10/25/50/100% complex data;
- paired bootstrap/statistical tests across complexes.

## Repository map

```text
configs/
  pilot.yaml                    frozen initial experiment defaults

docs/
  EXPERIMENT_SPEC.md            complete scientific protocol
  IMPLEMENTATION_CONTRACT.md    exact tensor/data/stage invariants

src/pr_pilot/
  data/
    features.py                 sequence-neutral Protein/RNA feature guards
    manifest.py                 deterministic 1000/1000/1100 freezing + leakage checks
  model/
    dmicf.py                    executable C + DeltaC + alpha implementation
  training/
    losses.py                   strict PI/PN/RI/RN normalized losses
    stages.py                   staged freezing/optimizer contracts
  baselines/
    wrappers.py                 official upstream adapters and fairness schema
  evaluation/
    battery.py                  mandatory tests, metrics and paired statistics
  cli.py                        manifest/audit/stage dispatch CLI

tests/
  test_dmicf_contracts.py       unit tests for scientific invariants

RUNBOOK.md                      end-to-end execution order
pyproject.toml                  environment and test definition
```

## Quick start

```bash
pip install -e '.[dev]'
pytest
python -m pr_pilot.cli test-registry
```

Before training, read in order:

1. `docs/EXPERIMENT_SPEC.md`
2. `docs/IMPLEMENTATION_CONTRACT.md`
3. `RUNBOOK.md`

## What is intentionally not faked

The repository does **not** invent local structure paths, pretend a dataset parser exists when it does not, or silently emulate upstream baselines. The model core, stage logic, loss logic, data-freezing contracts and evaluation statistics are executable; the remaining local adapter must connect the user's actual mmCIF/PDB storage layout to the documented tensor contract.

That adapter is the next implementation step once the real 1,000/1,000/1,100 source files are selected.

## Scientific non-negotiables

- final 100 test complexes never participate in tuning;
- ProteinMPNN/NA-MPNN and DM-ICF use identical frozen source pools for fair comparisons;
- native Protein side chains and RNA base identity atoms cannot leak into our structural-prior inputs;
- cross-polymer loss combination is normalized by `log(20)` and `log(4)`;
- `DeltaC` begins exactly at zero;
- learned alpha begins exactly from a distance prior;
- partner scrambling is evaluation-only in the primary model;
- predicted structures do not silently enter this experimental-only mini-pilot;
- external structure predictors are evaluators, not ground-truth teachers;
- a partially run experiment is not labeled a complete pilot.
