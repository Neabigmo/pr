# PR mini-pilot — post-audit implementation and fairness report

**Purpose.** This document records the repository self-audit requested before any expensive GPU experiment starts. It distinguishes (1) scientific design, (2) executable implementation, (3) external-reference comparisons, and (4) deliberately deferred full-scale engineering features.

The pilot is not allowed to claim a method or test as completed merely because it appears in a YAML file or manuscript draft.

---

## 1. Current verification state

The current `main` source snapshot was independently downloaded after the audit edits and checked locally with:

```bash
python -m compileall -q src tests tools
python -m pytest -q
ruff check src tests tools --select E9,F63,F7,F82
```

All three returned success. New long-form CLI modules were also imported through their `--help` entrypoints.

GitHub Actions history may still show old failed/cancelled runs. Earlier red runs were dominated by Ruff/style failures, and later API-based repository writes do not reliably trigger another Actions workflow because GitHub suppresses recursive workflow events from some token/App writes. Therefore **the green/red badge is not used as evidence for the scientific pipeline; current-source local checks are recorded separately.**

CI itself has been migrated away from Node20-era actions:

- `actions/checkout@v6`;
- `actions/setup-python@v6`;
- `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true`;
- concurrency cancels superseded runs.

---

# 2. Design-to-code audit

| Scientific design | Executable location | Audit status |
|---|---|---|
| independent Protein/RNA structural priors | `model/dmicf.py` | implemented |
| sequence-neutral Protein backbone input | `runtime/gemmi_adapter.py` | implemented |
| sequence-neutral RNA sugar/phosphate input; no native base identity atoms | `data/features.py`, `runtime/gemmi_adapter.py` | implemented + tested |
| rich PR geometry, not a single distance | `runtime/gemmi_adapter.py` | implemented: 5 Protein × 12 RNA atom families + local-frame terms |
| `q_ij = G(hP_i + hR_j + f(e_ij))` | `model/dmicf.py` | implemented |
| random global `C in R^(20x4)` | `model/dmicf.py` | implemented + tested |
| no PMI/frequency initialization of primary C | model tests | implemented |
| `DeltaC_ij` full 80D | `model/dmicf.py` | implemented |
| zero-init DeltaC output | `model/dmicf.py`, tests | implemented + tested |
| no explicit DeltaC norm penalty | config/loss | implemented |
| distance prior + learned alpha residual | `model/dmicf.py` | implemented + tested |
| learnable positive tau | `model/dmicf.py` | implemented |
| separate P<-R/R<-P neighbourhood softmax | `model/dmicf.py` | implemented |
| lambda fixed at 1 before joint | `stages.py` | implemented |
| bounded learnable lambda in final joint | `model/dmicf.py`, `stages.py` | implemented |
| P prior -> R prior -> C -> DeltaC -> alpha -> joint | `stages.py`, `engine.py` | implemented |
| variable masking curriculum | `training/corruption.py` | implemented + tested |
| explicit full-mask examples | `training/corruption.py` | implemented + tested |
| random + sequence/spatial patch corruption | `training/corruption.py` | implemented |
| wrong-token corruption | `training/corruption.py` | implemented |
| 0.10 A training coordinate noise | adapter/engine/config | implemented |
| intra edge dropout | `training/engine.py` | implemented + tested |
| PR edge dropout | `training/engine.py` | implemented + tested |
| stochastic depth / DropPath | `model/dmicf.py` | implemented + tested |
| label smoothing for priors, none for C | losses/engine | implemented |
| Protein/RNA/interface loss scale control | `training/losses.py` | implemented + tested |
| task curriculum 2:2:1 -> 1:1:1 | `training/stages.py`, engine | implemented |
| gradual unfreezing / discriminative LR | `training/stages.py` | implemented |
| BF16 + warmup/cosine + gradient clip | `training/engine.py` | implemented |
| gradient conflict audit | `training/gradient_audit.py` | implemented; development-only diagnostic |
| mixed Protein/RNA autoregressive decoding | `inference/sampler.py` | implemented |
| Protein-first/RNA-first order controls | `inference/sampler.py`, `evaluation/full_suite.py` | implemented |
| SPIR one-pass refinement | `inference/sampler.py` | implemented |
| repeated-SPIR ablation | `evaluation/full_suite.py` | implemented |
| C vs empirical PMI | `evaluation/empirical_contacts.py` | implemented independently of model graph |
| partner scramble | evaluation runner | implemented |
| counterfactual partner mutations | evaluation runner | implemented |
| alpha causal edge removal | robustness/full suite | implemented |
| geometry permutation | robustness/full suite | implemented |
| coordinate-noise / edge-loss / partner-hide robustness | robustness/full suite | implemented |
| calibration | full suite | implemented |
| multi-seed paired statistics | `evaluation/compare_runs.py` | implemented |
| same-data partner-blind control | `training/control_modes.py`, `tools/run_internal_controls.py` | implemented |
| geometry-only capacity control | same | implemented |
| C target-chain-context control | `tools/run_statistical_controls.py` | implemented |
| fixed empirical-PMI potential control | `tools/run_statistical_controls.py` | implemented; separate from primary C |
| full-1000 final refit | `training/refit.py` | implemented |
| official baseline full-1000 refit | `tools/run_official_baselines.py` | implemented |
| official baseline final-100 one-sided views | `tools/prepare_baseline_holdout.py` | implemented |
| official baseline per-position final-100 export | `tools/evaluate_official_baselines.py` | implemented |

---

# 3. Important corrections found during the audit

These were real issues, not cosmetic changes.

## 3.1 Split/index bug

An early split implementation could validate one shuffled row set and select another because transient dataframe indices were reused. All split selection now uses immutable `sample_id` and whole biological components.

## 3.2 Final-test leakage through structural pretraining

The original framework checked complex train/test leakage but did not guarantee that final-test P30/Rfam neighbours were absent from the 1,000 Protein/RNA prior pools. Current order is:

```text
joint candidate annotation
-> freeze strict final 100
-> purge final-test exact/P30/R80/Rfam neighbours from prior candidates
-> sample the two 1,000 prior pools
```

## 3.3 Multi-chain leakage

A semicolon-joined label such as `P30_A;P30_B` must not be compared as one opaque string. Constituent P30/R80/Rfam labels are now parsed and any shared label links samples into the same split component.

## 3.4 Rfam-versus-R80 inconsistency

An earlier splitter used Rfam when known and R80 only as a fallback, while the final audit required both. This could produce a split that failed its own audit. Current components connect under **any shared P30, any shared R80, or any shared Rfam**.

## 3.5 RNA feature dimensionality

The actual RNA node feature tensor had dimension 20 while an old trainer contract expected 21. Feature dimensions now have a single source of truth in the runtime adapter and a regression test.

## 3.6 Cross-edge geometry was too shallow

An early implementation represented fewer RNA atom families than agreed. Rich PR geometry now exposes Protein N/CA/C/O/virtual-CB against 12 RNA sugar/phosphate atoms, plus atom-pair missing masks and relative local-frame displacement/rotation.

## 3.7 YAML-only tricks

Coordinate/edge stochastic augmentation, DropPath, BF16, mask curriculum and explicit full-mask examples were present in configuration before every one had a true execution path. They are now wired and tested.

## 3.8 C gauge

The manuscript discussion initially suggested double-centering C during training. That would remove row/column main effects that can be meaningful in the shared bidirectional compatibility matrix. Current code:

- removes only a single global scalar offset during forward;
- retains row/column main effects;
- uses double-centering **only** as an interaction-only post-hoc visualization/comparison gauge.

The manuscript must be synchronized to this correction before submission.

## 3.9 Graph-derived PMI was not independent validation

Counting AA/base pairs from the model's capped PR graph would make the "biological validation" depend on model graph construction. It is now labelled only as a graph-proxy sanity check. The real post-hoc validation parses **all experimental heavy-atom residue–nucleotide contacts directly from frozen structures**, with no neighbour cap, and stratifies base/sugar/phosphate contacts.

## 3.10 900 versus requested 1,000 training structures

The initial 900/100 split was valid for development but violated the literal primary goal that the final model be trained on all 1,000 structures. Current protocol:

```text
900 train / 100 validation
-> select epoch count and all hyperparameters
-> restart from scratch
-> full-1,000 refit for the selected fixed epoch count
-> final 100 complexes
```

This is implemented for DM-ICF, ProteinMPNN, NA-MPNN and internal fairness controls.

## 3.11 Baseline residue conversion

The baseline converters previously risked silently dropping a modified residue accepted by our own parser. Screening, DM-ICF and both official baseline converters now share one canonical residue vocabulary. If canonicalized sequence differs from the frozen manifest, conversion fails.

## 3.12 ProteinMPNN default training budget

Official ProteinMPNN has defaults designed for a much larger corpus. The wrapper now explicitly defines approximately one frozen-pool traversal per pilot epoch and saves every epoch so a 900/100 development selection can be followed by fixed-epoch full-1,000 refit.

---

# 4. Data acquisition and screening audit

Canonical workflow is `docs/DATA_PIPELINE.md`.

The code now includes:

- RCSB discovery query persistence;
- deterministic oversized download queues;
- SHA256 download manifests and failed-download logs;
- deposited coordinate views for single polymers;
- biological assembly 1 for the complex pilot;
- local heavy-atom Protein-RNA contact verification;
- explicit ribosome/spliceosome exclusion plus size limits;
- resolution and missing-backbone QC;
- canonical modification handling;
- standalone RNA plus partner-hidden RNA-chain views from experimental complexes;
- joint MMseqs P90/P40/P30 and R90/R80 clustering;
- Infernal/Rfam family annotation;
- strict test-first freezing;
- full provenance/version records.

### Deliberate pilot limitation: assembly 1

The first pilot uses RCSB biological assembly 1 rather than enumerating every annotated assembly of an entry. This avoids treating several alternative assemblies from one PDB deposition as independent random samples. If a biologically relevant PR contact is absent from assembly 1, that deposition is rejected rather than replaced by another assembly after looking at model results. The full dataset can later implement explicit assembly enumeration with deposition-level grouping.

---

# 5. Fairness audit

## 5.1 What is genuinely matched

ProteinMPNN and our Protein prior use the identical frozen Protein IDs. NA-MPNN and our RNA prior use the identical frozen RNA IDs. The final full-refit track uses all 1,000 structures for both methods after the 900/100 development stage has frozen epoch counts.

All final models use the same final 100 complex IDs. Candidate/test definitions are frozen before training.

## 5.2 What must NOT be described as apples-to-apples

ProteinMPNN does not see RNA identity. NA-MPNN fixed-backbone RNA reference does not see Protein identity. DM-ICF conditional modes do. Therefore the external baselines answer:

> how strong is a standard one-sided fixed-backbone sequence model on the same target polymer/backbone?

They do **not** by themselves prove the value of cross-partner coupling. Causal evidence for coupling comes from:

- structural prior only;
- + global C;
- + DeltaC;
- + learned alpha;
- partner scramble;
- counterfactual mutation;
- partner-blind complex control;
- geometry-only capacity control.

## 5.3 Probability semantics are retained

ProteinMPNN backbone-only unconditional probability and NA-MPNN specificity PPM are not silently labelled as the same mathematical likelihood. `tools/evaluate_official_baselines.py` stores `probability_semantics` on every row. Direct significance tests should use only metrics whose definitions are sufficiently matched; recovery and structural-reference comparisons are always safe to report separately.

## 5.4 No residue-level pseudo-replication

Final significance testing uses the **biological complex** as the statistical unit. Repeated residues and multiple training seeds are not treated as independent test samples.

---

# 6. Final 100-complex evaluation battery

The final set is immutable and pre-registered analyses may all be run on it; none can feed back into hyperparameter/model selection.

Mandatory computational analyses include:

1. Protein conditional NLL/recovery;
2. RNA conditional NLL/recovery;
3. joint mixed-order pseudo-NLL/recovery;
4. interface and non-interface decomposition;
5. calibration;
6. partner-sequence scrambling;
7. bidirectional single-token counterfactual mutations;
8. local KL response versus distance;
9. C versus independent heavy-atom PMI;
10. base/sugar/phosphate PMI stratification;
11. PMI partner-permutation null;
12. DeltaC magnitude/context distributions;
13. alpha entropy/effective-neighbour behaviour;
14. highest-alpha edge removal;
15. geometry-feature permutation;
16. coordinate noise 0/0.05/0.10/0.20 A;
17. PR edge removal 0/5/10/20%;
18. partner hiding 0/10/20/40%;
19. mixed versus Protein-first versus RNA-first decoding;
20. no-SPIR versus one-pass SPIR versus repeated SPIR;
21. candidate recovery/diversity/collapse;
22. strict-OOD train/test covariate-shift report;
23. A--F component ladder;
24. 10/25/50/100% complex-data efficiency;
25. three independent primary training seeds;
26. partner-blind control;
27. geometry-only capacity control;
28. C backbone-context control;
29. fixed empirical-PMI reference;
30. paired 10,000-resample bootstrap + Holm primary correction;
31. training-seed stability.

External fold/complex predictors can be added after candidate generation as an orthogonal structural validation layer. They are not required to establish the internal mechanistic claims of this mini-pilot and must not be used to retune the final model after final-100 evaluation begins.

---

# 7. Deliberately NOT used in the primary mini-pilot

These are not missing implementations; they are consciously excluded from this small controlled experiment.

### Predicted-structure augmentation

The pilot complex set is 100% experimental. Predicted structures are a later full-scale augmentation experiment, not mixed into C/DeltaC learning here.

### Family-aware resampling

A utility was explored during the audit, but the primary mini-pilot deliberately uses **uniform exact sample exposure**. With only 1,000 frozen examples, `n_f^-0.5` sampling with replacement would mean some epochs do not actually traverse the promised 1,000 structures. Family imbalance is instead measured and reported; family-aware sampling belongs to the large-data phase.

### Dynamic token batching

The current pilot trainer is intentionally sample-wise for correctness and auditability. Dynamic token packing changes throughput, not the scientific objective. It can be added before large-scale training after the sample-wise implementation is validated.

### Automatic PCGrad

PCGrad is not silently activated. `gradient_audit.py` measures development-set conflict first. If the pre-registered negative-gradient threshold is exceeded, a PCGrad run becomes a separately labelled optimization ablation. The final 100 never decides whether PCGrad is used.

---

# 8. Execution gate before expensive training

Do not start GPU training until all of the following are true:

```text
[ ] data download manifests frozen
[ ] screening rejection tables archived
[ ] MMseqs and Infernal versions recorded
[ ] joint clustering/Rfam annotation complete
[ ] strict final 100 frozen
[ ] prior pools purged against final test
[ ] pr-pilot audit-data passes
[ ] baseline converters accept every frozen single-molecule sample
[ ] current source compileall/pytest/Ruff correctness passes
[ ] configs/pilot.yaml copied to an immutable run directory
[ ] third_party/LOCK.json commits verified
```

Once final-100 evaluation starts, **no architecture, cutoff, SPIR parameter, noise setting, checkpoint rule or plotting-driven model selection may be changed on the basis of those 100 results.** A scientifically motivated post-hoc analysis may be added, but it must be labelled exploratory.
