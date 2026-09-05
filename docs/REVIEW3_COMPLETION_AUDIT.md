<!-- REVIEW4-ROUTING-2026-09-05 -->
> **Review-4 routing notice (2026-09-05).** See [INDEX_REVIEW4.md](INDEX_REVIEW4.md) and the Review-4 runbook for updated contracts, known incompatibilities and outstanding integration gates. The original text below is preserved as historical context; it is not a claim that the new protocol or real experiments have passed.

# Review 3 — completion audit after direct repository fixes

This document records the third deep review of the PR Mini-Pilot implementation.
It distinguishes **scientific design closure**, **executable engineering closure**
and **experiments not yet run**. It must not be read as a claim that the real pilot
has already produced results.

## Executive status

The repository now has a coherent executable design for the requested
1,000 Protein + 1,000 RNA + 1,100 experimental-complex pilot. The third review
focused on discrepancies that could change conclusions rather than cosmetic code
style.

The main corrections are:

1. immutable upstream baseline locking is executable rather than documentary;
2. ProteinMPNN and NA-MPNN wrappers match their pinned upstream CLIs;
3. NA-MPNN probability columns are explicitly reordered to project A/U/G/C;
4. structural-prior/full-complex final refit replays the development schedule prefix;
5. scratch joint controls have every random parameter trainable from step 0;
6. Global C, DeltaC and alpha have cleaner stage ownership;
7. canonical interface labels no longer depend on the DM-ICF PR graph;
8. joint checkpoint selection uses sequential teacher-forced pseudo-NLL;
9. only five hypotheses are confirmatory; the rest are secondary/exploratory;
10. final-100 evaluation is split into all-seed core and one-seed diagnostic tiers;
11. configuration leaves are audited against code/declarative invariants;
12. README and RUNBOOK describe the actual implementation rather than an earlier scaffold.

## 1. Baseline execution contract

### Problem found

The earlier ProteinMPNN runner mixed project-level arguments with the official
`training.py` argparse interface. The earlier NA-MPNN runner treated `na_run.py`
as argparse even though the pinned code reads exactly `sys.argv[1]` as a JSON
config. That would have made the baseline code fail at real execution time.

### Fix

`tools/run_seeded_upstream.py` seeds Python/NumPy/PyTorch and executes the pinned
entrypoint in-process without editing third-party source.

ProteinMPNN now receives only parameters supported by the pinned script and its
`--path_for_training_data` points at the prepared root containing `list.csv`,
cluster files and `pdb/`.

NA-MPNN now receives one positional JSON configuration; per-run `BASE_FOLDER` and
random-init `PREV_CHECKPOINT` are materialized before launch. Its reported final
checkpoint is `BASE_FOLDER/last.pt`, matching pinned source behavior.

The pinned NA script stops only when `total_step > TOTAL_STEPS`; the wrapper
therefore lowers the generated threshold by one to avoid one unintended extra
pass at exact equality.

`tools/preflight_official_baselines.py` checks these assumptions on CPU before GPU
training.

## 2. NA-MPNN A/U/G/C mapping

### Problem found

With `NA_SHARED_TOKENS=1`, RNA A/C/G/U occupy DA/DC/DG/DT token slots. Reading
those slots in dictionary order produces A/C/G/U, not this project's A/U/G/C.
This would silently swap the interpretation of U and C probabilities.

### Fix

The evaluator maps project A/U/G/C to `DA, DT, DG, DC` (or `A, U, G, C` when
separate RNA slots are present). A regression test fixes this ordering.

## 3. Canonical biological interface vs model receptive field

### Problem found

The old runtime set `PolymerGraph.interface` from PR edges remaining after the
model's 8 A cutoff and neighbour cap. Interface loss/recovery would therefore
change if a message-passing hyperparameter changed.

### Fix

Manifest-backed complex loading now computes a **canonical full-heavy-atom 6 A
interface on the original unperturbed source coordinates**. It overwrites runtime
interface masks after the PR graph is constructed. The 8 A/capped PR graph remains
only a receptive field.

This canonical mask propagates automatically to:

- PI/RI loss grouping;
- interface/non-interface evaluation;
- external baseline position mapping;
- partner-scramble interface endpoints.

Coordinate-noise augmentation does not alter the canonical labels.

## 4. Clean stage ownership

Primary stages are now interpreted as:

```text
Protein prior  -> Protein encoder/decoder/head
RNA prior      -> RNA encoder/decoder/head
Global C       -> C only
Delta          -> interaction encoder + DeltaC only
Alpha          -> relevance score/tau only
Joint          -> Delta/alpha/decoders + gradual low-LR pretrained encoders; C frozen
```

Global C hard-context mining is disabled in the primary configuration, because it
would change the estimand from a population/global anchor to a hard-example-
weighted anchor.

The primary joint optimizer excludes C. This preserves the Stage-C anchor for the
post-hoc comparison to empirical interface enrichment.

## 5. Scratch-control fairness

### Problem found

A random-init scratch joint model previously passed through gradual unfreezing,
leaving random encoder layers frozen during early training. That unfairly
handicapped scratch relative to pretrained models.

### Fix

Any JOINT training/refit invoked without an initialization checkpoint is treated
as a true scratch control: **every parameter, including random C, is trainable
from step 0**. Primary pretrained joint training still uses gradual unfreezing.

## 6. Final-refit schedule fidelity

Development chooses the stop epoch. Full-1,000 refit must then replay the same
training algorithm prefix, not squeeze a 150-epoch curriculum/cosine/unfreezing
schedule into a 20-epoch selected run.

Primary and internal-control refits now retain the original stage
`schedule_horizon_epochs` while stopping after the selected number of epochs.
Checkpoints record selected epoch count, schedule horizon and progress at stop.

## 7. Joint validation now measures joint behavior

### Problem found

A simultaneous full-mask validation forward hides both partner sequences. Because
unknown partner tokens contribute zero DM-ICF correction, that metric largely
collapses back to structural priors and is misaligned with mixed-order generation.

### Fix

Joint validation now uses deterministic **teacher-forced sequential pseudo-NLL**:

- mixed order;
- Protein-first;
- RNA-first;
- additional fixed mixed orders when requested.

At each step, already visited positions reveal their native token, the current and
future positions remain unknown, and the current native log probability is
recorded. Protein and RNA are normalized by log(20)/log(4) and equally weighted.
The expensive sequential metric is run on a deterministic frozen validation
subset, while conditional validation remains on the complete complex-val set.

## 8. Hypothesis hierarchy and multiplicity

The project previously described nearly the entire battery as primary. That would
weaken the paper and make multiplicity correction unnecessarily severe.

Exactly five hypotheses are now confirmatory:

1. full DM-ICF vs dual structural prior on canonical-interface normalized NLL;
2. full DM-ICF vs partner-blind/geometry-only controls;
3. contextual field vs C-only;
4. partner-scramble interface degradation;
5. high-alpha edge removal vs distance-matched lower-alpha removal.

Holm correction applies only to this confirmatory family. Other experiments remain
mandatory for a complete pilot but are labelled secondary or exploratory.

## 9. Final-100 compute budget

A full candidate-generation battery over 3 seeds x 100 complexes x several order/
SPIR settings can become more expensive than training because the current sampler
is autoregressive.

`tools/run_final100.py` therefore enforces:

- Tier A: all three primary seeds x all 100 complexes for core checkpoint-level
  evaluation;
- Tier B: full mechanistic/robustness/order/SPIR battery on the predeclared
  analysis seed;
- candidate-based ablations: 16 candidates/complex;
- primary mixed joint generation: 64 candidates/complex.

A 10-complex runtime profiling gate is still required before the full final run.
Any compute-budget reduction must be frozen before final metrics are inspected.

## 10. Configuration hygiene

`tools/audit_config_usage.py` enumerates every YAML leaf. A leaf must be referenced
by executable Python or explicitly classified as a declarative scientific
assertion. Unknown/dead leaves fail the audit.

Several misleading historical switches were removed from `configs/pilot.yaml`
rather than retained as no-op knobs.

## 11. Remaining scientific cautions (not code blockers)

These do not require redesign of DM-ICF, but they must remain explicit when the
real data are frozen and results are interpreted:

- NMR structures should be reported separately from X-ray/cryo-EM composition and
  ideally receive a sensitivity table;
- strict P30/R80/Rfam splitting should be followed by a read-only nearest-neighbour
  local sequence-similarity audit before training;
- Rfam resolved release/version, source URLs and SHA256 values should accompany
  the frozen data manifest;
- DeltaC may develop a non-zero population mean. Report
  `||E[DeltaC]|| / RMS(DeltaC)` before treating raw C as the complete final
  population-average interaction pattern;
- alpha is model relevance, not physical causality;
- high external structure-prediction confidence is not evidence of binding.

## 12. What “complete” means here

After current code/tests pass, **implementation closure** means the repository is
ready to begin real data acquisition/freezing and smoke execution.

It does **not** mean the empirical experiment has been completed. The following
still have to happen with real data:

```text
download/screen/annotate
-> freeze exact 1000/1000/1000 + 100 test
-> data audit
-> tiny real-data smoke
-> 3-seed development training
-> full-1000 refit
-> official-baseline training/refit
-> freeze final protocol
-> final-100 evaluation
-> statistics/figures
```

No numerical scientific result should be written into the manuscript before those
steps are actually executed.
