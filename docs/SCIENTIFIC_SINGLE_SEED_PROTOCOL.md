# Scientific single-seed Adapter pilot

This document defines the isolated scientific-system branch created after the
Adapter V2 C0--C4 round. The historical C0--C4 outputs remain valid audit
records, but they are not mixed with this protocol because they used a
different manifest and selection contract.

## Frozen protocol

- One registered seed: `20260917` for every experiment, fold, control and
  refit.
- Protein length: `40--2000`; RNA length: `10--500`.
- Resolution and experimental-method fields are retained as covariates and
  are not used as this round's exclusion filters.
- Development data are grouped by bilateral Protein P30, RNA R80 and Rfam
  connected components. The current length-compliant development manifest has
  891 train and 100 validation complexes.
- The 86-complex legacy holdout is immutable and is not read by preflight,
  cache preparation, architecture search, or checkpoint selection.

The exact exclusions and counts are stored outside Git in
`I:\PR_PILOT_SCIENTIFIC\20260917\manifest_length_v1\length_audit.json`.

## Selection

Search is sequential: geometry, direction-specific K, radius (r95/r98/r99),
aggregation, interaction, then residual. Each candidate uses one seed and the same grouped
fold assignment. Search first requires improvement relative to both frozen
priors, selecting the smallest worst-direction ratio. Partner identity is a
late promotion gate: shortlisted candidates receive 20 composition-preserving
partner shuffles, and both native directions must beat their shuffle control
before a partner-aware claim is made.

No unregistered specificity weight is used in the selection score.

## Checkpoints and recovery

Each run directory contains at most `best.pt`, `last.pt`, `metrics.jsonl` and
`summary.json`. A final refit additionally writes `final.pt`. `last.pt` is
overwritten atomically each epoch and contains optimizer, scheduler and RNG
state; `best.pt` contains only the validation-selected model state and its
selection metadata. Individual epoch weight files are not written, and no
weights or caches are committed to Git.

## Scientific interpretation

The Adapter track measures a frozen-prior residual correction. The full DM-ICF
scratch controls are reported as separate model families; their probability
semantics are not silently merged with Adapter ratios. A failed shuffle gate
does not erase an NLL improvement, but it prevents the stronger claim that the
model has learned the correct partner identity.
