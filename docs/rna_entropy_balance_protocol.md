# E0--E3 RNA entropy/balance protocol

This round keeps the locked reciprocal Adapter architecture fixed: G2,
R→P K=8, P→R K=12, radius `14.979730606 Å`, A0/mean aggregation,
multiplicative interaction, separate directional edge encoders, and the
modality projector. Protein is unchanged.

## Registered groups

| Group | Training data | RNA residual |
|---|---|---|
| E0 | original development sampling | current learned scalar gate |
| E1 | original development sampling | prior-entropy-scaled residual |
| E2 | fold-local balanced sampling | current learned scalar gate |
| E3 | fold-local balanced sampling | prior-entropy-scaled residual |

For E1/E3, every RNA residue uses

`logits_R = prior_log_probs_R + H(prior_R)/log(4) * Delta_R`.

The RNA scalar gate is removed in this mode. Positions with no selected
interaction edge already have `Delta_R = 0`. The Protein branch keeps its
registered scalar gate and all other parameters.

## Fold-local balancing

For E2/E3, each fold computes rank quartiles and frequencies on its training
complexes only, separately for RNA GC fraction, RNA length, and P→R mean
degree. P→R mean degree is the selected K=12 edge count divided by the total
RNA residue count. Each variable's inverse-frequency weights are normalized
to mean one, combined by geometric mean, clipped to `[0.5, 2.0]`, and sampled
with replacement for exactly the original number of training complexes per
epoch. Validation is never reweighted.

The registered training seed is `20260917`, with one seed only. Optimizer,
batch size, epoch budget, patience, and weight decay are identical across
groups. The old 86 holdout and the new blind manifest are not opened by the
training runner.

## Selection and checkpoints

Each fold selects by the equal-weight RNA difficulty score

`S_RNA = mean(R_Q1, R_Q2, R_Q3, R_Q4)`.

The hard checks are Protein interface ratio `< 1` and native partner NLL
lower than composition-preserving partner-shuffle NLL in both directions.
The complex-level Adapter-better-than-prior fraction is the secondary
tie-breaker. Each fold retains only `best.pt`, `last.pt`, `metrics.jsonl`,
and `summary.json`; epoch checkpoints are atomically replaced.

## Blind set lock

The blind candidate source is the local experimental mmCIF collection listed
in `PDB_LENGTHS_AND_PATHS.md` on H:. It is screened with Protein length
40--2000, RNA length 10--500, total-token, interface-contact, and atom
completeness rules. Resolution/method filtering remains disabled. Candidates
are audited jointly against the development and old holdout annotations for
sample ID, Protein30, RNA80, and Rfam leakage. The blind manifest is frozen
by SHA-256 before training; no blind metric is used for training, checkpoint
selection, or group selection.
