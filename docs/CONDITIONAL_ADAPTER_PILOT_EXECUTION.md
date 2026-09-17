# Conditional Adapter pilot execution record

## Scope and data contract

This pilot follows the scientific plan in the attached proposal. The frozen
split is 900 complexes for training, 100 for validation/model selection, and
86 for the final test. The test manifest is not read by the audit, cache
preparation, or screening commands. The 86-complex test set is read only after
the checkpoint and configuration have been selected on validation.

The ProteinMPNN and NA-MPNN checkpoints are frozen. Stage 1 trains only a
reciprocal conditional residual Adapter: RNA-to-protein and protein-to-RNA
directions are sampled 1:1, with the target's native token excluded from its
own Adapter input. The Adapter output projection is zero-initialized, so the
zero-Adapter state is numerically the prior state. The configured hidden size
is 128, edge size 64, token size 32, message size 128, one message layer,
dropout 0.1, 16 RBF bins, AdamW learning rate 3e-4, weight decay 1e-3, and
gradient clipping 1.0. Early stopping uses validation relative bidirectional
interface NLL with patience 18 in the current screen.

## Correctness gates

The following gates passed before cache generation was resumed:

1. Frozen ProteinMPNN wrapper reproduction: maximum absolute logit difference
   `9.5367431640625e-07` against the official path.
2. Frozen NA-MPNN wrapper reproduction: maximum absolute difference `0`.
3. Zero-initialized Adapter: maximum difference `0` in both directions.
4. Random rigid rotation/translation geometry invariance: maximum difference
   approximately `5.96e-08`.
5. Missing-anchor and vocabulary mapping tests: passed.

The automated test command is:

```powershell
$env:PYTHONPATH = 'F:\111临时\PR PILOT\pr\src'
& 'E:\anaconda3\envs\pytorch-clean\python.exe' -m pytest -q tests/test_adapter_pilot.py
```

An additional boundary smoke test for `9KFX-assembly1` passed after fixing
occurrence-aware RNA residue matching. This sample contains a duplicate
runtime residue key with different nucleotide identities; matching now uses
the residue key and the canonical nucleotide token rather than positional
offsets.

## Geometry audit

The audit uses the 900-complex training split and defines a contact by the
minimum full heavy-atom residue-pair distance being below 5 Angstrom. The
corrected audit found 87,829 true contact pairs. For those contacts, the
C-alpha to C1-prime anchor-distance quantiles are:

| quantile | radius (Angstrom) |
|---:|---:|
| 95% | 13.308568573 |
| 98% | 14.357456360 |
| 99% | 14.979730606 |

All 87,829 true contacts had complete C-alpha/C1-prime anchors. Integer-radius
neighbor-count quantiles used to construct N1--N4 are recorded in
`audit/summary.json`; the resulting caps are N1 `(13.3086, 30)`, N2
`(14.3575, 37)`, N3 `(14.3575, 51)`, and N4 `(14.9797, 51)` for
`(radius, K)`. The cache is prepared at radius 15 Angstrom with K=64, which
covers the audited candidate settings without truncating them.

## Reproducible outputs

The active output root is:

`F:\111临时\PR PILOT\pilot_conditional_adapter_20260916`

The zero-noise train/validation cache is under `cache/noise0p0/`. Each
completed complex appends one JSON record to `cache/noise0p0/progress.jsonl`.
Each screening candidate writes `metrics.jsonl`, `last.pt`, `best.pt`, and a
summary containing the validation selection metric. Intermediate checkpoint
files are not written once per epoch; only the resumable last checkpoint and
the validation-selected best checkpoint are retained per candidate.

## Selection and test policy

The planned G0/G1/G2, A0/A1/A2, N1--N4, and learning-rate candidate groups
were screened on train/validation only. The primary selection metric was

`0.5 * (protein_interface_NLL / protein_prior_interface_NLL + RNA_interface_NLL / RNA_prior_interface_NLL)`.

After selection, a test cache was generated separately from the frozen 86
complex manifest. RNA comparisons use one common prior-score mask carried
through cache creation and evaluation. Final reporting includes all, active,
and strict-interface NLL/recovery, paired deltas, native versus token-off
diagnostics, and complex-level bootstrap confidence intervals.

No test-derived value was used to choose radius, K, architecture, learning
rate, seed, or checkpoint.

## Completed screening, robustness, and replication

The zero-noise screen completed all 13 planned candidates. The best single
seed validation result was G2/A2/N3 with learning rate `1e-3`, relative
interface NLL `0.9062041690`, selected at epoch 4. The second-best was the
same geometry and neighborhood with learning rate `1e-4`, relative interface
NLL `0.9069145513`, selected at epoch 12. Both used batch size 16 and the
same hidden/message dimensions recorded above.

The two leading learning rates were each replicated with seeds 0, 1, and 2.
The `1e-3` configuration had mean validation metric `0.9108652906` (SD
`0.00403665`); `1e-4` had mean `0.9093239300` (SD `0.00210310`). For the
final comparison below, the lower-variance `1e-4` configuration and its
seed-0 validation-selected `best.pt` were retained. The competing `1e-3`
seed-0 result is also evaluated and reported, so the choice is auditable.

The geometry-noise sweep kept the selected G2/A2/N3 configuration and all
training parameters fixed. Its best validation metrics were:

| coordinate noise | best relative interface NLL | best epoch |
|---:|---:|---:|
| 0.00 Angstrom | 0.9069145513 | 12 |
| 0.05 Angstrom | 0.9065277194 | 12 |
| 0.10 Angstrom | 0.9069684888 | 12 |

The 0.05 result is a small single-seed validation fluctuation, not sufficient
to replace the replicated zero-noise selection. The 0.10 run stopped at epoch
30 by the same patience-18 rule and is recorded under
`noise_sweep/noise0p10/`.

## Mechanistic partner-use gate

The validation diagnostic was run on the top-1 seed-0 checkpoint without
reading the test split. The intended gate requires native partner information
to outperform token-off and partner-permutation controls. It did not pass:
on the strict interface, protein token-off was better than native by
`0.00700964` NLL and RNA token-off was better by `0.00075659`; the protein
target also became slightly better under RNA permutation by `0.00093708`.
The RNA target showed the expected direction under protein permutation, but
that isolated result is not enough to establish reciprocal partner use.

Therefore the planned P7 joint refinement was deliberately not run. This is
a scientific stopping decision, not an execution failure: an adapter that
does not demonstrate partner-dependent information on validation should not
be further optimized or selected using the frozen test set.

## Frozen-test result after selection

The final evaluation used 86 frozen test complexes and 2,440 common RNA
positions. The adapter checkpoint was always selected using train/validation
only. Paired complex bootstrap confidence intervals used 10,000 resamples.
The principal strict-interface results are:

| target | selected adapter | frozen prior | adapter improvement |
|---|---:|---:|---:|
| protein NLL, LR `1e-4` | 2.25601494 | 2.38225066 | -0.12623572 |
| RNA NLL, LR `1e-4` | 1.21155765 | 1.14920972 | +0.06234792 |
| protein recovery, LR `1e-4` | 0.28622781 | 0.27735783 | +0.00886997 |
| RNA recovery, LR `1e-4` | 0.46395067 | 0.50377823 | -0.03982756 |

Here NLL improvement is reported as `adapter - prior`, so negative is better;
recovery improvement is also `adapter - prior`, so positive is better. The
paired-bootstrap 95% intervals for the LR `1e-4` strict-interface deltas
were protein NLL `[-0.13843820, -0.11383306]`, protein recovery
`[+0.00041582, +0.01728199]`, RNA NLL `[+0.02310760, +0.10198498]`, and RNA
recovery `[-0.06785673, -0.01098934]`.

The LR `1e-3` checkpoint showed the same qualitative pattern: protein NLL
improved by `0.10566359` and protein recovery by `0.01970387`, while RNA NLL
worsened by `0.07082924` and RNA recovery declined by `0.04508453`. Thus the
adapter improves the protein-side likelihood but fails the pre-specified
two-direction improvement criterion and is not promoted as a validated
reciprocal conditional model.

The complete machine-readable records are under
`F:\111临时\PR PILOT\pilot_conditional_adapter_20260916\`, notably
`screening/summary.json`, `replicates/`, `diagnostics/seed0_top1/`, and
`test_eval/`.

## B0--B3 direction-balance round

This follow-up round changed only the four requested mechanisms: independent
directional edge sets, optional independent edge encoders, frozen-prior
normalized loss, and worst-direction checkpoint selection. The upstream
ProteinMPNN and NA-MPNN priors, G2 geometry, radius `14.35745636`, batch size
16, hidden/message dimensions, dropout, optimizer, learning rate `3e-4`,
gradient clipping, seed 0, and patience 18 were held fixed. The approved
directional caps were `K_R->P=8` and `K_P->R=12`; B0/B1 retained the original
shared union graph with `K=51`.

The training objective for B1--B3 is computed per batch using the same
direction-specific active mask for Adapter and detached frozen prior:

`0.5 * (L_P / L_P_prior + L_R / L_R_prior)`.

The validation checkpoint metric for every group is
`max(rP, rR)`, where each ratio is the strict-interface Adapter NLL divided by
the corresponding frozen-prior NLL. The four single-seed validation results
are:

| group | loss | graph | edge encoder | best epoch | rP | rR | max(rP,rR) |
|---|---|---|---|---:|---:|---:|---:|
| B0 | absolute 0.5/0.5 | shared K=51 | shared | 7 | 0.92032 | 0.90179 | 0.92032 |
| B1 | prior-normalized | shared K=51 | shared | 7 | 0.92103 | 0.90157 | 0.92103 |
| B2 | prior-normalized | R->P=8, P->R=12 | shared | 14 | 0.91734 | 0.91761 | **0.91761** |
| B3 | prior-normalized | R->P=8, P->R=12 | separate | 7 | 0.92228 | 0.91426 | 0.92228 |

B2 is the best single-seed validation result and is notably more balanced:
both directions are close to `0.9175`. In this round, direction-specific K
provided the clear improvement; splitting the edge encoder did not improve the
single-seed validation score. This is not a claim about seed-robustness; a
multi-seed replicate is still required before promoting any configuration as
the final reciprocal model.

## Corrected contact-partner and data-balance report

The corrected audit defines a contact partner using the minimum full
heavy-atom residue-pair distance `<5 A`, rather than an anchor-radius neighbor.
Across the 900-complex train split, there were 87,829 true contact pairs, all
with complete C-alpha/C1-prime anchors. Among active target nodes only, the
contact partner quantiles were Protein q95/q99 `4/5` and RNA q95/q99 `11/14`.
The previously observed RNA `30--51` values are anchor-neighborhood counts,
not heavy-atom contact-partner counts.

The per-complex report records Protein/RNA lengths, contact-active nodes,
canonical interface nodes, heavy-atom contact partner counts for every target,
shared-K edge counts, R->P edge counts, P->R edge counts, and target-node
counts. Its aggregate `P->R edges / R->P target nodes` ratio is `1.8837` on
train and `1.4876` on validation for the approved 8/12 directional graph.

The report and B0--B3 outputs are under:

`F:\111临时\PR PILOT\pilot_conditional_adapter_20260917_b0123\audit\data_balance.jsonl`

`F:\111临时\PR PILOT\pilot_conditional_adapter_20260917_b0123\audit\data_balance_summary.json`

`F:\111临时\PR PILOT\pilot_conditional_adapter_20260917_b0123\balance\summary.json`

The frozen test split was not read during the B0--B3 training/selection round.

## B0--B3 frozen-test comparison against the priors

After all four checkpoints had been selected from train/validation only, each
was evaluated once on the same frozen 86-complex test cache. The test cache
contains 2,440 RNA positions. The evaluation uses the same cached ProteinMPNN
and NA-MPNN prior outputs for every group and reports the canonical interface
subset as the primary comparison. No test metric was used to change a
checkpoint, hyperparameter, or architecture.

For the table below, `rP` and `rR` are adapter interface NLL divided by the
corresponding frozen-prior interface NLL. `Delta NLL` is `prior - adapter`, so
positive values favor the adapter. `Delta recovery` is `adapter - prior`, so
positive values favor the adapter.

| group | rP | rR | max(rP,rR) | Protein Delta NLL | RNA Delta NLL | Protein Delta recovery | RNA Delta recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 absolute/shared K=51/shared edge | 0.95206 | 1.04043 | **1.04043** | +0.11421 | -0.04646 | +0.01354 | -0.03132 |
| B1 prior-normalized/shared K=51/shared edge | 0.95220 | 1.03842 | **1.03842** | +0.11388 | -0.04415 | +0.01432 | -0.02741 |
| B2 prior-normalized/K=8/12/shared edge | 0.95785 | 1.18382 | 1.18382 | +0.10041 | -0.21124 | +0.00886 | -0.11002 |
| B3 prior-normalized/K=8/12/separate edge | 0.95284 | 1.06373 | 1.06373 | +0.11234 | -0.07324 | +0.00941 | -0.04368 |

The frozen-test conclusion is therefore different from the single-seed dev
ranking. B2 was the best dev checkpoint by the predeclared worst-direction
criterion, but its RNA-side performance deteriorated substantially on the
frozen test. B1 is the closest of the four to preserving both priors by the
test worst-direction ratio, with B0 nearly tied; neither is a two-direction
improvement because RNA remains worse than its prior. B3's separate edge
encoders reduce the B2 RNA failure but do not eliminate it. All four improve
Protein interface NLL relative to ProteinMPNN, while all four reduce RNA
interface recovery and increase RNA interface NLL relative to NA-MPNN.

The paired 10,000-resample complex-bootstrap intervals for interface NLL
(adapter minus prior; negative is better) were:

| group | Protein 95% CI | RNA 95% CI |
|---|---:|---:|
| B0 | [-0.12493, -0.10351] | [+0.01335, +0.08115] |
| B1 | [-0.12461, -0.10310] | [+0.01132, +0.07803] |
| B2 | [-0.11457, -0.08591] | [+0.15379, +0.27229] |
| B3 | [-0.12701, -0.09723] | [+0.04085, +0.10637] |

The complete per-group machine-readable records are under:

`F:\111临时\PR PILOT\pilot_conditional_adapter_20260917_b0123\test_eval\B0_absolute_sharedK51_sharedEdge\summary.json`

`F:\111临时\PR PILOT\pilot_conditional_adapter_20260917_b0123\test_eval\B1_priorNormalized_sharedK51_sharedEdge\summary.json`

`F:\111临时\PR PILOT\pilot_conditional_adapter_20260917_b0123\test_eval\B2_priorNormalized_K8_12_sharedEdge\summary.json`

`F:\111临时\PR PILOT\pilot_conditional_adapter_20260917_b0123\test_eval\B3_priorNormalized_K8_12_separateEdge\summary.json`

A compact copy of these summaries, together with the training and balance
summaries, is versioned in the repository at
`results/conditional_adapter_20260917_b0123/`.
