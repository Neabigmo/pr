# E4--E5 scalar-gate entropy results

E4 and E5 were trained from scratch with the original complex sampling, the
same grouped 3-fold development split, seed `20260917`, optimizer, batch size,
patience, geometry, K values, and frozen priors as E0.  Protein was unchanged.
The old 86 holdout and the new blind manifest were not read.

| group | RNA residual | RNA Q1--Q4 mean ratio | Protein interface ratio | RNA better-complex fraction |
|---|---|---:|---:|---:|
| E4 | `g_R * H/log(4) * Delta_R` | 0.98832 | 0.98037 | 0.72345 |
| E5 | `g_R * min(1, H/tau) * Delta_R` | **0.98352** | **0.97280** | 0.68508 |

Both groups passed all Protein `<1` and native-vs-composition-preserving-
shuffle checks in all three folds.

## Fold-local E5 thresholds

The raw entropy threshold was computed only from RNA residues in each fold's
training set:

| fold | `tau` (raw entropy, nats) | best epoch |
|---:|---:|---:|
| 0 | 1.020866 | 2 |
| 1 | 0.894165 | 1 |
| 2 | 0.854059 | 4 |

E5 was selected within the E4/E5 extension and refit on all 991 development
complexes for the median selected epoch count (2).  This does **not** replace
the global E0 main model: E0 remains slightly better on the full E0--E5
development comparison (`S_RNA=0.98309` versus E5 `0.98352`).  E5 does,
however, keep Q1/Q2 closer to one than the scalar-gate-free E1/E3 variants.

The refit checkpoint is on I: under
`rna_entropy_gate_e45/refit/E5_original_entropy_threshold/`.
