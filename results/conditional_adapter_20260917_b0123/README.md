# Conditional adapter B0--B3 result snapshot

This directory contains the compact, versioned summaries for the four
single-seed conditional-adapter ablations.

- `audit/`: corrected heavy-atom contact and per-split balance summaries.
- `balance/`: train/validation checkpoint-selection summaries for B0--B3.
- `test_eval/`: frozen-test comparisons against the cached ProteinMPNN and
  NA-MPNN priors.

The frozen-test evaluation used 86 complexes and 2,440 RNA positions. The
primary interface NLL ratios (adapter/prior) were:

| group | Protein | RNA | worst direction |
|---|---:|---:|---:|
| B0 | 0.952056 | 1.040431 | 1.040431 |
| B1 | 0.952196 | 1.038421 | 1.038421 |
| B2 | 0.957851 | 1.183816 | 1.183816 |
| B3 | 0.952843 | 1.063728 | 1.063728 |

Values below 1 favor the adapter. The compact snapshot intentionally omits
large caches, per-epoch JSONL logs, per-token TSV files, and checkpoint
weights; the original complete outputs remain in the local experiment root.
