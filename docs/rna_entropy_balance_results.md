# E0--E3 results

The four registered groups were run with seed `20260917` on the same grouped
3-fold development split.  The old 86 holdout and the new blind manifest were
not read.  The table reports the mean over folds.

| group | RNA Q1--Q4 mean ratio | Protein interface ratio | RNA better-complex fraction | hard/partner gates |
|---|---:|---:|---:|---|
| E0 original + current | 0.98309 | 0.97279 | 0.67704 | pass/pass |
| E1 original + entropy | 0.99001 | 0.97708 | 0.56902 | pass/pass |
| E2 balanced + current | 0.98407 | 0.97140 | 0.67400 | pass/pass |
| E3 balanced + entropy | 0.98556 | 0.97859 | 0.59431 | pass/pass |

The pre-registered selection rule therefore selects **E0**: all candidates pass
the Protein and native-vs-composition-preserving-shuffle checks, and E0 has the
lowest development `S_RNA`.  This is a development-CV decision, not a test-set
decision.  Entropy scaling was implemented and tested, but it did not improve
the registered selection score in this single-seed round.

The selected E0 configuration was refit on all 991 development complexes for
the median fold-selected epoch count (2 epochs; fold best epochs 2, 1, and 4).
The refit wrote `final.pt`, `best.pt`, `last.pt`, and `metrics.jsonl` under the
I: drive result directory.

## Blind-set status

The new blind manifest was frozen before training.  Of 5 structurally eligible
candidate complexes from the supplied H: experimental mmCIF source, all 5 were
rejected by the required leakage audit: 3 by Protein30 overlap, 1 by protein
hash overlap, and 1 by RNA hash overlap with the development/old holdout
references.  The locked blind test therefore contains zero complexes.  No blind
metric is reported and the old 86 holdout is not substituted as a final test.

## Verification

- `pytest -q` with the repository `src` path: 176 tests passed.
- `compileall` over `src` and `tools`: passed.
- `git diff --check`: passed.
- No per-epoch model copies were retained; fold checkpoints use atomic
  `best.pt`/`last.pt` replacement and the refit additionally writes `final.pt`.
