# Scientific single-seed Adapter pilot

This directory contains the compact, development-only result of the isolated
scientific search completed on 2026-09-18.

Protocol:

- One registered seed: `20260917`.
- Protein length: `40–2000`; RNA length: `10–500`.
- No resolution or experimental-method filtering in this round.
- Grouped development CV with P30/R80/Rfam leakage checks.
- Search used prior-normalized loss and selected by the worse directional
  interface-NLL ratio.
- `test_read=false`: the frozen 86-complex holdout was not used for search,
  architecture selection, or checkpoint selection.

The compact selection record is in `selection_summary.json`. Large G3 caches,
fold checkpoints, and training logs remain outside Git under
`I:\PR_PILOT_SCIENTIFIC\20260917\`.

The final locked search configuration was selected using development CV only:

```text
geometry=G2
R->P K=8
P->R K=12
radius=14.979730606 Å
aggregation=A0
interaction=multiplicative
residual=scalar_gate
separate_edge_encoders=true
```

This file is a result summary, not a holdout evaluation report. Final refit
and holdout evaluation must remain a separate, later protocol step.
