# Retrospective E0 resplit protocol and final blind evaluation

This record belongs to the independent branch `pilot/rna-uncertainty-balanced`.
It does not modify the previous scientific branch or its results.

## Split and screening

- Source: local experimental raw CIF pool, 5,209 files.
- Locked length protocol: Protein 40–2,000; RNA 10–500.
- Resolution/method filtering was disabled for this round; those fields remain
  available for stratified reporting.
- Strict screen result: 1,075 eligible complexes.
- Fresh P30/R80/Rfam annotation was computed jointly with historical reference
  rows, then only the 1,075 eligible candidates were partitioned by fresh
  bilateral connected components.
- Development: 860 complexes. Final blind: 215 complexes (exact 80/20 split).
- This is a retrospective resplit: some structures may have participated in
  earlier development/diagnostic work. It is not historically untouched data.
- The blind manifest was frozen before the new E0 refit and was not used for
  model selection.

## Locked model and training

Frozen ProteinMPNN and NA-MPNN priors were reused without retraining. The
Adapter was retrained with one seed (`20260919`) and the fixed configuration:

`G2 + R→P K=8 + P→R K=12 + radius 14.979730606 Å + A0 mean + multiplicative interaction + separate directional edge encoders + modality projector + learned scalar gates`.

Three grouped development folds selected best epochs 1, 2, and 2; the final
refit therefore used the predeclared median of 2 epochs on all 860 development
complexes. Each fold retained `best.pt`, `last.pt`, and metrics; refit retained
`final.pt` and its summary.

## Final blind conditions

- P0: frozen priors only.
- P1: native E0 Adapter.
- P2: E0 with partner tokens off.
- P3: E0 averaged over 20 deterministic composition-preserving partner-token
  shuffles. Geometry, masks, prior logits, and selected edges remain fixed.

All six Protein/RNA × all/active/interface metrics were evaluated. The final
summary contains per-complex results and 10,000 complex-level paired bootstrap
intervals:

`I:\PR_PILOT_SCIENTIFIC\20260919\reports\resplit_e0_blind\blind_summary.json`

The final checkpoint is outside Git at:

`I:\PR_PILOT_SCIENTIFIC\20260919\reports\resplit_e0_cv\refit\E0_resplit\final.pt`

The pre-blind lock is:

`I:\PR_PILOT_SCIENTIFIC\20260919\reports\resplit_e0_cv\final_lock.json`

## Main blind result

P1 versus P0, ratio of means:

| subset | Protein | RNA |
|---|---:|---:|
| all | 0.993589 | 0.991382 |
| active | 0.986720 | 0.990150 |
| interface | **0.972615** | 0.991463 |

Protein interface bootstrap 95% CI for the ratio: `[0.970723, 0.974379]`.
RNA interface bootstrap 95% CI: `[0.982880, 1.000237]`; therefore the RNA
improvement is modest and its 95% interval does not fully clear 1.

RNA interface prior-difficulty quartiles for P1/P0 were:

`Q1=1.097443, Q2=0.986604, Q3=0.971843, Q4=0.961957`.

Thus the blind result repeats the easy-RNA degradation / hard-RNA gain pattern.
P1 versus P2 was only a very small RNA-interface difference (native lower by
about 0.000235 NLL on average). P1 versus P3 did not pass: native RNA
interface NLL was about 0.000225 higher than the shuffled average, and native
was better on only about 41.4% of the paired repeated-shuffle comparisons.

The composition audit passed all 8,600 shuffle checks, and manifest/cache IDs,
active masks, and selected edge counts were consistent.

These results support a robust Protein-side benefit, but do not support a
strong claim that this E0 implementation uses the correct RNA partner identity
on the retrospective blind set.
