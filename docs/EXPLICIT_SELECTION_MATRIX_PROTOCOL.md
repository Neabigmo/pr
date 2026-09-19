# Explicit dynamic selection-matrix protocol

This branch freezes one development-only experiment. It does not read the old
215-complex diagnostic set or any new blind test.

## Data and controls

- 860 development complexes from the resplit manifest, grouped 3-fold CV.
- One registered seed: `20260919`.
- Frozen ProteinMPNN and NA-MPNN priors.
- G2 geometry, radius `14.979730606 Å`, R→P `K=8`, P→R `K=12`.
- Prior-normalized objective: `0.5 * (LP/LP_prior + LR/LR_prior)`.
- Active masks, target labels, coordinates, and selected directional edges are
  identical between native and shuffle evaluations.
- Twenty composition-preserving partner permutations are used for the formal
  native-vs-shuffle check. The existing cache has no residue-to-chain offset
  table, so this implementation uses global within-complex permutations and
  records that limitation explicitly; no chain composition is changed.

## Models

- **A0**: retrained locked E0 baseline: multiplicative interaction, mean
  aggregation, separate directional edge encoders, modality projectors, and
  learned scalar gates. For a fair comparison, A0 is retrained using the
  same newly re-exported structure-only hidden cache as A1/A2; its old E0
  weights and old decoder-hidden cache are not reused.
- **A1**: `M_ij = C`, a zero-initialized learnable global `20×4` matrix plus
  the two learned scalar gates.
- **A2**: `M_ij = C + ΔC_ij`, using one shared G2 edge encoder and one shared
  `192 → 128 → 80` MLP. The final output layer is zero initialized, so the
  adapter begins exactly at the frozen prior.

For RNA prediction the matrix row is selected by the native partner protein
AA, `M_ij[a_i, :]`. For Protein prediction the column is selected by the
native partner RNA base, `M_ij[:, b_j]`. The same matrix generator is shared
between directions; only `gP` and `gR` are direction-specific.

The dynamic generator receives only sequence-free prior encoder hidden states
and G2 geometry. It never receives decoder hidden states, teacher-forced
sequence hidden states, or partner tokens before the final matrix indexing.

## Selection and reporting

Checkpoint selection minimizes the equal-weight RNA difficulty score
`(RQ1 + RQ2 + RQ3 + RQ4)/4`, with hard safety gates:

1. Protein interface ratio `< 1`.
2. Native Protein interface NLL `<` shuffled Protein interface NLL.
3. Native RNA interface NLL `<` shuffled RNA interface NLL.

Per-complex win rates, all/active/interface NLL and ratios, quartile metrics,
and 10,000-replicate paired complex bootstrap intervals are retained. The
comparisons are `A1` versus `A0` and `A2` versus `A1`; lower ratios are better.

The decomposition `C + ΔC` is not treated as identifiable. A2 is interpreted
through the observed mean matrix and geometry-conditioned means, while A1's
`C` can be directly plotted as a global matrix.
