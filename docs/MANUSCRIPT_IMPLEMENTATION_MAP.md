# Manuscript-to-implementation map

## Scope and classification

Basis: the supplied 25-page working manuscript, dated 30 August 2026; source
revision `f0561cb189d59b1e74bea923fc2bfb7093b96a81`; and the reviewed source excerpts.
Page references below are printed PDF page numbers, not a PDF viewer's zero-based
indices. `VERIFIED` means directly supported by complete source or an observed
configuration, not an empirical result. `PENDING` means a full execution path must
still be inspected or run. No missing implementation is inferred solely from an
absence in the three-source snapshot.

## Reconciliation table

| PDF location | Manuscript specification | Observed implementation | Assessment and explicit disposition |
| --- | --- | --- | --- |
| pp. 3, 12-14; Eq. 2, 9, 16-19 | Structural prior plus explicit multi-neighbor cross-polymer correction | `dmicf.py` computes geometry-only hP/hR, q, C, DeltaC, directed alpha, and additive logits | VERIFIED central design is present. These equations alone do not establish empirical value. |
| pp. 10-11; Eq. 6-7 | Structural logits written directly as linear heads of structural h | Source uses a same-polymer known-token decoder between h and the head | Clarification required: introduce h_geom and h_context; only h_geom enters q. Mask corruption otherwise has an underspecified entry point in the PDF. |
| p. 13; Eq. 11-12 | Double-center C and each DeltaC during optimization/forward evaluation | `global_center` removes only one overall scalar; `double_center` is post-hoc | Deliberate, scientifically material revision. Row/column effects cannot both be removed as a harmless gauge with the two priors frozen. Retain current forward centering and amend Methods. |
| p. 14; Eq. 13-15 | Shared edge score, two neighborhood softmax normalizations | Implemented | VERIFIED. Directed coefficients usually differ; do not claim a single undirected Gibbs energy is implied. |
| pp. 14-16 | Positive learned gains; Stage C may train gains | Gains are fixed to one before Joint and bounded by 2*sigmoid during Joint | Explicit scope clarification. This reduces early C-scale confounding; state bounds and stage ownership. |
| p. 16, Table 1 | Relevance stage also trains G_PR and DeltaC | `training/stages.py` freezes both and trains only relevance; old low-level setter did not | High-level protocol is a deliberate isolation revision; low-level mismatch is repaired in this overlay. Full trainer integration remains pending. |
| pp. 16-17 | Joint can release C at a much smaller learning rate | Primary coordinator freezes C | Retain as stable Stage-C anchor; report residual mean drift. An unfrozen-C run is a separately labelled ablation. Low-level setter now agrees. |
| p. 16 | Up to about 20% hard-context batches in later C stage | Primary YAML sets this fraction to zero | Reasonable pilot restriction, not a missing innovation. The learned anchor depends on the actual family-aware sampling distribution. |
| pp. 9, 18-19 | Quality tiers; predicted structures may be introduced later at lower weights | Pilot YAML excludes predicted structures; has 4 A cutoff and an NMR exception | Narrower experimental-only pilot. Do not claim predicted-data augmentation or quality-tier results. Audit NMR separately and freeze all actual filters. |
| pp. 9-10 | Broad monomer pretraining and bilateral leakage control | Target pilot pools are 1000 P, 1000 R, 1000 development complexes plus 100 final targets | A target size, not proof that the pool exists. The stricter manifest must be frozen before claiming family generalization. |
| p. 11; graph construction | Interface is defined by the cross-edge graph | Latest repository protocol uses native full-heavy-atom 6 A labels for loss/metrics, but 8 A capped neutral graph for model input | Keep both masks; record their disagreement. Do not let geometry hyperparameters redefine evaluation strata. |
| pp. 7, 20; SPIR | Reopen uncertain interface positions in a fixed-backbone-only input setting | Old sampler consumes the graph object's interface field, which manifest loading declares canonical/native-heavy-atom | This creates an inference-input mismatch when only backbones are supplied. Review-4 uses a graph-derived design interface for SPIR; canonical labels remain scoring metadata. Explicit legacy mode is an ablation. |
| p. 20; decoding | Mixed-order sequential generation, unknown partners contribute zero | Implemented; recent source recomputes second-side SPIR uncertainty after first-side update | Preserve this behavior. New cache stores only token-independent geometry; decoder state is never frozen. |
| pp. 19-20 | Selection by normalized conditional/joint NLL, joint random order | Latest training description uses deterministic teacher-forced sequential scores | Distinguish exact fixed-order AR NLL, finite-order-mixture NLL, normalized selection score, and leave-one-out compatibility. These are not interchangeable. |
| p. 21; candidate ranking | Recompute internal NLL when both sequences are known | Same-chain decoder permits token return paths through neighboring nodes | An all-known self-score is unsafe. New leave-one-out scorer hides the scored token; it is not called a normalized joint likelihood. Actual historical ranking caller is PENDING. |
| pp. 15, 19 | Coordinate noise, edge dropout, dynamic token batching, bf16 | Configuration and source intentions exist; full runtime/trainer unavailable in this session | PENDING execution evidence. A maximum sample length is not itself dynamic batching; frozen weights do not automatically imply eval mode. |
| pp. 17-19 | Balanced per-sample PI/PN/RI/RN loss; empty groups omitted | Observed prior documentation and tests specify this | Preserve. Full loss/trainer execution was not rerun here; test absent polymers, absent interface groups, and quality weights on real tensors. |
| p. 22 | Resample highest independent biological grouping, not residues | Original helper paired by complex and silently dropped missing rows | Silent deletion repaired. New confirmatory runner aggregates seeds within samples and samples within predeclared independence groups. Group-weighted estimand is a documented protocol change. |
| pp. 4-7, 21-22 | Rich multi-module validation; external structure checks and eventual experimental evidence | Registry lists many analyses; five are confirmatory | Experiment names are not completion evidence. New matrix specifies inputs, controls, output files, failure interpretations and evidence level. |

## The centering amendment, derived explicitly

Write C[a,b] = mu + r[a] + c[b] + I[a,b], where rows and columns of I average
to zero under uniform alphabet weights. For protein prediction with b fixed,
c[b] is a common logit offset and disappears under softmax, but r[a] changes
amino-acid odds. For RNA prediction with a fixed, r[a] disappears but c[b] changes
base odds. With both tasks and frozen priors, the shared matrix can therefore
carry identifiable row and column effects; only mu is universally irrelevant.

Removing r and c is a different model, not merely a prettier heatmap. Conversely,
a gain produced only by r/c does not demonstrate partner-identity specificity.
Report the main-effect and interaction-only views separately and include a
main-effects-only control. Do not change the primary model back to double
centering merely to match an outdated Methods equation.

## The joint-probability amendment, derived explicitly

For a fixed order o, normalized categorical predictors define the valid joint
p_o(SP,SR|B,F) = product_t p(S_o[t] | known prefix, B,F). Teacher forcing the
native sequence along that same order computes its exact AR log likelihood.
This remains true even though a masked training objective is not identical to a
fixed-order maximum-likelihood training objective.

A uniform finite set of M orders defines p_mix = (1/M) sum_o p_o. Thus
log p_mix = logsumexp_o(log p_o) - log M, not mean_o(log p_o). The latter is a
useful expected-order score but a different quantity. Alphabet-normalized,
polymer-balanced scores are model-selection objectives, not normalized sequence
probabilities.

Even before considering the same-chain decoder, a generic undirected pairwise
energy with one edge coefficient would require lambdaP*alphaP_ij to equal
lambdaR*alphaR_ji on the interaction component. The source's separate destination
normalizations do not enforce this. The method can still define useful
order-conditioned generators and conditional design policies. Avoid claiming
that every conditional policy is a conditional marginal of one order-independent
joint distribution. A compatible energy model would be a distinct architecture
and is not silently substituted in this revision.

## Additional source-level questions that remain open

The old contract states `known == ~masked`, but the manuscript also calls for
wrong-token context corruption. In the reviewed decoder, tokens at `known=False`
are multiplied by zero. Replacing values only at unknown sites would have no
effect. The unseen corruption/trainer route must demonstrate that deliberately
wrong *visible context* is represented separately from prediction supervision,
without making the native target visible.

A fixed prior can still be stochastic if its DropPath layers remain in training
mode. Inspect the trainer's `train()/eval()` handling in Stage C and record whether
this is deliberate augmentation. Also verify that final refit replays both the
schedule horizon and the chosen stopping prefix after dataset size changes.
These are review questions, not asserted bugs in source that was unavailable.

The original PDF is preserved. A manuscript author should amend its Methods and
Table 1 explicitly before final evaluation rather than quietly rewriting the
historical scientific plan after looking at test outcomes.
