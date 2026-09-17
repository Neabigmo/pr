# Implementation contract v2: tensors, information flow, probabilities and ownership

**Applies to:** the Review-4 overlay on the source revision identified in
`INDEX_REVIEW4.md`. A specification is not proof of full implementation; see the
status of each route in `REVIEW4_CHANGELOG.md` and the release gates.

## 1. What the model is allowed to know

The advertised primary input is a fixed protein backbone, a fixed RNA
sugar/phosphate backbone, explicit visible sequence tokens, and user-fixed sites.
Native protein side chains and RNA base-ring atoms are not allowed geometric
inputs. They can be used for independent reference labels or post-hoc chemistry
analysis, provided no values derived from them enter prediction or deployment
site selection. Missingness indicators and atom vocabularies need their own
identity-leakage audit; stripping atom names alone is insufficient.

There are two independent geometry encoders, not a shared encoder. Let
hP_geom = EP(BP), hR_geom = ER(BR). Same-polymer sequence context is introduced
by separate decoders: hP_context = DP(hP_geom, SP_visible, knownP), and similarly
for RNA. Structural-context logits are linear heads of these context states.
Only h_geom, never h_context, enters the cross-edge q network. Partner identity
selects entries of C+DeltaC; it does not determine q or alpha.

This separation yields directly testable contracts: changing an RNA visible token
may change protein interaction logits, but must not change protein structural
logits, hP_geom/hR_geom, C, DeltaC or alpha for fixed geometry. Changing a hidden
token's stored value must have no effect. These are architectural properties,
not empirical evidence that the learned compatibility is biologically correct.

## 2. Shapes and masks

| Tensor | Shape/dtype | Meaning |
| --- | --- | --- |
| protein.node_x / rna.node_x | [NP,FP] / [NR,FR], float | Allowed intrinsic geometry only. |
| polymer.edge_index | [2,E], int64 | Within-polymer, valid index space. |
| polymer.edge_x | [E,F], float | Features in declared units with explicit missingness. |
| polymer.sequence | [N], int64 | Labels, not automatically visible input. Protein 0..19; RNA A/U/G/C 0..3. |
| polymer.valid | [N], bool | Physically usable positions. Padding is never a valid context token. |
| polymer.fixed | [N], bool | User constraints; never overwritten by generation or SPIR. |
| known | [N], bool | Which current token values a forward pass may read. |
| predict_mask | [N], bool | Which positions contribute supervised loss in this pass. |
| interface_eval | [N], bool | Canonical native full-heavy-atom 6 A metric/loss stratum. |
| interface_design | [N], bool | At least one neutral PR graph edge; available at inference. |
| pr.protein_index / rna_index | [EPR], int64 | Same edge list, endpoint indices in two node spaces. |
| pr.edge_features | [EPR,FPR], float | Neutral multi-atom distances and local-frame quantities. |
| pr.effective_distance | [EPR], float | Scalar distance; [EPR,1] is rejected to prevent silent NxN broadcasting. |

`PRBatch.validate` now checks ranks, dtypes, matching edge counts, devices and
index bounds when graph sizes are supplied. `check_values=True` additionally
checks finite features and nonnegative distances and should be used in loader
preflight. Expensive value checks are not silently assumed to have run just
because a forward call accepted a shape.

`known` and `predict_mask` are conceptually distinct. A prediction target may
never expose its native token. Visible wrong-token contexts, when used, must be
specified separately. Fixed and invalid positions must not be inadvertently
supervised. No implementation may infer visibility from a token's numerical value:
zero is a legitimate amino-acid/base index, not a mask token.

The project-wide RNA order is A,U,G,C. External models require explicit,
version-locked probability-column permutations and position mappings. Reordering
FASTA symbols without reordering probability columns is insufficient.

## 3. Two interfaces, different responsibilities

Canonical evaluation/loss labels use full-heavy-atom 6 A contacts on unperturbed
reference coordinates. The model graph uses neutral atoms, the frozen radial
cutoff and neighbor cap. Keep these labels fixed when changing graph cutoff,
coordinate noise or edge-removal interventions. Report contact coverage: how
many canonical interface sites have no neutral PR neighbor, and vice versa.

SPIR cannot require native side-chain/base atoms when the declared deployment
input contains only backbones. Its new default `spir_interface_scope="design_graph"`
therefore derives site eligibility from PR endpoints and valid/fixed masks.
`"canonical_legacy"` reproduces the earlier site-selection assumption for an
explicitly labelled comparison. Candidates record the chosen scope. A site can
be a metric non-interface site yet a design-interface site; reports must not call
changes to that site a violation of the *design*-interface freeze contract.

Do not use experimental interface masks as neural features, cross-edge selectors,
confidence features or implicit user-fixed constraints. They remain acceptable
loss/metric strata when declared as reference-derived supervision.

## 4. Field computation and identifiability

C is a small-random-initialized trainable 20x4 matrix. DeltaC has shape Ex20x4 and
an exactly zero-initialized final projection. Both forward paths remove only the
overall matrix scalar. Their row/column main effects are retained. No direct
DeltaC Frobenius penalty is added. Parameter weight decay and matrix-amplitude
constraints must not be conflated.

For edge e=(i,j), q uses projected hP_geom, hR_geom and rich edge geometry. The
current implementation sums modality-specific projections before a residual MLP;
it is not a full cross-edge message-passing network. This is within the PDF's
compact-MLP option, but it should not be described as iterative global interface
reasoning. Same-polymer encoders can convey structural context over their graph
receptive fields.

s_e = -d_e/tau + score(q_e), with tau=softplus(raw_tau). Initial score residuals
are zero. AlphaP normalizes s over RNA neighbors of each protein site; alphaR
normalizes the same s over protein neighbors of each RNA site. Both are
normalized over the full surviving graph, not re-normalized over visible tokens.
A hidden partner contributes zero after multiplication by `known`. Therefore the
available alpha mass can be below one during joint decoding. Log this mass when
analyzing partial-context predictions rather than mistaking it for an alpha bug.

Directional corrections select C[:,b]+DeltaC_e[:,b] for protein and C[a,:]+
DeltaC_e[a,:] for RNA. Gains are exactly one and frozen through Stage Alpha;
Joint uses the bounded 2*sigmoid parameterization. Empty PR neighborhoods produce
exactly zero correction.

For interpretation, export C_full, its row/column effects, C_interaction_only,
DeltaC moments, and both directed alpha arrays. The Stage-C anchor is stable but
not necessarily the final population-average field because E[DeltaC] need not be
zero and the structural prior can later change. The new `residual_drift_audit`
reports this descriptively; significance requires biological-unit resampling.

## 5. Trainable ownership

| Stage | Allowed trainable families | Prohibitions / acceptance test |
| --- | --- | --- |
| P prior | P encoder, P decoder, P head | No RNA geometry/tokens or PR edges in its objective. |
| R prior | R encoder, R decoder, R head | No P context or native RNA base atoms. |
| Global C | C only | Gains fixed; Delta/learned alpha disabled; priors frozen. |
| Delta | CrossInteractionEncoder and contextual residual | C, gains, priors and relevance frozen. |
| Alpha | Relevance score and tau only | G_PR and Delta frozen; isolates weighting adaptation. |
| Primary Joint | Contextual modules, decoders/heads; staged encoder release | C remains frozen; gains can adapt. |
| Scratch Joint control | All random parameters from step zero, including C | Must explicitly use the scratch override, not pretrained unfreezing. |

The low-level `set_trainable_stage` now obeys the same C/Alpha policy as the
previously observed high-level coordinator. It is not the gradual-unfreezing
scheduler: the trainer must still call its higher-level schedule. Scratch code
must continue to call `make_joint_fully_trainable` and its optimizer must include
C. Full trainer/optimizer integration is an outstanding release gate.

At every transition record parameter counts, requires_grad groups, actual
optimizer parameter IDs, learning rates, changed tensor families after a step,
and selected checkpoint provenance. Frozen parameters absent from an optimizer
are a stronger assertion than a configuration label. Separately log module
training/eval modes because frozen encoders can still contain stochastic layers.

## 6. Loss and validation semantics

For each sample, average supervised cross-entropy separately over P-interface,
P-noninterface, R-interface and R-noninterface sites. Omit empty groups; do not
insert zeros. Divide P losses by log20 and R losses by log4 only for composite
balancing. Average available groups within a polymer, then average available
polymer task losses. Apply sample-quality weights afterward. Report raw NLL too.

Conditional validation must specify which same-chain tokens are visible. Full-mask
conditional recovery and teacher-forced same-chain recovery have different input
information and cannot be silently compared. Joint validation should use frozen
orders and a frozen validation subset, with deterministic teacher forcing.
Pre-register the selected stage metric, stopping rule, order list and subset.

The full-development refit should replay the selected development schedule prefix,
not compress a 150-epoch curriculum into a selected 20-epoch run. Record both
schedule horizon and stopping epoch and audit any change in optimizer-step count
caused by the larger data pool. Do not tune the final test at any stage.

## 7. Four scores that must remain distinct

1. **Fixed-order AR log probability:** sum of untempered model log probabilities
   along a known prefix. Implemented by `teacher_forced_order_score`.
2. **Finite-order-mixture log probability:** logsumexp of full-sequence log
   probabilities plus log order weights. Implemented by
   `order_mixture_log_probability`. Never apply mixture arithmetic to normalized
   per-polymer loss.
3. **Initial sampler log probability:** uses temperature-adjusted sampling
   probabilities. The legacy Candidate.token_logprobs retains this meaning.
   Greedy sampling has log probability zero for its chosen action.
   Candidate.token_model_logprobs separately records untempered values, and
   Candidate.decoding_order records the actual order. Both describe the initial
   trajectory, not the post-SPIR sequence.
4. **Leave-one-out pair compatibility:** every candidate token is scored with
   itself hidden and all other valid sites visible. Implemented by
   `leave_one_out_pair_score`; not a normalized joint likelihood. Appropriate for
   a declared internal ranking heuristic, not a binding-affinity claim.

Never score a token while its own native/current identity is visible to a
multi-layer same-chain decoder. Even without a direct self edge, a two-hop
message path can return that identity to its own output. This is why the new
scorer hides the current site explicitly.

## 8. Inference cache lifecycle

`prepare_inference_cache` stores geometry-only h states and the token-independent
field once per model/sample call. `decode_cached` recomputes both within-polymer
sequence decoders and partner-token lookup every step. The sampler reuses the
cache across candidates and SPIR steps with identical geometry and parameters.
All per-module train/eval modes are restored, including on exceptions.

Cache creation and use require eval mode. Ordinary in-place parameter or geometry
changes, different model identity and replaced PR tensors are rejected. Rebuild
for a new geometry, ablation, checkpoint, device, precision/autocast context or
mutated input. Mutation through `.data` and mutation of versionless inference-mode
tensors cannot be reliably detected; neither is supported. Cache is intentionally
call-scoped, not serialized or stored globally. Tests cover CPU equivalence;
CUDA/bf16/autocast parity and real-size peak memory remain required.

Caching does not make sampling parallel or eliminate same-chain decoder costs.
Report operation counts and measured target-size throughput, not an invented
GPU speedup extrapolated from tiny synthetic graphs.

## 9. Metrics and fail-closed behavior

Multiclass Brier is mean_i sum_k (p_ik - 1[k=y_i])^2, with no division by K in
this implementation. Uniform prediction scores 1-1/K. It cannot be reconstructed
from only native probability, maximum probability and correctness. The former
three-scalar `brier_multiclass` API now fails with a migration message. Pass full
probabilities and target indices. A separately named top-label score is available.

Empty calibration groups do not score as perfectly calibrated. Missing token
probabilities and incomplete paired target sets fail rather than being silently
removed by pandas mean/dropna. The legacy bootstrap p-value key now aliases a
labelled sign-flip p value; bootstrap intervals and null inference are separate.
An exact/Monte Carlo sign-flip test assumes independent, sign-exchangeable units;
that assumption must be justified by the frozen grouping, not by the function.

## 10. Boundaries of this contract

This revision does not replace the missing full-repository integration evidence.
Real mmCIF parsing, graph equivariance, corruption visibility, gradient ownership,
refit schedules, baseline execution and external structure evaluation must pass
the gates in the companion documents before this implementation is called
publication-ready.


### Wrapper/cache integration safety

The sampler automatically falls back to full forward calls for a subclass,
instance-level forward override or top-level forward/pre-forward hook. This avoids
bypassing control-model behavior merely to accelerate inference. Novel wrappers
must prove equivalence before adding their own cache path. Instrumentation that
counts top-level model calls will deliberately disable the default cache; hook
`encode_backbones` when measuring redundant geometric work on the standard model.
