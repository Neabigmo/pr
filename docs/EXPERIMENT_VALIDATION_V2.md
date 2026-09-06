# Experiment and validation specification, Review 4

**Status:** a proposed protocol amendment dated 2026-09-05, not a completed
experiment or a claim of external preregistration. Freeze this document, the
configuration and the analysis code before inspecting final-test outcomes.
No biological numerical results are supplied by this patch.

## 1. What the study can establish

The primary question is whether an explicit field adds reproducible,
partner-dependent sequence information beyond intramolecular structural priors
on fixed protein/RNA backbones. Distinguish four increasingly strong claims:

1. the implementation obeys its visibility, geometry and optimization contracts;
2. the predictor improves held-out conditional sequence distributions;
3. generated PAIRS retain both chains' structure and compatible interface geometry;
4. designed pairs bind or perform the intended biological function experimentally.

A unit test establishes (1), not (2). Native recovery establishes neither (3) nor
(4). Structure-prediction confidence does not establish (4). Partner-scramble
sensitivity alone cannot establish a favorable sequence change or specificity.

## 2. Module-to-evidence map

| Link in the model | Required comparison | Main readout and artifact | What a positive result does NOT establish |
| --- | --- | --- | --- |
| Protein prior | scratch / protein-only / dual pretraining, matched complex data | protein NLL, interface/non-interface NLL, data-efficiency learning curves, parameter and training-token counts | generality to all protein families |
| RNA prior | scratch / RNA-only / dual pretraining | RNA NLL and structural consistency; RNA family/length/quality strata | protein binding |
| Global C | frozen priors vs +C; main-effects-only vs interaction-only vs full C | held-out interface NLL, C ANOVA components, across-seed stability | a physical potential or a uniquely identified population interaction law |
| Contextual DeltaC | retrained C-only vs C+DeltaC at fixed distance alpha | held-out gain; residual mean drift; geometry strata and contact-matched interventions | that a large residual is biologically correct |
| Learned alpha | fixed-distance alpha vs learned alpha, otherwise matched | held-out NLL; distance-matched interventions; entropy/effective neighbors | causal importance in the real molecule |
| Joint coordination | pre-joint checkpoint vs coordinated model; matched extra-training control | both conditional endpoints and fixed-order joint scores | which component caused a gain without the extra-training control |
| Joint decoder | mixed / protein-first / RNA-first / repeated alternation | order-specific log probability, between-order variation, candidate diversity, both-chain consistency | existence of a unique order-independent undirected energy model |
| SPIR | none / one pass / repeated passes under equal initial candidates | paired pre/post metrics and diversity, fixed-position invariance | independent new sequence samples after refinement |
| Ranking | random pick / model LOO ranking / diversity-aware ranking | success at fixed top-k and oracle-vs-selected gap | binding if only predictor confidence is used |
| Biological specificity | matched and mismatched pairs; targeted compensatory changes | orthogonal structure/function measurements with uncertainty | universal transfer from a few hand-selected cases |

**Implementation status:** the existing repository describes many of these
analyses. This overlay adds reliable scoring/statistical primitives and synthetic
contract tests. It does not supply retrained controls, frozen real datasets,
external predictor jobs or experimental measurements. Every analysis must have
an executable producer, validated input schema and a nonempty output artifact
before it is marked executed. A registry entry is only a design intention.

## 3. Controls that prevent an easy but misleading result

### 3.1 Same data, capacity and optimization opportunity

Use the same frozen complex IDs, masking realizations where practicable,
validation rules and candidate budget. Report total and trainable parameters,
training tokens/steps, wall time, device and memory. A late-stage full model vs an
early C-stage checkpoint is a useful stage trajectory, but is NOT a clean capacity
or training-budget ablation. Add retrained controls with equivalent optimization
opportunity. Report unavoidable differences instead of calling them matched.

A partner-blind complex-trained model controls for having seen complex examples.
A geometry-only model controls for added PR geometry/capacity. A C main-effects
control is separately needed: a base-independent amino-acid row effect can help
interface composition without encoding which base is present. Do not confuse this
with a random-scramble control or with double-centering at training time.

For main-effects ablations, specify whether both C and DeltaC are decomposed and
whether the control is retrained. In a trained-model diagnostic, report the full,
row-only, column-only and double-centered interaction contributions separately;
this is not equivalent to retraining those restricted models.

### 3.2 Partner and geometry interventions

Use composition-preserving within-chain shuffles; additionally use alternative
partners matched on family/length/composition and local contact context where
valid. Keep target and partner backbones fixed for the token intervention. Store
permutation seeds and excluded invalid cases before viewing outcomes. Report
interface minus appropriately matched non-interface responses, not only a global
NLL change. Evaluate correct native partners and deliberately wrong partners.

A local KL-distance curve is partly a consequence of this model's explicit local
edge architecture. It is a sensitivity diagnostic, not evidence by itself that the
learned preference is physically right. Add direction-of-change tests where
experimental mutation data exist and compare to geometry-only/main-effects
controls. Do not generate missing mutation labels from the tested model.

For geometry permutation, state whether topology is fixed or rebuilt. For alpha
interventions distinguish removing an edge AND renormalizing from zeroing its
contribution with normalization held fixed. Match distance, degree, interface
status and removal count. Define both directions separately. Alpha weight already
scales an edge, so a larger effect of high-alpha deletion is not surprising by
itself. The learned-vs-distance retraining control is essential.

## 4. Five confirmatory hypotheses and exact signs

Let L_I be the mean of available protein-interface NLL/log(20) and RNA-interface
NLL/log(4), using canonical source-coordinate 6-A labels. Store per-chain metrics
as well; the composite must not hide a large deterioration on one side. Freeze a
non-inferiority/safety tolerance for each side using development data and intended
use, rather than inventing one after final100 is seen.

All exported `effect` values below use **negative = improvement**:

| Registry key | Components | Per-complex, per-training-seed effect |
| --- | --- | --- |
| `full_vs_dual_prior_interface_nll` | `primary` | L_I(full) - L_I(dual prior) |
| `full_vs_partner_identity_controls` | `partner_blind`, `geometry_only` | L_I(full) - L_I(the named control), separately for EACH control |
| `contextual_vs_global_c` | `primary` | L_I(full contextual field) - L_I(retrained C-only) |
| `partner_scramble` | `primary` | mean DeltaNLL(matched non-interface) - mean DeltaNLL(interface), DeltaNLL = scrambled minus native, alphabet-normalized |
| `alpha_edge_removal` | `primary` | mean DeltaNLL(distance-matched lower-alpha removal) - mean DeltaNLL(high-alpha removal) |

The H3 definition establishes the whole contextual field; it does NOT alone
isolate DeltaC from alpha or later fine-tuning. Those require the module ladder.
H4 with no valid matched non-interface positions is undefined. Handle such
eligibility with a pre-unblinding, prespecified roster and show intended-pool
coverage; never silently fill with zero. The current numeric utility requires a
single common eligible roster across the five tests. If a scientific design needs
different rosters, amend the schema/protocol and test the change BEFORE unblinding;
do not drop rows to make the tool run.

### 4.1 Independent units, training seeds and multiplicity

The immutable roster contains `sample_id,group_id`. Construct `group_id` from the
highest relevant dependence structure available (biological pair/mother sample,
family or connected leakage group) without using results. Multiple conformers,
mutation-series members, seeds and residues are NOT independent test units.

The supplied analyzer averages training seeds within each complex, then complexes
within each independence group, then weights groups equally. Its estimand is the
**average group effect**, not uniform-complex or pooled-residue performance. This
is an explicit amendment to the older complex-level default. Report macro-complex,
per-seed and per-chain descriptives alongside it. If every complex is independent,
set `group_id=sample_id`; do not use that shortcut when dependence is known.

Use a paired sign-flip test with a prespecified direction, on group-level effects.
Its validity requires sign-exchangeable differences under the null; it is not
assumption-free, and the underlying task is observational. State that assumption
and include a robust paired sign/Wilcoxon sensitivity analysis when appropriate.
The implementation enumerates all signs for <=16 groups and otherwise uses the
Monte Carlo (b+1)/(B+1) estimate. Group bootstrap intervals summarize effects.
They are marginal 95% intervals, NOT simultaneous multiplicity-adjusted intervals.

H2 asserts improvement over BOTH controls, so its intersection-union p value is
max(p_partner_blind, p_geometry_only). Apply Holm over the resulting FIVE primary
p values. The global decision never chooses the more favorable control. A small
number of independent groups may make any rejection impossible; report that
limitation rather than changing the test after looking at results. Three training
seeds are robustness checks, not three extra biological cohorts.

### 4.2 Executable export and analysis contract

`tools/run_review4_statistics.py` consumes:

```text
roster.csv: sample_id,group_id
effects.csv: hypothesis,component,sample_id,group_id,seed,effect
```

Exactly six component rows per sample per seed are required (H2 has two).
The analyzer rejects duplicate keys, missing methods/targets/seeds, nonfinite
scores, unknown hypotheses/components and group mismatches. The producing
pipeline must also attach method/checkpoint hashes, task/score definitions,
interface version, contrast recipe and source token-table hashes. This utility
checks numeric completeness; it cannot infer whether those upstream scientific
contrasts were measured correctly. `--data-kind synthetic` permanently marks the
output as a synthetic validation, not a biological result.

The five primary contrasts need all declared seeds. Expensive SECONDARY diagnostics
may use one analysis seed chosen in advance. In particular, do not quietly move
H4/H5 into a one-seed-only diagnostic tier while claiming a three-seed primary
analysis. Freeze any affordable revised budget before final-test access.

## 5. Joint generation must be judged as PAIRS

Record at least three pair controls: generated P with native R; native P with
generated R; generated P with its jointly generated R. Also compare intended
pairs to matched cross-pairings of generated chains. A native counterpart can be
an informative control, not automatically the correct partner for a co-adapted
new sequence. Where labels exist, targeted double changes and compensatory rescue
are stronger evidence of pair compatibility than two separately good recoveries.

Report the fraction passing BOTH per-chain structure checks AND an interface
placement check, not just an arithmetic mean that lets one successful chain
compensate for the other. Separate per-chain alignments from complex-relative
placement: independently superposing two chains can conceal incorrect docking.
Use frozen atom mappings, missing-residue handling and RNA-appropriate metrics.
Do not invent a universal success cutoff after examining the benchmark.

Order-specific autoregressive scores sum the log probabilities of all generated
sites with only preceding sites visible. Average negative log probabilities over
orders is an order-sensitivity summary. It is NOT the NLL of the order mixture;
the latter requires logsumexp of complete-sequence log probabilities with the
specified order weights. A finite mixture is not automatically a marginal over
all possible orders. LOO re-scoring is a compatibility statistic, not a normalized
joint likelihood. Sampling-temperature probabilities, untempered model scores and
post-SPIR scores must use different names.

## 6. SPIR, diversity and candidate ranking

Primary deployment-compatible SPIR uses the sequence-neutral PR graph to identify
design-interface positions. Canonical full-heavy-atom labels remain for outcomes
and training groups. The `canonical_legacy` SPIR option is an explicitly labeled
native-structure-assisted ablation, not the default pure-backbone inference task.

Compare no SPIR, one cycle and repeated cycles from the SAME initial candidates,
with fixed selection fractions, directions and temperatures. Score the second
side AFTER the first side has changed. Retain fixed sites, pre/post sequences,
per-site changes and scores; a refinement is not a fresh independent replicate.
Report uniqueness, pairwise distances within each chain, compositional entropy,
RNA GC/composition, motif retention, and both-chain joint pass rates. Rank without
native recovery or final-test labels. Evaluate top-k success at equal candidate
and external-predictor budgets, plus a random-pick control and the oracle upper
bound for the fixed candidate population.

Provisional budgets inherited from the pilot are 64 candidates for primary mixed
joint generation and 16 for candidate-based ablations. Profile ten representative
DEVELOPMENT complexes before freezing the final budget. Cache-vs-full equivalence
must hold before using acceleration for any benchmark. Do not present tiny CPU
fixture timings as GPU throughput or scientific improvement.

## 7. External and experimental validation

Freeze predictor name/version/weights, templates/MSA settings, seeds, relaxation,
confidence filters, failure policy and candidate-selection budget. Where models
share training data or architecture with the designer, call the check orthogonal
but not necessarily statistically independent. Use sequence-scrambled and
mismatched-pair controls. Report unfiltered coverage and failed predictions.
High confidence and native-like structures do not measure binding free energy.

For a later functional validation study, specify target selection without seeing
favorable final outcomes, both single-chain and pair controls, an orthogonal
binding/function assay and replication/uncertainty reporting. This release does
not claim assay execution or measured affinity. Structural or functional pilot
validation in one system must not be generalized to all protein-RNA biology.

## 8. Publication gate

A complete evidence packet includes manifests and checksums, frozen protocol,
training histories, all selected/refit checkpoints, full intended and intersection
rosters, per-token probabilities, per-complex metrics, failure ledgers, effect
exports, analysis code/version, five-hypothesis report, secondary diagnostics and
an honest statement of remaining scope limits. A plot with no such lineage is not
sufficient evidence of an implemented, reproducible experiment.
