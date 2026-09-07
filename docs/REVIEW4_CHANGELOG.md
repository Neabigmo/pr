# Review-4 change log and compatibility notes

Review date: 2026-09-05. Known source commit:
`f0561cb189d59b1e74bea923fc2bfb7093b96a81`.
This is an offline overlay, not a successful clone or a remote commit.

## Local integration update (2026-09-05)

The overlay has since been applied to a live clone on branch
`review/review4-scientific-audit`, using the recorded source revision and the
`pytorch-clean` environment. The complete local pytest suite and the static
Review-4 integration check pass. The evaluation runner now exports complete
canonical probability vectors, and robustness calibration reports explicit
multiclass and top-label Brier scores. This update does not certify real-data,
GPU, baseline, or final100 scientific results.

## Authenticated source boundary

Three source snapshots were reconstructed from complete GitHub responses already
retrieved in the conversation. Git blob hashes exactly match the fetched values:

| File | Original Git blob SHA |
| --- | --- |
| `src/pr_pilot/model/dmicf.py` | `a305f204050bae4dc2916f569170d4452272ca12` |
| `src/pr_pilot/inference/sampler.py` | `dd5ecc4233b847dca7d0239a28a9cde0617a0f1f` |
| `src/pr_pilot/evaluation/battery.py` | `0ad871486ad837795f377fe7f3a497e9b13e7b10` |

The live tip was not re-read successfully. The original trainer, structure loader,
all baseline wrappers and evaluation producers were not available as a complete
checkout. Their full integration is explicitly outstanding. Existing source files
outside these three are NOT reconstructed speculatively or overwritten by this patch.

## Executable changes

### Model and sampler

* `model/dmicf.py`: evaluation-only, call-scoped geometric cache; stale model/geometry
  checks; sequence-dependent decoding still recomputed on every step; stronger PR
  tensor rank/dtype/device/bounds checks; low-level stage ownership aligned with the
  already-correct high-level stage controller.
* `inference/sampler.py`: reuse cached geometry across steps/candidates; fall back to
  full forward for custom wrappers/forward overrides/hooks; restore all module modes
  even on failure; explicit initial sampling vs untempered model scores and order;
  valid arguments; primary SPIR scope based on the neutral PR design graph, with a
  labeled `canonical_legacy` ablation. Second-side uncertainty remains recomputed
  after first-side refinement.
* `inference/scoring.py` (new): fixed-order autoregressive sequence scoring and
  own-token-hidden LOO compatibility scoring. These are different score types.

### Metrics and analysis

* `evaluation/battery.py`: true multiclass Brier, strict matched target sets,
  separately labeled sign-flip p values, all-zero Wilcoxon handling, validated
  multiplicity inputs, nonempty/finite calibration and probability checks.
* `evaluation/audit_metrics.py` (new): probability-vector Brier, explicit top-label
  score, finite-order log-mixture, C ANOVA, residual drift and directed-coefficient
  diagnostic. Descriptive diagnostics are not promoted to physical interpretations.
* `evaluation/paired_statistics.py` (new): strict alignment, exact/Monte Carlo
  sign-flip tests and independent-unit bootstrap intervals.
* `evaluation/confirmatory.py` (new): strict five-hypothesis export validation,
  seed-within-complex and complex-within-group aggregation, H2 intersection-union
  combination followed by Holm across five tests. This requires effect exports
  from a separately validated producer; it does not generate missing experiments.
* `tools/run_review4_statistics.py` (new): immutable-output CLI with declared data
  kind and input SHA256 values.
* `tools/check_review4_integration.py` (new): conservative static legacy-metric/full
  checkout gate. It fails on the original partial source snapshot and passes on
  the current full checkout after probability migration.

## Deliberate breaking changes that MUST be migrated

1. **Brier input:** `brier_multiclass(probabilities[N,K], targets[N])` now requires
   the full probability vector. Old three-scalar calls raise an error. The old
   `native_probability_brier` cannot reconstruct the missing squared-probability
   sum and also fails explicitly. Migrate upstream exporters/callers rather than
   catching the error and silently dropping the metric. In the current checkout,
   `evaluation/runner.py` exports the full vector and
   `evaluation/robustness.py` consumes it for calibration.
2. **P-value meaning:** the legacy `bootstrap_p_two_sided` key remains as an alias
   for a newly labeled permutation p value. Use `permutation_p_two_sided` and
   `p_method` in new reports. Old and new numbers must not be merged without
   rerunning. The sign-exchangeability assumption is explicit.
3. **SPIR inputs:** primary `design_graph` scope replaces canonical native-heavy-
   atom labels for position selection. Old behavior is opt-in `canonical_legacy`.
   This is a scientific information-access amendment, not a numerically neutral
   refactor. Freeze it before the final benchmark and report the selected scope.
4. **Stage helper:** low-level `set_trainable_stage(..., 'joint')` no longer makes
   C trainable. True scratch controls must use the existing explicit all-trainable
   helper. High-level primary stage behavior was already intended to freeze C.
5. **Missing targets:** silently intersecting series/ignoring nonfinite rows is
   no longer permitted. Predeclare intersections/eligibility; expose failure
   ledgers; do not synthesize replacement target scores.
6. **Confirmatory estimand:** the new optional analyzer uses equal independence-
   group means, not automatically uniform-complex weighting. It is not wired
   into every historical entrypoint and must be prospectively adopted as specified.

Model parameter/state-dictionary keys and shapes are unchanged in the tested
architecture. That does not guarantee every serialized object or downstream
wrapper is compatible; loaders, call sites, precision and original tests must run.
The new cache must not be persisted across changes of geometry/model/autocast mode.

## Verification actually performed

The delivery contains logs and JUnit evidence for **81 passing Review-4 tests**,
using synthetic CPU tensors and statistical examples. It also includes exact-source
counterexamples, a standalone statistics-CLI smoke and patch/installer fixture
checks. The original full repository suite, GPU training, actual structure
preprocessing, official baselines and final100 were NOT run. No empirical claim of
improved design, throughput or binding follows from these software checks.

One reproducible counterexample: with five identical nonzero paired differences,
the old bootstrap tail calculation returns zero; exact two-sided sign enumeration
returns 2/32 = 0.0625 under the stated null. This is a synthetic demonstration of
why uncertainty intervals and null p values need separate semantics, not a result
from protein-RNA data.

## Outstanding integration and scientific work

Read `DATA_AND_RELEASE_GATES_V2.md` for the complete gates. Highest priorities are
legacy probability exports, real structure/mask/augmentation tests, conditional
NA-MPNN/LigandMPNN adapters, fully matched retrained controls, freeze/holdout audit,
three-seed primary outputs and independent pair-level validation. Do not call the
project "experimentally complete" until those artifacts actually exist.
