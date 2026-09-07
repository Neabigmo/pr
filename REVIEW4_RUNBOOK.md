# Review-4 application and execution runbook

This overlay was prepared without a complete live checkout. Apply it to a separate
review branch, inspect the diff and complete integration before merging. It does
not contain model weights, real manifests, experimental outputs or a full clone.
The original manuscript is not modified; explicit amendments are documented.

## 1. Apply safely

The outer delivery bundle contains `apply_review4.py`, `BASE_SOURCE.json`,
`overlay/`, `baseline/`, `review4.patch` and `evidence/`.
The preferred route checks source blobs and creates backups before writing:

```bash
git -C /path/to/pr status --short
git -C /path/to/pr switch -c review/review4-scientific-audit
python /path/to/bundle/apply_review4.py --repo /path/to/pr
python /path/to/bundle/apply_review4.py --repo /path/to/pr --apply --annotate-legacy-docs
```

Dry-run is the default. A dirty tree or incompatible source blobs blocks application.
If HEAD moved beyond the known base, manually inspect that history, then use
`--allow-compatible-head` ONLY when the three guarded source files still match the
old or already-installed new hashes. The installer still refuses conflicting new
paths. It does not guess how to merge divergent source files. It does not commit,
force-push, create a remote branch or overwrite the working PDF.

`--annotate-legacy-docs` prepends a dated routing notice to existing README/RUNBOOK
and historical specification/handoff files, preserving their full original text.
The notice directs readers to the new index and identifies earlier documents as
history rather than silently pretending their old rules still agree. Without this
flag only explicit overlay paths are installed. Backups are stored OUTSIDE the
repository in a sibling `pr.review4-backups/` directory; inspect the printed path.
Do not add backups or evidence marked synthetic to a scientific results table.

`review4.patch` is also provided for ordinary Git review. The hash-guarded installer
is safer than applying a diff blindly to an unknown revision. Neither route is a
substitute for the tests below.

## 2. Migrate metrics before final evaluation

```bash
cd /path/to/pr
python tools/check_review4_integration.py
```

The gate flags legacy/unknown Brier signatures and a missing complete checkout.
For every flagged Brier producer, export the full canonical probability vector:

```python
from pr_pilot.evaluation.audit_metrics import multiclass_brier_score
# logits: [N,20] for protein OR [N,4] for RNA; do not mix alphabets.
probabilities = logits.float().softmax(-1).detach().cpu().numpy()
y = targets.detach().cpu().numpy()
value = multiclass_brier_score(probabilities, y)
```

If only native and maximum probabilities were saved historically, true multiclass
Brier is unavailable; rerun probability export or label the limited metric
explicitly. Never manufacture the other class probabilities. RNA output columns
must use project A/U/G/C order; an upstream shared DNA/RNA alphabet needs a tested
mapping. Changing metric semantics requires rerunning affected comparisons.

Explicitly serialize SPIR scope, order, initial temperature and raw-vs-sampling
scores in your downstream candidate export. The added Candidate fields have
backward-compatible defaults, but an external hand-written serializer will not
automatically include them. Do not label initial token probabilities as post-SPIR
likelihood. Use the new scoring functions for the score actually intended.

## 3. Execute the complete test hierarchy

```bash
pip install -e '.[dev]'
pytest
python -m compileall -q src tests tools
ruff check src tests tools --select E9,F63,F7,F82
python tools/audit_config_usage.py --config configs/pilot.yaml
python tools/check_review4_integration.py
```

Use the project's supported locked environment for full integration. Review-4's
recorded environment is only a CPU test environment, not a validated deployment
lock. The dedicated tests can also be run separately:

```bash
PYTHONPATH=src pytest -q tests/test_review4_core.py tests/test_review4_statistics.py
```

Then run the tiny REAL-structure smoke, stage/refit smoke, official-baseline
preflight and all data gates in `docs/DATA_AND_RELEASE_GATES_V2.md`. Those have not
been executed by this offline review. If a gate fails, preserve its output and
repair the actual producer; do not turn a failed check into a skipped metric.

## 4. Scores and caching

`sample_joint(..., use_cache=True)` reuses only sequence-neutral geometry and
field tensors. Same-chain token context remains dynamic. Custom subclasses,
forward overrides and top-level forward hooks automatically take the uncached
reference path. Preserve `use_cache=False` for equivalence and wrapper diagnostics.
Use `teacher_forced_order_score` for an explicitly specified AR order, and
`leave_one_out_pair_score` for complete-pair compatibility with each own site
hidden. The latter is not a normalized joint distribution.

Default SPIR selects design-interface positions from the PR graph. Reproducing an
old native-label-assisted analysis requires explicitly passing
`spir_interface_scope='canonical_legacy'` and labeling it as a separate ablation.
Do not silently merge old and new candidate populations.

## 5. Five-hypothesis statistics

Prepare validated REAL per-complex contrast exports from the frozen evaluation
pipeline according to `docs/EXPERIMENT_VALIDATION_V2.md`. Then:

```bash
python tools/run_review4_statistics.py \
  --effects artifacts/final100/confirmatory_effects.csv \
  --roster artifacts/frozen/analysis_roster.csv \
  --seeds 20260905 20260906 20260907 \
  --data-kind experimental \
  --resamples 10000 \
  --out artifacts/final100/confirmatory_review4.json
```

Use `--data-kind synthetic` for software smoke fixtures. The CLI refuses an existing
output file. It checks numeric coverage but does not certify that a claimed
experimental input is real, independent, leakage-free or correctly calculated.
Verify those upstream artifacts first. Do not derive `group_id` from metric values.

## 6. Commit only after review

Review the complete diff, run the original and new tests, migrate callers, and
choose the primary scientific amendments before final-test access. Stage only the
intended overlay/document changes, then create a normal Git commit on the review
branch and push that branch for a pull request. Do not amend unrelated history or
force-push the default branch. The outer delivery's apply report lists written
paths and backup locations for inspection and rollback.
