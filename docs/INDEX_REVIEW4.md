# PR Mini-Pilot: Review-4 documentation index

**Review date:** 2026-09-05. **Source revision:** `f0561cb189d59b1e74bea923fc2bfb7093b96a81`.
**Delivery type:** offline, hash-guarded overlay, not a complete repository clone.

The repository connection returned 404 in this review session; local Git cloning
failed at DNS resolution. Three complete source files previously retrieved in the
conversation were reconstructed byte-for-byte and verified against their Git blob
SHAs. Changes and tests use those authenticated snapshots. Other previously read
files inform the review but were not all available for a full integration run.
The live `main` revision and remote write access have not been re-certified.

## Current branch integration status

The overlay is now applied to a live clone on branch
`review/review4-scientific-audit`, based on the recorded source revision. The
full local pytest suite passes, the static Review-4 integration check passes,
and the complete probability-vector migration is implemented in the evaluation
runner and calibration path. This confirms software integration only; real
structure preprocessing, GPU training, official baselines, and final100
scientific results remain unverified.

## Active documents

| Document | Question it answers |
| --- | --- |
| `REVIEW4_AUDIT_ZH.md` | What is strong, what is wrong, what changed, and what remains unverified? |
| `MANUSCRIPT_IMPLEMENTATION_MAP.md` | Where do PDF equations/claims differ from implementation? |
| `IMPLEMENTATION_CONTRACT_V2.md` | Exact tensors, visibility, score meanings, stage ownership and cache rules. |
| `EXPERIMENT_VALIDATION_V2.md` | Which experiment supports each module and what counts as a successful claim? |
| `DATA_AND_RELEASE_GATES_V2.md` | What must be demonstrated before training, final-test access and publication? |
| `RELATED_WORK_AND_FAIRNESS.md` | What can be compared fairly to the relevant primary literature? |
| `REVIEW4_CHANGELOG.md` | What changed in executable code, including deliberate compatibility breaks? |
| `../REVIEW4_RUNBOOK.md` | How to apply, test, migrate and execute the new utilities. |

## Evidence hierarchy

A design intention, an implementation, a synthetic contract test, a real-data
integration test, and an empirical scientific finding are five different things.
Do not replace any one of them by the others. Test counts in the delivery concern
only the Review-4 suite. The original full suite, real structure loader, GPU
training, official baseline wrappers, frozen manifests and final-100 results have
not been executed in this session.

The PDF is an immutable **30 August 2026 working manuscript**, not a completed
results paper. This overlay does not edit that PDF. The mapping document supplies
explicit amendments for a future manuscript revision. Proposed scientific
protocols below are not described as externally registered or empirically proven.

Earlier `IMPLEMENTATION_CONTRACT.md`, `EXPERIMENT_SPEC.md`, review reports and
handoff files are retained as history. The installer can prepend a dated routing
notice without deleting their contents. Where there is a disagreement, consult
the mapping and change log rather than silently selecting the most favorable rule.
