<!-- REVIEW4-ROUTING-2026-09-05 -->
> **Review-4 routing notice (2026-09-05).** See [docs/INDEX_REVIEW4.md](docs/INDEX_REVIEW4.md) and the Review-4 runbook for updated contracts, known incompatibilities and outstanding integration gates. The original text below is preserved as historical context; it is not a claim that the new protocol or real experiments have passed.

# PR Mini-Pilot Runbook

This is the canonical execution order. Old one-off commands are not authoritative.
Every reported run must start from a clean immutable commit and save the resolved
config, manifest hashes, upstream SHAs, seed and command lines.

## 0. Hard stop rules

Do not start long GPU training until all of the following are true:

- current HEAD compiles and tests pass;
- config-usage audit has zero unknown/dead leaves;
- immutable ProteinMPNN/NA-MPNN preflight passes;
- exactly 1,000 Protein, 1,000 RNA, 1,000 complex-development and 100 final-test samples are frozen;
- P30/R80/Rfam/exact/mother-sample leakage audit is zero;
- final-test neighbours have been purged from both structural-prior pools;
- canonical interface labels are independent of the PR graph;
- a tiny end-to-end smoke run completes.

The final 100 may never be used to change architecture, cutoff, training duration,
SPIR settings, candidate budget or checkpoint policy.

## 1. Environment and code audit

```bash
conda create -n pr-pilot python=3.11 -y
conda activate pr-pilot
pip install -e '.[dev]'

python -m compileall -q src tests tools
pytest
ruff check src tests tools --select E9,F63,F7,F82
python tools/audit_config_usage.py --config configs/pilot.yaml --out artifacts/preflight/config_usage.json
python -m pr_pilot.cli test-registry
```

GitHub Actions uses Node-24-compatible `actions/checkout@v6` and `actions/setup-python@v6`.

## 2. Lock and preflight official baselines

`third_party/LOCK.json` is authoritative. Do not run a moving branch.

```bash
python tools/preflight_official_baselines.py \
  --clone \
  --out artifacts/preflight/official_baselines.json
```

Current frozen upstream SHAs are:

- ProteinMPNN: `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`
- NA-MPNN: `9fabc2482092b725e067969fba21297a806b6fda`

The preflight checks the real CLI contracts and the NA-MPNN A/U/G/C probability-column mapping before GPU time is spent.

## 3. Discover and download oversized experimental candidate pools

Use the data pipeline utilities, not hand-curated final IDs.

```bash
python -m pr_pilot.cli discover-rcsb --kind protein --out data/candidates/protein.tsv
python -m pr_pilot.cli discover-rcsb --kind rna --out data/candidates/rna.tsv
python -m pr_pilot.cli discover-rcsb --kind complex --out data/candidates/complex.tsv

python -m pr_pilot.cli download-rcsb --kind protein --candidates data/candidates/protein.tsv --out data/raw/protein --max-candidates 5000
python -m pr_pilot.cli download-rcsb --kind rna --candidates data/candidates/rna.tsv --out data/raw/rna --max-candidates 3000
python -m pr_pilot.cli download-rcsb --kind complex --candidates data/candidates/complex.tsv --out data/raw/complex --max-candidates 6000
python -m pr_pilot.cli download-rfam --out data/rfam
```

Keep download failures and checksums. Do not silently substitute a different structure for one method only.

## 4. Coordinate-level screening

The formal pilot uses the strict resolution/method rule described below. The
temporary 2026-09-05 execution exception is documented separately in
`docs/ROUND_20260905_SCREENING_EXCEPTION.md` and must not be treated as the
`pilot_v1` protocol.

```bash
python -m pr_pilot.cli screen --kind protein --config configs/pilot.yaml --download-manifest data/raw/protein/download_manifest.tsv --out data/screened/protein
python -m pr_pilot.cli screen --kind rna --config configs/pilot.yaml --download-manifest data/raw/rna/download_manifest.tsv --out data/screened/rna
python -m pr_pilot.cli screen --kind complex --config configs/pilot.yaml --download-manifest data/raw/complex/download_manifest.tsv --out data/screened/complex
```

Primary complex screening uses experimental Protein–RNA biological assemblies,
excludes DNA and large ribosome/spliceosome-like systems, enforces length/quality
limits and requires a real heavy-atom Protein–RNA contact.

If standalone RNA is insufficient, use the repository's experimental-complex RNA
chain-view augmentation. Protein partner atoms remain invisible to the RNA prior.
Final-test R80/Rfam neighbours are still purged before the 1,000 RNA pool is frozen.

## 5. Joint sequence clustering and Rfam annotation

Run Protein P30 and RNA R80/Rfam annotations across candidate pools before freezing.

```bash
python -m pr_pilot.cli annotate \
  --proteins data/screened/protein/protein_eligible.tsv \
  --rnas data/screened/rna/rna_eligible.tsv \
  --complexes data/screened/complex/complex_eligible.tsv \
  --rfam-cm-gz data/rfam/Rfam.cm.gz \
  --rfam-clanin data/rfam/Rfam.clanin \
  --cpu 8 \
  --out data/annotated
```

## 6. Freeze final test first, then priors

```bash
python -m pr_pilot.cli freeze \
  --config configs/pilot.yaml \
  --proteins data/annotated/proteins.tsv \
  --rnas data/annotated/rnas.tsv \
  --complexes data/annotated/complexes.tsv \
  --out manifests/pilot_v1
```

The order is non-negotiable:

```text
freeze strict 100-complex final test
-> remove all test P30/R80/Rfam/exact neighbours from prior candidates
-> freeze 1,000 Protein
-> freeze 1,000 RNA
-> freeze the 1,000 non-overlapping complex-development set
```

Expected files:

```text
protein_pool.tsv  1000
protein_train.tsv  900
protein_val.tsv    100
rna_pool.tsv      1000
rna_train.tsv      900
rna_val.tsv        100
complex_dev.tsv   1000
complex_train.tsv  900
complex_val.tsv    100
complex_test.tsv   100
```

## 7. Data audit

```bash
python -m pr_pilot.cli audit-data \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/data_audit
```

Also inspect method composition (X-ray/cryo-EM/NMR), length/interface-size distribution and nearest-train similarity. If an unacceptable near-homologue is found, version a new split **before training**; never repair the split after seeing final-test metrics.

## 8. Tiny end-to-end smoke run

Before long training, create a tiny fixture manifest (2 Protein, 2 RNA, 2 complexes) from already screened development candidates and run:

```text
Protein prior -> RNA prior -> C -> DeltaC -> Alpha -> Joint -> refit -> evaluate
```

The smoke run validates plumbing, not scientific performance. It must also verify:

- Stage C only changes C;
- Stage Delta changes q/DeltaC but not C/alpha;
- Stage Alpha changes relevance only;
- primary Joint keeps C frozen;
- scratch Joint trains all parameters from step 0;
- final refit replays the development schedule prefix;
- canonical 6 A interface masks do not change when PR graph settings change.

## 9. Development training of DM-ICF

```bash
python -m pr_pilot.cli train-all \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/training/primary_development/seed20260905
```

Repeat for the three predeclared seeds through the experiment orchestrator.

Primary stages are:

```text
P: Protein structural prior
R: RNA structural prior
C: Global C only, random init, no PMI, no hard-context mining
Delta: q/interactions + DeltaC, C frozen, distance alpha
Alpha: relevance score + tau only; q/DeltaC/C frozen
Joint: Delta/alpha/decoders + gradual low-LR encoder unfreezing; C frozen
```

Joint validation is not a simultaneous full-mask forward. It combines conditional metrics with deterministic sequential teacher-forced pseudo-NLL under mixed, Protein-first and RNA-first orders on a frozen validation subset.

## 10. Full-1,000 DM-ICF refit

After development selects epoch counts and all settings:

```bash
python tools/run_pilot_experiments.py --config configs/pilot.yaml --manifests manifests/pilot_v1 --out artifacts/pilot --execute
```

The refit restarts from random initialization and uses all 1,000 Protein, 1,000 RNA and 1,000 complex-development samples. It replays the selected prefix of the original schedule rather than compressing curriculum/unfreezing/cosine decay into fewer epochs.

## 11. Official from-scratch baselines

```bash
python tools/run_official_baselines.py \
  --repo-root . \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/baselines \
  --third-party-root third_party/checkouts
```

For each seed:

- 900/100 determines the training epoch/pass count;
- the model restarts from random initialization;
- all 1,000 frozen structures enter final refit;
- final 100 complexes are never used for selection.

Prepare the one-sided final-test views and evaluate the full-refit checkpoints only.

## 12. Freeze the final protocol before touching final 100 metrics

Write and hash:

```text
artifacts/FROZEN_FINAL_CONFIGURATION.yaml
artifacts/FROZEN_FINAL_CONFIGURATION.sha256
```

Record:

- git commit SHA;
- all manifest SHA256 values;
- ProteinMPNN/NA-MPNN SHAs;
- five confirmatory hypotheses;
- checkpoint policy;
- inference/candidate/SPIR budgets;
- external predictor budget if used.

## 13. Final-100 evaluation: two compute tiers

Use the dedicated runner:

```bash
python tools/run_final100.py \
  --config artifacts/FROZEN_FINAL_CONFIGURATION.yaml \
  --manifests manifests/pilot_v1 \
  --refit-root artifacts/pilot/training/primary_refit_full1000 \
  --out artifacts/final100 \
  --execute
```

Tier A runs all three seeds across all 100 complexes for core/confirmatory checkpoint metrics.
Tier B runs the full mechanistic/robustness/order/SPIR battery only on the predeclared analysis seed. Candidate-based ablations use 16 candidates; the primary mixed-order generation keeps 64.

Before this full run, profile approximately 10 representative complexes and record seconds/candidate, peak GPU memory and estimated GPU-hours. If the budget is impossible, freeze a revised evaluation budget **before** running final metrics.

## 14. Confirmatory statistics

The statistical unit is the **complex**, never the residue/token.

Only five hypotheses share the primary Holm family:

1. full DM-ICF vs dual structural prior interface normalized NLL;
2. full DM-ICF vs partner-blind/geometry-only controls;
3. contextual field vs C-only;
4. partner-scramble interface degradation;
5. top-alpha edge removal vs distance-matched lower-alpha edge removal.

Use 10,000 paired bootstrap resamples and report effect sizes/CIs. Other analyses are secondary/exploratory and must be labelled as such.

## 15. Interpretation rules

- `C` is a learned Stage-C compatibility anchor, not a thermodynamic binding energy;
- `DeltaC` is a context correction; audit its mean drift before interpreting raw C as the final population average;
- `alpha` is model relevance, not causal physical importance;
- partner-scramble/edge-removal are model-interventional evidence, not biological causality;
- external structure-prediction confidence is not binding evidence;
- sequence recovery alone is not design success.

## 16. GO / NO-GO checklist

GPU training is GO only if:

- [ ] compileall / pytest / Ruff correctness pass on current HEAD
- [ ] config usage audit has zero unknown/dead leaves
- [ ] official baseline preflight passes
- [ ] exact frozen counts are correct
- [ ] final-test P30/R80/Rfam/exact/mother overlap = 0
- [ ] prior pools have no final-test-linked neighbours
- [ ] canonical interface is independent of PR graph
- [ ] tiny end-to-end smoke succeeds
- [ ] primary hypotheses are frozen
- [ ] final configuration hash is recorded

Final-100 evaluation is GO only after all model development, controls and candidate budgets are frozen.
