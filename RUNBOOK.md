# PR Mini-Pilot Authoritative Runbook (Review 3)

This is the **only supported execution order** for the current pilot. Older commands in git history are obsolete.

The central safety rule is simple:

```text
build/freeze/audit data
-> train/select/refit without reading final100
-> profile evaluation on development only
-> freeze evaluation protocol
-> explicitly unlock final100
-> evaluate once
-> statistics
```

Do not shortcut this order.

---

## 0. Start from a clean commit

```bash
git status
python --version
nvidia-smi
```

Record git SHA, Python, PyTorch/CUDA, GPU and OS in the run artifact directory.

Recommended environment:

```bash
conda create -n pr-pilot python=3.11 -y
conda activate pr-pilot
python -m pip install -e '.[dev]'
```

Before downloading data or using a GPU:

```bash
python -m compileall -q src tests tools
python -m pytest -q
python tools/audit_config_usage.py --config configs/pilot.yaml
ruff check src tests tools --select E9,F63,F7,F82
```

All must return zero.

---

# Part I. Build and freeze the real data

Detailed source/QC rationale is in `docs/DATA_PIPELINE.md`.

## 1. Broad RCSB discovery

```bash
mkdir -p data/discovery data/coordinates data/screened data/annotated

pr-pilot discover-rcsb --kind protein --out data/discovery/protein.tsv
pr-pilot discover-rcsb --kind rna     --out data/discovery/rna.tsv
pr-pilot discover-rcsb --kind complex --out data/discovery/complex.tsv
```

Discovery is intentionally broad. Scientific QC happens locally and is logged row-by-row.

## 2. Oversized coordinate download

Download substantially more than the final target counts because screening, clustering and final-test purge remove many candidates.

```bash
pr-pilot download-rcsb --kind protein \
  --candidates data/discovery/protein.tsv \
  --out data/coordinates/protein --max-candidates 5000

pr-pilot download-rcsb --kind rna \
  --candidates data/discovery/rna.tsv \
  --out data/coordinates/rna --max-candidates 3000

pr-pilot download-rcsb --kind complex \
  --candidates data/discovery/complex.tsv \
  --out data/coordinates/complex --max-candidates 6000
```

The downloader is deterministic despite concurrent network completion. It writes checksums, failures and provenance.

## 3. Coordinate-level screening

```bash
pr-pilot screen --kind protein --config configs/pilot.yaml \
  --download-manifest data/coordinates/protein/download_manifest.tsv \
  --out data/screened/protein

pr-pilot screen --kind rna --config configs/pilot.yaml \
  --download-manifest data/coordinates/rna/download_manifest.tsv \
  --out data/screened/rna

pr-pilot screen --kind complex --config configs/pilot.yaml \
  --download-manifest data/coordinates/complex/download_manifest.tsv \
  --out data/screened/complex
```

**Important:** complex screening freezes the canonical biological interface from full-heavy-atom contacts at `6.0 A` and stores exact Protein/RNA residue IDs. This interface is not the 8 A DM-ICF PR message graph.

## 4. Expand the RNA structural-prior candidate pool if standalone RNA is insufficient

The structural prior is allowed to use an RNA chain extracted from a non-final experimental complex with the Protein partner hidden. It remains an RNA-only training view.

Use the repository RNA candidate builder described in `docs/DATA_PIPELINE.md`; do not lower RNA QC merely to force the count to 1000.

## 5. Download Rfam and annotate the joint candidate universe

```bash
pr-pilot download-rfam --out data/rfam

pr-pilot annotate \
  --proteins data/screened/protein/protein_eligible.tsv \
  --rnas data/screened/rna/rna_eligible.tsv \
  --complexes data/screened/complex/complex_eligible.tsv \
  --rfam-cm-gz data/rfam/Rfam.cm.gz \
  --rfam-clanin data/rfam/Rfam.clanin \
  --cpu 8 \
  --out data/annotated
```

Protein P30 and RNA R80/Rfam must be computed over a **joint universe that includes both single-molecule candidates and complex chains**. Do not cluster those sources independently.

## 6. Freeze final100 first, then purge and draw the prior pools

```bash
pr-pilot freeze \
  --config configs/pilot.yaml \
  --proteins data/annotated/protein_candidates.tsv \
  --rnas data/annotated/rna_candidates.tsv \
  --complexes data/annotated/complex_candidates.tsv \
  --out manifests/pilot_v1
```

Expected exact counts:

```text
Protein pool   1000 = 900 train + 100 val
RNA pool       1000 = 900 train + 100 val
Complex dev    1000 = 900 train + 100 val
Final complex   100 immutable strict bilateral OOD holdout
```

The final100 is selected as whole connected components under P30 + R80 + Rfam. Exact sequences/mother samples are also prohibited from crossing. Final-test neighbours are purged from Protein/RNA prior candidates before those 1000 pools are sampled.

## 7. Mandatory post-freeze audits

```bash
pr-pilot audit-data \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/data_audit

python tools/audit_frozen_complexes.py \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/data_audit/canonical_interface

python tools/post_freeze_similarity_audit.py \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/data_audit/local_similarity \
  --threads 8
```

The local-similarity audit is read-only. If it reveals a scientifically unacceptable near duplicate **before training**, create a new versioned `pilot_v2`; never modify `pilot_v1` after final metrics are seen.

Also archive:

```text
data/rfam/rfam_resource_metadata.json
all RCSB query JSONs
download manifests/failures/checksums
screen eligible/rejected tables
MMseqs/Infernal logs
manifest SHA256 metadata
```

**NO-GO if any required data audit fails.**

---

# Part II. Verify official one-sided baselines before GPU use

## 8. Clone pinned upstreams and prepare exact 900/100 views

`third_party/LOCK.json` is already frozen to immutable official commits.

```bash
python tools/run_official_baselines.py \
  --repo-root . \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/baselines \
  --prepare-only
```

For each primary seed, run CPU preflight on the prepared development directory:

```bash
python tools/preflight_official_baselines.py \
  --repo-root . \
  --third-party-root third_party/checkouts \
  --prepared artifacts/baselines/seed20260905/prepared_development
```

Repeat for 20260906 and 20260907.

Preflight checks the exact pinned SHA, ProteinMPNN directory layout/CLI and NA-MPNN positional-JSON/checkpoint contract.

## 9. Train and full-1000-refit official baselines

```bash
python tools/run_official_baselines.py \
  --repo-root . \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/baselines
```

Primary external references:

- ProteinMPNN: from random init, exact frozen Protein 900/100, 0.10 A backbone noise; selected epoch count restarted/refit on exact full1000.
- NA-MPNN/MPNN-fixbb: from random init, exact frozen RNA 900/100; selected data-pass count restarted/refit on exact full1000.

These models **do not see partner identity** and are external structural references, not the only cross-molecular causal controls.

Do not evaluate either baseline on final100 yet.

---

# Part III. Formal DM-ICF training while final100 remains locked

## 10. Train/select/refit the three primary seeds

Use the safe training-only entrypoint:

```bash
python tools/run_primary_training_only.py \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/pilot_experiments \
  --device cuda
```

This script **never opens `complex_test.tsv`**.

For each seed it runs:

```text
P -> R -> C -> Delta -> Alpha -> Joint
```

then restarts/refits the full pipeline on the full 1000 development samples using the validation-selected epoch count and the **same original development schedule prefix**.

Primary ownership:

```text
C:      C only; lambda=1; no PMI init; no hard-context mining
Delta:  q + DeltaC only; C/priors/alpha frozen
Alpha:  relevance residual + tau only; q/DeltaC/C/priors frozen
Joint:  C remains frozen; contextual heads adapt; pretrained encoders gradual-unfreeze
```

The script also runs the **development-only DeltaC mean-drift audit** for every final refit checkpoint.

Expected gate artifact:

```text
artifacts/pilot_experiments/PRIMARY_TRAINING_READY.json
```

## 11. Train internal H2 controls before final100

```bash
python tools/run_internal_controls.py \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --primary-root artifacts/pilot_experiments/training \
  --out artifacts/internal_controls \
  --device cuda
```

Control semantics:

- `partner_blind`: same dual priors, then partner-blind Joint adaptation directly. It intentionally skips C/Delta/Alpha because those terms are disabled and would have no meaningful gradient.
- `geometry_only`: preserves cross-geometry and interaction capacity but removes specific partner AA/base identity; follows C -> Delta -> Alpha -> Joint.

Both use development selection and schedule-prefix full1000 refit.

## 12. Optional secondary controls/ablations before final evaluation

Statistical controls:

```bash
python tools/run_statistical_controls.py \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --primary-root artifacts/pilot_experiments/training \
  --out artifacts/statistical_controls \
  --device cuda
```

This includes:

- Global-C backbone-only-context control;
- fixed development empirical-PMI statistical potential.

PMI is computed from **development experimental structures only**, using the same canonical 6 A heavy-atom cutoff. It never initializes primary C.

Secondary data-efficiency/geometry ablations may use `tools/run_pilot_experiments.py`, but the main primary path above is preferred. Secondary families default to the predeclared analysis seed; choose any all-seed expansion before inspecting final results.

---

# Part IV. Profile and freeze final-evaluation protocol before opening final100

## 13. Development-only runtime profiling

Use the analysis-seed final refit checkpoint recorded in `PRIMARY_TRAINING_READY.json`. Example:

```bash
python tools/profile_evaluation_budget.py \
  --config artifacts/pilot_experiments/configs/primary_seed20260905.yaml \
  --checkpoint artifacts/pilot_experiments/training/primary_refit_full1000/seed20260905/joint/refit.pt \
  --manifest manifests/pilot_v1/complex_val.tsv \
  --out artifacts/evaluation_budget_profile \
  --device cuda
```

This uses development validation complexes only. Inspect:

```text
artifacts/evaluation_budget_profile/summary.json
```

If the predeclared Tier-B budget is too large, adjust it **now**, before opening final100, and create a new frozen config/version.

## 14. Freeze the evaluation protocol lock

Only after the runtime budget is accepted:

```bash
python tools/freeze_evaluation_protocol.py \
  --config configs/pilot.yaml \
  --test-manifest manifests/pilot_v1/complex_test.tsv \
  --runtime-profile artifacts/evaluation_budget_profile/summary.json \
  --out artifacts/EVALUATION_PROTOCOL_LOCK.json
```

The lock records SHA256s of the config, final-test manifest and runtime profile, plus the Tier-A/Tier-B budget and H1-H4 definitions.

From this point onward:

```text
NO architecture changes
NO cutoff changes
NO mask/noise changes
NO checkpoint-rule changes
NO candidate-budget tuning
NO SPIR tuning
NO deletion of hard final targets
```

---

# Part V. Open final100 exactly under the frozen protocol

## 15. Primary DM-ICF final evaluation

```bash
python tools/run_primary_final_evaluation.py \
  --protocol-lock artifacts/EVALUATION_PROTOCOL_LOCK.json \
  --training-ready artifacts/pilot_experiments/PRIMARY_TRAINING_READY.json \
  --config configs/pilot.yaml \
  --test-manifest manifests/pilot_v1/complex_test.tsv \
  --dev-manifest manifests/pilot_v1/complex_dev.tsv \
  --out artifacts/pilot_experiments/evaluation \
  --device cuda
```

The script verifies frozen hashes and performs **no training/selection**.

Evaluation tiers:

### Tier A — every primary seed x all 100 final complexes

- conditional Protein/RNA per-position probabilities;
- canonical interface/non-interface metrics;
- deterministic teacher-forced joint pseudo-NLL;
- partner scrambling.

### Tier B — predeclared analysis seed only

- counterfactual mutation/KL-distance;
- C/PMI and DeltaC/alpha analyses;
- geometry/edge/partner-hide robustness;
- calibration;
- order sensitivity;
- SPIR/no-SPIR/repeated-SPIR;
- candidate diversity/collapse;
- dataset-shift diagnostics.

## 16. External baseline final100 evaluation

After the same protocol lock exists, prepare the exact one-sided final views and run:

```bash
python tools/prepare_baseline_holdout.py \
  --manifest manifests/pilot_v1/complex_test.tsv \
  --out artifacts/baseline_holdout

python tools/evaluate_official_baselines.py \
  --repo-root . \
  --baseline-summary artifacts/baselines/baseline_run_summary.json \
  --prepared-holdout artifacts/baseline_holdout \
  --out artifacts/baseline_final100
```

Do not merge ProteinMPNN/NA-MPNN probability semantics with DM-ICF silently; the exporter records the semantics explicitly.

## 17. Internal control final100 evaluation

`run_internal_controls.py` writes final control evaluation tables after their full1000 refits. Run it only under the frozen final-evaluation protocol; do not use control final metrics to retune the controls.

---

# Part VI. Statistics

## 18. Confirmatory H1-H4 only

Prepare run manifests with these exact model names:

```text
F_joint_full1000
B_dual_prior_full1000
C_global_C_full1000
E_alpha_full1000
partner_blind
geometry_only
```

Then:

```bash
python tools/run_confirmatory_statistics.py \
  --component-runs artifacts/pilot_experiments/statistics/component_ladder_runs.tsv \
  --control-runs artifacts/internal_controls/control_runs.tsv \
  --out artifacts/statistics/confirmatory
```

Primary statistical unit = **biological complex**.

Seed aggregation occurs per complex before inference. Holm correction is applied only across H1-H4.

## 19. Broad descriptive/secondary tables

```bash
python -m pr_pilot.evaluation.compare_runs \
  --runs <run_manifest.tsv> \
  --reference F_joint_full1000 \
  --out artifacts/statistics/exploratory
```

These broad NLL/recovery comparisons are **not primary**. They receive exploratory BH adjustment/effect sizes/CI.

---

# Part VII. External structure checks

Independent structure predictors may be used only after sequence-design evaluation is frozen.

Use the same predictor/version, MSA/template policy, seeds and candidate budget for all compared methods.

Report Protein, RNA and interface consistency separately. Predictor confidence is not binding energy.

---

# Final completion criteria

The **code framework** is ready only if CI/tests/audits/preflight pass.

The **pilot experiment** is complete only when all boxes below are satisfied:

- [ ] exact 1000 Protein pool frozen;
- [ ] exact 1000 RNA pool frozen;
- [ ] exact 1000 complex development pool + immutable 100 final frozen;
- [ ] P30/R80/Rfam/exact/mother-sample leakage audit = 0;
- [ ] canonical 6 A interface schema audit passes;
- [ ] post-freeze local similarity audit archived;
- [ ] ProteinMPNN official preflight passes and 3 from-scratch full1000 refits finish;
- [ ] NA-MPNN official preflight passes and 3 from-scratch full1000 refits finish;
- [ ] primary DM-ICF P/R/C/Delta/Alpha/Joint development + full1000 refit finishes for 3 seeds;
- [ ] DeltaC development drift audit archived;
- [ ] partner-blind and geometry-only controls trained/refit;
- [ ] runtime budget profiled on development only;
- [ ] `EVALUATION_PROTOCOL_LOCK.json` frozen;
- [ ] Tier A final100 completed for all primary seeds;
- [ ] Tier B diagnostic battery completed for analysis seed;
- [ ] official baseline final100 output exported;
- [ ] component ladder/control run manifests complete;
- [ ] H1-H4 confirmatory complex-level statistics complete;
- [ ] broad secondary/robustness/interpretability reports archived;
- [ ] exact git/config/manifest/upstream SHAs and software environment archived.

A code-complete repository is **not** the same thing as a completed experiment. Do not claim experimental completion until the real data and GPU runs exist.
