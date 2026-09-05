# PR Mini-Pilot Runbook — audited v3

This is the canonical execution order. Do not use older ad-hoc commands from notebooks or issue comments.

## 0. Rules before anything expensive

- work from a clean commit;
- archive resolved config, git SHA, software versions and manifest SHA256s for every run;
- never open final-100 metrics before the development configuration is frozen;
- primary final checkpoints must be full-1,000 refits, not 900-sample development checkpoints;
- all methods use the exact frozen IDs assigned to their task.

## 1. Environment and source audit

```bash
conda create -n pr-pilot python=3.11 -y
conda activate pr-pilot
pip install -e '.[dev]'

python -m compileall -q src tests tools
pytest
python tools/audit_config_usage.py --config configs/pilot.yaml --repo-root .
pr-pilot --help
```

The config audit must report zero `unknown_or_dead` keys.

## 2. Pin official baseline repositories

Copy:

```text
third_party/LOCK.template.json -> third_party/LOCK.json
```

Replace both placeholders with reviewed immutable 40-character SHAs for:

- `dauparas/ProteinMPNN`;
- `baker-laboratory/NA-MPNN`.

Then run:

```bash
python tools/preflight_official_baselines.py \
  --repo-root . \
  --third-party-root third_party/checkouts \
  --out artifacts/preflight/baselines.json
```

This checks the pinned upstream CLI contracts, including ProteinMPNN's actual training arguments, NA-MPNN's positional JSON training interface, and the project AUGC probability-column mapping.

## 3. Data acquisition

Follow `docs/DATA_PIPELINE.md` exactly.

### 3.1 Discover

```bash
mkdir -p data/discovery
pr-pilot discover-rcsb --kind protein --out data/discovery/protein.tsv
pr-pilot discover-rcsb --kind rna --out data/discovery/rna.tsv
pr-pilot discover-rcsb --kind complex --out data/discovery/complex.tsv
```

### 3.2 Oversized deterministic downloads

Do not stop at exactly 1,000/1,000/1,100 raw entries.

```bash
pr-pilot download-rcsb --kind protein --candidates data/discovery/protein.tsv \
  --out data/raw/protein --seed 20260905 --max-candidates 5000
pr-pilot download-rcsb --kind rna --candidates data/discovery/rna.tsv \
  --out data/raw/rna --seed 20261006 --max-candidates 3000
pr-pilot download-rcsb --kind complex --candidates data/discovery/complex.tsv \
  --out data/raw/complex --seed 20261107 --max-candidates 6000
```

### 3.3 Coordinate screening

```bash
pr-pilot screen --kind protein --config configs/pilot.yaml \
  --download-manifest data/raw/protein/download_manifest.tsv --out data/screened/protein
pr-pilot screen --kind rna --config configs/pilot.yaml \
  --download-manifest data/raw/rna/download_manifest.tsv --out data/screened/rna
pr-pilot screen --kind complex --config configs/pilot.yaml \
  --download-manifest data/raw/complex/download_manifest.tsv --out data/screened/complex
```

Complex screening enforces Protein and RNA individual length limits as well as total length. Rejection tables are permanent audit artifacts.

### 3.4 Build RNA prior candidate views

```bash
python tools/build_rna_candidate_pool.py \
  --standalone data/screened/rna/rna_eligible.tsv \
  --complexes data/screened/complex/complex_eligible.tsv \
  --out data/screened/rna_prior_candidates.tsv
```

For complex-derived RNA views, the Protein partner is not loaded during RNA-prior training.

### 3.5 Joint clustering and Rfam

```bash
pr-pilot download-rfam --out data/reference/rfam

pr-pilot annotate \
  --proteins data/screened/protein/protein_eligible.tsv \
  --rnas data/screened/rna_prior_candidates.tsv \
  --complexes data/screened/complex/complex_eligible.tsv \
  --rfam-cm-gz data/reference/rfam/Rfam.cm.gz \
  --rfam-clanin data/reference/rfam/Rfam.clanin \
  --cpu 16 \
  --out data/annotated
```

Single-polymer candidates and complex chains must be clustered in the same universe.

## 4. Freeze manifests — final test first

```bash
rm -rf manifests/pilot_v1
pr-pilot freeze \
  --config configs/pilot.yaml \
  --proteins data/annotated/protein_annotated.tsv \
  --rnas data/annotated/rna_annotated.tsv \
  --complexes data/annotated/complex_annotated.tsv \
  --out manifests/pilot_v1
```

Expected:

```text
protein_pool  1000 = 900 train + 100 P30-disjoint validation
rna_pool      1000 = 900 train + 100 R80/Rfam-disjoint validation
complex_dev   1000 = 900 train + 100 bilateral-disjoint validation
complex_test   100 = immutable strict P30 + R80 + Rfam holdout
```

The final 100 are frozen before prior-pool sampling. Exact sequences and final-test P30/R80/Rfam neighbours are purged from both prior pools.

## 5. Mandatory manifest audit

```bash
pr-pilot audit-data \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/data_audit
```

Also inspect manually:

- length distributions;
- experimental method/resolution distribution;
- RNA standalone vs extracted-chain source fraction;
- interface size and missingness;
- family/component sizes;
- rejection reasons;
- strict dev/test covariate shifts.

Do not “repair” an awkward final 100 after seeing performance.

## 6. Prepare and smoke-test official baselines

```bash
python tools/run_official_baselines.py \
  --repo-root . \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/baselines \
  --third-party-root third_party/checkouts \
  --prepare-only
```

Before full training, manually run one prepared Protein and one prepared RNA sample through the pinned upstream code. The canonicalized sequence must match the frozen manifest exactly.

## 7. Development training

Dry-run the project orchestration first:

```bash
python tools/run_pilot_experiments.py \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/pilot_experiments \
  --families primary \
  --dry-run
```

Then execute only after the data and baseline gates pass:

```bash
python tools/run_pilot_experiments.py \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/pilot_experiments \
  --families primary \
  --execute
```

Development stage order is mandatory:

```text
Protein prior -> RNA prior -> C -> DeltaC -> alpha -> joint
```

Parameter ownership:

- C: C only;
- DeltaC: interaction encoder + DeltaC only;
- alpha: relevance/tau only;
- joint: C remains frozen; context field/lambdas adapt; pretrained encoders gradually unfreeze.

For the scratch-joint control, all random-initialized encoder parameters are trainable from step 0.

## 8. Full-1,000 refit

This is automatically performed by `run_pilot_experiments.py` for the primary track.

A development best epoch is interpreted as a **prefix of the original schedule**. Refit uses the same curriculum/cosine/unfreezing horizon and stops at that selected prefix; it does not compress the entire schedule into fewer epochs.

Only `primary_refit_full1000/.../joint/refit.pt` checkpoints support the primary final claim.

## 9. Official baseline full training and refit

```bash
python tools/run_official_baselines.py \
  --repo-root . \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/baselines \
  --third-party-root third_party/checkouts
```

The wrapper uses the unmodified pinned upstream code and records every command. ProteinMPNN upstream worker reseeding prevents a claim of bitwise determinism; therefore report independent runs rather than claiming exact rerun identity.

## 10. Freeze final configuration and hypotheses

Before opening final-test metrics, archive:

```text
artifacts/FROZEN_FINAL_CONFIGURATION.yaml
configs/hypotheses.yaml
third_party/LOCK.json
manifest checksums
repository commit SHA
```

The primary family in `configs/hypotheses.yaml` is the only family receiving Holm confirmatory correction. Other tests are secondary/exploratory.

## 11. Final 100 evaluation

All three primary seeds receive core evaluation. Only the predeclared `evaluation.analysis_seed` receives the heavyweight mechanistic/order/SPIR battery.

The canonical interface is a full-heavy-atom 6 Å label and is independent of the model's 8 Å/capped PR message graph.

Heavy candidate ablations use 16 candidates/cell by default. Primary design generation may still use 64 candidates/complex.

## 12. Internal controls and component ladder

```bash
python tools/run_pilot_experiments.py \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/pilot_experiments \
  --families component_ladder \
  --execute
```

Required controls include:

- scratch joint;
- dual structural prior;
- + global C;
- + DeltaC;
- + alpha;
- full joint;
- partner-blind;
- geometry-only capacity control;
- fixed empirical-PMI reference;
- backbone-only-context C control.

External ProteinMPNN/NA-MPNN references must not be presented as substitutes for these same-data partner-coupling controls.

## 13. Data-efficiency curve

The 10/25/50/100% complex-training subsets must be **nested**. The v3 orchestrator uses one deterministic ranking seed for every fraction.

## 14. Interpretability audits

Run/retain:

```bash
python -m pr_pilot.evaluation.field_audit \
  --config configs/pilot.yaml \
  --checkpoint <final-refit.pt> \
  --manifest manifests/pilot_v1/complex_dev.tsv \
  --out artifacts/interpretability/delta_c_drift
```

Report:

- Stage-C anchor `C`;
- raw and double-centered C views;
- independent heavy-atom empirical PMI;
- `mean(DeltaC)` and alpha-weighted mean drift;
- `C_eff = C + mean(DeltaC)` when residual drift is non-negligible;
- alpha entropy/effective-neighbour behavior.

Do not call C or C+DeltaC a thermodynamic binding energy.

## 15. Completion gate

The pilot is complete only when:

- [ ] source/CI/config audit passes;
- [ ] official-baseline preflight passes pinned SHAs;
- [ ] frozen 1000/1000/1100 manifests and rejection tables archived;
- [ ] strict data audit passes;
- [ ] three primary development + full-1000 refit runs complete;
- [ ] official ProteinMPNN and NA-MPNN development + full-1000 refits complete;
- [ ] internal controls/component ladder complete;
- [ ] nested data-efficiency analysis complete;
- [ ] final 100 core metrics complete for all primary seeds;
- [ ] heavy final battery complete for the predeclared analysis seed;
- [ ] C/PMI, DeltaC drift and alpha analyses complete;
- [ ] paired complex-level bootstrap and predeclared multiplicity handling complete;
- [ ] exact run manifests, commands, versions and checksums archived.
