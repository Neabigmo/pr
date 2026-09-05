# PR Mini-Pilot Runbook

This runbook is the operational order of execution. Do not skip stages. Every command should be executed from a clean git commit and should write its resolved configuration and manifest checksums to `artifacts/`.

## 1. Environment

Recommended:

```bash
conda create -n pr-pilot python=3.11 -y
conda activate pr-pilot
pip install -e '.[dev]'
pytest
```

For the user's Windows workstation, the preferred existing environment can be used instead, but the exact Python/PyTorch/CUDA versions must be written into the run metadata.

## 2. Freeze upstream baseline versions

Clone third-party code outside our package namespace:

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/dauparas/ProteinMPNN.git proteinmpnn
git clone https://github.com/baker-laboratory/NA-MPNN.git na_mpnn
```

Then record exact immutable SHAs in:

```text
third_party/LOCK.json
```

Never run `git pull` during a reported experiment without updating the lock and starting a new experiment version.

## 3. Build eligible source tables

The pilot assumes three normalized eligible tables exist:

```text
data/eligible/proteins.tsv
data/eligible/rnas.tsv
data/eligible/complexes.tsv
```

### proteins.tsv minimum columns

```text
sample_id
structure_path
sequence
sequence_hash
protein_cluster_p90
protein_cluster_p40
protein_cluster_p30
experimental
resolution
release_date
```

### rnas.tsv minimum columns

```text
sample_id
structure_path
sequence
sequence_hash
rna_cluster_r90
rna_cluster_r80
rfam_family
experimental
resolution
release_date
```

### complexes.tsv minimum columns

```text
sample_id
structure_path
protein_sequence
rna_sequence
protein_hash
rna_hash
protein_cluster_p30
rna_cluster_r80
rfam_family
mother_sample_id
experimental
resolution
release_date
```

The complex table must contain **biological-assembly, interface-connected mother samples**, not arbitrary asymmetric-unit rows.

## 4. Freeze the pilot manifests

Use the manifest library from `pr_pilot.data.manifest`. A CLI wrapper should be run after installation:

```bash
python -m pr_pilot.cli freeze \
  --config configs/pilot.yaml \
  --proteins data/eligible/proteins.tsv \
  --rnas data/eligible/rnas.tsv \
  --complexes data/eligible/complexes.tsv \
  --out artifacts/manifests
```

Expected outputs:

```text
protein_pool.tsv      # 1000
protein_train.tsv     # 900
protein_val.tsv       # 100
rna_pool.tsv          # 1000
rna_train.tsv         # 900
rna_val.tsv           # 100
complex_pool.tsv      # 1100
complex_dev.tsv       # 1000
complex_train.tsv     # 900
complex_val.tsv       # 100
complex_test.tsv      # 100 -- LOCKED FINAL HOLDOUT
```

Immediately commit only the checksums/IDs if structure redistribution is inappropriate. Do not leak final test labels into notebooks used for development.

## 5. Run data audit

Required before any training:

```bash
python -m pr_pilot.cli audit-data \
  --config configs/pilot.yaml \
  --manifest-root artifacts/manifests \
  --out artifacts/data_audit
```

The audit must verify:

- exact counts;
- no duplicate mother samples;
- no exact sequence leakage;
- P30/Rfam strict overlap status;
- protein/RNA length distributions;
- resolution distribution;
- interface residue/nucleotide counts;
- PR edge counts under multiple distance cutoffs;
- 20x4 AA/base observations;
- base/sugar/phosphate contact breakdown where possible;
- missing-atom rates;
- RNA prohibited-atom leakage audit;
- manifest SHA256.

**Do not train if the audit fails.**

## 6. Baseline A: ProteinMPNN

### 6.1 Prepare upstream data

```bash
python tools/prepare_proteinmpnn.py \
  --manifest artifacts/manifests/protein_train.tsv \
  --validation artifacts/manifests/protein_val.tsv \
  --upstream third_party/proteinmpnn \
  --out artifacts/baselines/proteinmpnn/data
```

The adapter must preserve `sample_id` and emit a report proving exactly 900 train / 100 validation structures were converted.

### 6.2 Train from scratch

Use the upstream training entrypoint corresponding to the pinned SHA. Do not silently use published pretrained weights.

The exact command must be written to:

```text
artifacts/baselines/proteinmpnn/train_command.txt
```

### 6.3 Export standardized predictions

```bash
python tools/export_proteinmpnn.py ...
```

Output schema is defined in `pr_pilot.baselines.wrappers.standardized_prediction_schema()`.

## 7. Baseline B: MPNN-fixbb / NA-MPNN RNA

Repeat the same logic using `third_party/na_mpnn` and the frozen 900/100 RNA manifest.

No published pretrained weight may be used for the from-scratch comparison unless explicitly named as a separate transfer-learning baseline.

## 8. DM-ICF Stage P: protein structural prior

```bash
python -m pr_pilot.cli train \
  --stage protein_prior \
  --config configs/pilot.yaml \
  --manifest artifacts/manifests/protein_train.tsv \
  --validation artifacts/manifests/protein_val.tsv
```

Checkpoint selection is allowed only by protein validation normalized NLL.

## 9. DM-ICF Stage R: RNA structural prior

```bash
python -m pr_pilot.cli train \
  --stage rna_prior \
  --config configs/pilot.yaml \
  --manifest artifacts/manifests/rna_train.tsv \
  --validation artifacts/manifests/rna_val.tsv
```

The RNA input view must pass the no-base-identity-atom unit tests.

## 10. Stage C: global 20x4 C

Initialize from the selected P/R prior checkpoints.

```bash
python -m pr_pilot.cli train \
  --stage global_c \
  --config configs/pilot.yaml \
  --manifest artifacts/manifests/complex_train.tsv \
  --validation artifacts/manifests/complex_val.tsv
```

Hard requirements:

- priors frozen;
- C small random initialized;
- no empirical PMI input;
- protein-conditional and RNA-conditional batches alternate 1:1;
- no learned DeltaC;
- no learned alpha;
- test manifest inaccessible.

Save C heatmaps for debugging, but never use final-test PMI to choose training duration.

## 11. Stage Delta: contextual residual

```bash
python -m pr_pilot.cli train --stage delta_c ...
```

Hard requirements:

- priors frozen;
- C frozen;
- DeltaC output zero at step 0;
- rich PR geometry enabled in primary run;
- no explicit DeltaC norm penalty.

## 12. Stage Alpha

```bash
python -m pr_pilot.cli train --stage alpha ...
```

Hard requirements:

- alpha residual starts at zero;
- initial alpha equals the distance prior;
- early weak entropy regularization anneals to zero;
- PR edge dropout and partner-token dropout are logged.

## 13. Stage Joint

```bash
python -m pr_pilot.cli train --stage joint ...
```

Task schedule: 2:2:1 -> 1:1:1.

Use gradual unfreezing and discriminative learning rates. Raw and normalized PI/PN/RI/RN losses must be logged separately.

## 14. Frozen hyperparameter decision

At this point—and **before reading the final test metrics**—write:

```text
artifacts/FROZEN_FINAL_CONFIGURATION.yaml
```

It must contain:

- architecture;
- coordinate noise;
- cutoffs/neighbour caps;
- loss rules;
- selected checkpoint policy;
- SPIR fraction/temperature;
- inference candidate count;
- all ablation definitions.

## 15. Optional full-development refit

Retrain the selected configuration on:

- all 1000 protein structures;
- all 1000 RNA structures;
- all 1000 complex development samples.

Do **not** change hyperparameters after this point.

## 16. Generate joint candidates

```bash
python -m pr_pilot.cli design \
  --config artifacts/FROZEN_FINAL_CONFIGURATION.yaml \
  --manifest artifacts/manifests/complex_test.tsv \
  --mode joint \
  --candidates 64
```

Generation must include random mixed decoding and create both pre-SPIR and post-SPIR outputs.

## 17. Run the final 100-complex battery

```bash
python -m pr_pilot.cli evaluate \
  --config artifacts/FROZEN_FINAL_CONFIGURATION.yaml \
  --test-manifest artifacts/manifests/complex_test.tsv \
  --registry all \
  --out artifacts/metrics/final_100
```

Mandatory families are enumerated in `pr_pilot.evaluation.battery.MANDATORY_TESTS`.

## 18. Primary statistical report

```bash
python -m pr_pilot.cli statistics \
  --metrics artifacts/metrics/final_100 \
  --bootstrap 10000 \
  --out artifacts/statistics/final_100
```

Paired resampling unit = complex, not residue.

## 19. Interpretability report

Required outputs:

```text
artifacts/interpretability/C_learned.csv
artifacts/interpretability/C_heatmap.*
artifacts/interpretability/PMI_experimental.csv
artifacts/interpretability/C_vs_PMI.json
artifacts/interpretability/DeltaC_summary.csv
artifacts/interpretability/DeltaC_examples/
artifacts/interpretability/alpha_summary.csv
artifacts/interpretability/alpha_examples/
```

## 20. Robustness report

Required stress matrix:

```text
coordinate noise: 0, .05, .10, .20 Å
PR edge drop:      0, .05, .10, .20
partner hide:      0, .10, .20, .40
decoding orders:   multiple independent random orders
SPIR:              none, 1 cycle, repeated
```

## 21. Data-efficiency report

Nested complex-training subsets: 10/25/50/100%.

Compare at minimum:

- scratch joint model;
- dual structural priors + DM-ICF.

Do not choose the subset seeds separately for each model; all methods use identical nested subsets.

## 22. External structure prediction (optional but valuable)

Run after the sequence-design evaluation. Structure predictors must remain external evaluators in this pilot.

Do not call a model score “binding energy”. Keep protein, RNA and interface structural consistency metrics separate.

## 23. Completion checklist

The experiment is not complete until:

- [ ] 1,000 protein manifest frozen
- [ ] 1,000 RNA manifest frozen
- [ ] 1,100 complex manifest frozen
- [ ] 100 final test locked
- [ ] ProteinMPNN trained from scratch
- [ ] MPNN-fixbb/NA-MPNN trained from scratch
- [ ] Stage P complete
- [ ] Stage R complete
- [ ] Stage C complete
- [ ] Stage Delta complete
- [ ] Stage Alpha complete
- [ ] Stage Joint complete
- [ ] SPIR implemented
- [ ] all core ablations complete
- [ ] partner scramble complete
- [ ] local counterfactual complete
- [ ] C vs PMI complete
- [ ] DeltaC analysis complete
- [ ] alpha analysis complete
- [ ] robustness matrix complete
- [ ] calibration complete
- [ ] decoding-order analysis complete
- [ ] data-efficiency curves complete
- [ ] paired statistics complete
- [ ] final reproducibility manifest complete
