# Audit v3 — deep review, corrections, and remaining limits

This document is the authoritative post-review record for the 1k/1k/1.1k mini-pilot. It supersedes earlier statements that the repository was already ready for expensive training.

## Bottom line

The scientific core is retained:

```text
sequence preference = intramolecular structural prior + local cross-molecular selection
Gamma_ij = alpha_ij * (C + DeltaC_ij)
```

The third audit did **not** find a reason to redesign DM-ICF. It did find several execution, fairness and interpretation defects around the model. Those defects were corrected in code before any reported GPU run.

The repository should be considered:

```text
scientific design:        frozen for the mini-pilot
software contracts:       audited and CI-gated
real dataset manifests:   NOT yet frozen until the data pipeline is actually run
GPU experiment status:    NO-GO until the real data audit and baseline preflight pass
```

---

## 1. Official baseline execution was not actually safe

### Found

The previous wrapper assumed training command-line interfaces that the pinned upstream repositories do not expose.

- ProteinMPNN training accepts `--path_for_training_data`, not separate train/validation-cluster CLI flags.
- NA-MPNN `na_run.py` consumes one positional JSON configuration path.
- NA-MPNN inference did not expose the previously supplied `--rna_backbone_noise` flag.
- The NA-MPNN PPM export was mapped in A/C/G/U-like shared-token order while this project uses A/U/G/C, silently exchanging C and U at the project index level.
- Pinned `na_run.py` concatenates `BASE_FOLDER + 'log.txt'`, so a missing trailing slash sends files to the wrong path.

### Fixed

- Official baselines are pinned in `third_party/LOCK.json` to immutable upstream SHAs.
- `tools/run_official_baselines.py` drives unmodified upstream code through the verified interfaces.
- `tools/preflight_official_baselines.py` checks those assumptions against the pinned source before GPU use.
- NA probability export is explicitly `AUGC = DA,DT,DG,DC` when shared tokens are active.
- The unsupported inference flag was removed.
- `BASE_FOLDER` is normalized with a trailing separator.

### Remaining upstream reproducibility limitation

ProteinMPNN's pinned training DataLoader workers explicitly call NumPy reseeding without a supplied seed. The main process is seeded by `tools/run_seeded_upstream.py`, but we do **not** claim bitwise repeatability of the third-party training implementation. Three independent reported runs are therefore the reproducibility unit.

---

## 2. Full-1,000 refit previously changed the optimization algorithm

### Found

Development might select epoch 12 of a 150-epoch curriculum. The old refit then treated 12 as the complete schedule and compressed curriculum, cosine decay and gradual unfreezing into 12 epochs. That is not “refit the selected model on 1,000 samples”; it is a different training algorithm.

### Fixed

A validation-selected epoch is now a **prefix length of the original schedule**.

Example:

```text
development horizon = 150 epochs
best epoch count     = 12
full-1000 refit      = replay epochs 1..12 on the same 150-epoch schedule horizon
```

Checkpoints record both `selected_epoch_count` and `schedule_horizon_epochs`.

---

## 3. Scratch joint control was handicapped

### Found

The random-initialized scratch control inherited gradual unfreezing intended to protect pretrained representations. Freezing random encoder layers is not a fair scratch comparison.

### Fixed

The scratch control has an explicit `all_trainable_from_start` joint mode. Primary pretrained DM-ICF retains `pretrained_gradual` unfreezing.

---

## 4. C / DeltaC / alpha ownership was too entangled

### Found

If Alpha stage keeps updating the contextual encoder and DeltaC, an Alpha-stage improvement cannot be attributed to neighbour relevance. If C also drifts in Joint stage, the Stage-C heatmap ceases to be a stable learned anchor.

### Fixed primary ownership

```text
Stage C      : C only
Stage DeltaC : interaction q + DeltaC only
Stage Alpha  : relevance score/tau only
Stage Joint  : context field + lambdas + gradual pretrained-encoder adaptation
               C remains frozen
```

This gives the component ladder an interpretable meaning.

### Additional residual-field audit

Even with C frozen, `mean(DeltaC)` can be non-zero. `evaluation/field_audit.py` therefore reports:

- Stage-C anchor C;
- mean DeltaC;
- alpha-weighted mean DeltaC;
- `C_eff = C + mean(DeltaC)`;
- mean-shift / residual-RMS ratios.

If mean drift is large, the paper must call raw C the **Stage-C global anchor**, not the final population-average compatibility field.

---

## 5. Interface labels were confounded with the model graph

### Found

Screening defined contact by full heavy atoms at 6 A, while runtime interface masks came from the model's wider 8-A capped PR graph. Changing PR cutoff/neighbour cap would therefore change “interface recovery/NLL”, which is scientifically unacceptable.

A second, subtler risk appears if the fix simply uses native heavy atoms everywhere: SPIR at real design time cannot rely on native side chains or native base atoms.

### Fixed with two roles

**Reporting/training interface**

- computed from clean, unaugmented full-heavy-atom coordinates;
- fixed 6-A biological contact definition;
- used for PI/RI loss grouping, interface recovery/NLL and external-baseline labels;
- independent of DM-ICF PR graph cutoff/cap and coordinate-noise augmentation.

**Inference refinement region**

- derived only from sequence-neutral PR message-graph indices;
- used by SPIR to decide which positions are eligible for reopening;
- available from the supplied target backbones without native side-chain/base identity.

---

## 6. Joint validation did not represent joint decoding

### Found

A single both-sides-full-mask forward pass makes partner contributions zero on unknown edges. It mainly measures the two structural priors and is a poor checkpoint-selection proxy for sequential joint generation.

### Fixed

Joint validation combines:

1. Protein-conditional canonical-interface normalized NLL;
2. RNA-conditional canonical-interface normalized NLL;
3. deterministic teacher-forced sequential joint normalized pseudo-NLL.

The sequential term follows mixed, Protein-first and RNA-first fixed orders. A token is scored while unknown and only then is its **native token** revealed to later positions. The sequential term is evaluated on a fixed hash-selected validation subset to keep checkpoint selection computationally manageable.

---

## 7. Prior validation was too easy

### Found

The final complex test was family-strict, but the 900/100 Protein and RNA structural-prior development split itself was random. Closely related structures could make prior checkpoint selection look better than true generalization.

### Fixed

- Protein prior validation holds out complete P30 components.
- RNA prior validation holds out connected components under either R80 or Rfam.
- `pr-pilot audit-data` now has a separate prior-validation leakage audit.

---

## 8. RNA chain-view fallback could reuse a downstream complex backbone

### Found

RNA chains extracted from experimental Protein-RNA complexes are legitimate partner-hidden prior views, but a view coming from one of the exact frozen 1,100 downstream complex structures would let pretraining see the downstream RNA backbone in advance.

### Fixed

After the 1,100 complex pool is frozen, any extracted RNA prior view whose `source_complex_sample_id` belongs to that frozen complex pool is removed. Other screened experimental complexes remain eligible, subject to the final-test R80/Rfam/exact-sequence purge.

---

## 9. Complex length screening was asymmetric

### Found

Complex screening enforced minimum Protein/RNA lengths and total `P+R <= 1000`, but not the configured individual maximum Protein/RNA limits.

### Fixed

Complex mother samples must satisfy all three simultaneously:

```text
30 <= Protein <= 1000
5  <= RNA     <= 500
Protein + RNA <= 1000
```

---

## 10. Data-efficiency subsets were not nested

### Found

The old 10/25/50/100% subsets changed the random seed with fraction, so the 10% set was not guaranteed to be contained in 25%, etc.

### Fixed

One deterministic ranking is used per training seed. Each fraction is a prefix of that same ranking.

---

## 11. Final-100 evaluation was computationally overbuilt

### Found

64 candidates × multiple orders × SPIR variants × three seeds × 100 complexes can make the evaluation battery more expensive than the small training pilot itself.

### Fixed

- all three primary seeds receive inexpensive core final-100 evaluation;
- heavyweight mechanistic/order/SPIR candidate-generation analyses run on one **predeclared analysis seed**;
- primary sequence generation retains 64 candidates/complex;
- repeated order/SPIR ablation cells default to 16 candidates/complex;
- budgets are written to the output rather than hidden.

This changes compute allocation, not the primary scientific claim.

---

## 12. Too many tests were implicitly “primary”

### Found

Dozens of robustness and interpretability analyses should not all form one confirmatory multiplicity family.

### Fixed

`configs/hypotheses.yaml` freezes a compact primary family:

- full DM-ICF vs dual structural priors;
- full DM-ICF vs partner-blind;
- full DM-ICF vs geometry-only capacity control;
- full DM-ICF vs global-C-only;
- partner scrambling interface degradation.

Holm correction applies to this primary family only. PMI, DeltaC, alpha, robustness, decoding and calibration analyses are secondary/exploratory with effect sizes and confidence intervals.

For in-silico partner scramble/mutation/edge-removal experiments, use **model-interventional sensitivity**, not claims of biological causality.

---

## 13. Configuration semantics are now audited

`tools/audit_config_usage.py` fails CI when a configuration leaf is neither found in runtime code nor explicitly declared as a hard protocol assertion. This prevents a YAML option from being presented as an implemented trick simply because it exists in the config file.

---

# Remaining deliberate limitations

These are not hidden defects.

### Per-chain loss balancing

The current loss is balanced across Protein/RNA and interface/non-interface groups, but it does not additionally force equal contribution from every individual chain inside a multi-chain mother sample. This is recorded as a pilot limitation rather than retrofitted at the last minute. Multi-chain versus single-chain performance should be reported as a stratified secondary analysis. If the larger dataset later contains many asymmetric multi-chain samples, per-chain balancing should be revisited before the full-scale study.

### Biological assembly 1

The pilot uses biological assembly 1 and rejects entries whose required PR contact is absent there. Alternative assembly enumeration is deferred to the full dataset, where all assemblies from one deposition must remain grouped.

### Experimental structures only

No Boltz/AF-predicted complex structures enter the primary pilot. Predicted-structure augmentation is a later experiment.

### Sample-wise training

The pilot favors auditability over throughput. Dynamic token batching can be added only after the sample-wise implementation has been validated on real data.

### No physical-energy interpretation

C and C+DeltaC are conditional sequence-compatibility contributions, not binding free energies.

---

# GO / NO-GO gate

Passing repository CI is necessary but not sufficient for GPU training.

GPU status remains **NO-GO** until all of the following are true:

```text
[ ] current commit CI passes
[ ] pinned official-baseline preflight passes
[ ] real RCSB/Rfam/MMseqs/Infernal versions and checksums archived
[ ] 1000 Protein / 1000 RNA / 1100 complex manifests frozen
[ ] strict final 100 frozen before prior sampling
[ ] extracted RNA views from frozen complex pool purged
[ ] Protein prior P30 validation separation passes
[ ] RNA prior R80/Rfam validation separation passes
[ ] final-test P30/R80/Rfam/exact-sequence purge passes
[ ] one real ProteinMPNN conversion/inference smoke test passes
[ ] one real NA-MPNN conversion/inference smoke test passes
[ ] config semantic audit reports no dead/unknown knobs
```

Only after this gate should the first expensive training run start.
