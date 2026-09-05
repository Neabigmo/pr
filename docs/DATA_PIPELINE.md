# Mini-pilot data pipeline: download -> screen -> annotate -> freeze -> audit

This document is the **only supported ordering** for constructing the 1,000 protein / 1,000 RNA / 1,100 Protein-RNA pilot. Do not hand-pick IDs after looking at model results.

## 0. External requirements

Python dependencies are installed with the project. Two external sequence-analysis programs are also required before the `annotate` step:

- **MMseqs2**: joint P90/P40/P30 and RNA R90/R80 clustering.
- **Infernal** (`cmscan`, `cmpress`): Rfam family annotation.

Record their versions:

```bash
mmseqs version > artifacts/software_versions.txt
cmscan -h | head -n 3 >> artifacts/software_versions.txt
python --version >> artifacts/software_versions.txt
```

The pipeline fails if these programs are unavailable; it never substitutes an unrecorded approximate clustering method.

---

## 1. Discover RCSB candidates

Discovery is deliberately broad. Scientific filtering happens from downloaded coordinates and every rejection is recorded.

```bash
mkdir -p data/discovery

pr-pilot discover-rcsb --kind protein --out data/discovery/protein.tsv
pr-pilot discover-rcsb --kind rna     --out data/discovery/rna.tsv
pr-pilot discover-rcsb --kind complex --out data/discovery/complex.tsv
```

The exact RCSB query is written next to each TSV as `*.query.json`.

### Candidate definitions

- Protein candidates: experimental entries with protein and no RNA/DNA.
- Standalone RNA candidates: experimental RNA entries with no protein/DNA.
- Complex candidates: experimental entries containing both protein and RNA and no DNA.

RNA structural-prior data are later augmented by **RNA-chain views extracted from screened experimental Protein-RNA complexes**. This is preferred over weakening structural QC if standalone RNA alone is insufficient.

---

## 2. Download a deliberately oversized candidate pool

Do **not** download exactly 1,000/1,000/1,100 structures. QC, clustering and strict-test purge will remove many entries. Start with a much larger deterministic prefix and enlarge it if the eligible pool remains too small.

Example pilot starting point:

```bash
mkdir -p data/raw/{protein,rna,complex}

pr-pilot download-rcsb --kind protein \
  --candidates data/discovery/protein.tsv \
  --out data/raw/protein --seed 20260905 --max-candidates 5000

pr-pilot download-rcsb --kind rna \
  --candidates data/discovery/rna.tsv \
  --out data/raw/rna --seed 20261006 --max-candidates 3000

pr-pilot download-rcsb --kind complex \
  --candidates data/discovery/complex.tsv \
  --out data/raw/complex --seed 20261107 --max-candidates 6000
```

Protein/RNA single-molecule views use deposited entry coordinates. Complexes use RCSB biological assembly 1. Download records contain URL, local path, SHA256 and byte count; failed downloads are written separately and are never silently replaced.

If screening produces fewer eligible candidates than required after strict-test purge, increase `--max-candidates`; do not relax biology/QC behind the scenes.

---

## 3. Coordinate-level screening

```bash
mkdir -p data/screened/{protein,rna,complex}

pr-pilot screen --kind protein --config configs/pilot.yaml \
  --download-manifest data/raw/protein/download_manifest.tsv \
  --out data/screened/protein

pr-pilot screen --kind rna --config configs/pilot.yaml \
  --download-manifest data/raw/rna/download_manifest.tsv \
  --out data/screened/rna

pr-pilot screen --kind complex --config configs/pilot.yaml \
  --download-manifest data/raw/complex/download_manifest.tsv \
  --out data/screened/complex
```

### Hard filters in the primary pilot

**All data**
- experimental structure only;
- no DNA / nucleic-acid hybrid target;
- unsupported target-polymer modifications are rejected rather than guessed;
- standard/CCD-modified residues that map unambiguously to canonical protein/RNA tokens use one shared vocabulary in screening, DM-ICF and both official baselines;
- resolution <= 4.0 A when a resolution applies; NMR without a conventional resolution is allowed and recorded separately.

**Protein prior**
- length 30--1,000;
- complete N/CA/C/O for selected chain;
- no RNA partner in the protein-only candidate view.

**Standalone RNA prior**
- length 5--500;
- sequence restricted to canonical A/U/G/C after legitimate CCD parent mapping;
- stable C1'/C3'/C4' sugar frame;
- no protein partner in the standalone view.

**Protein-RNA complex**
- biological assembly view;
- actual heavy-atom Protein-RNA contact required, not metadata alone;
- at least 3 contacting residue-nucleotide pairs at the 6 A screening contact threshold;
- only Protein/RNA chains participating in the contact graph are retained as the mother sample;
- total selected Protein + RNA tokens <= 1,000;
- interface backbone missing-atom fraction <= 10%;
- ribosome/spliceosome keyword filter plus the total-size limit;
- all selected chains must have the reference atoms required by the runtime adapter.

Every rejection is written to `*_rejected.tsv` with an explicit reason. Never delete that file.

---

## 4. Build the RNA structural-prior candidate pool

The paper design allows RNA chains extracted from experimental complexes with the partner removed. This is especially important because 1,000 high-quality standalone RNA-only PDB entries may not remain after strict QC and test-family purge.

```bash
python tools/build_rna_candidate_pool.py \
  --standalone data/screened/rna/rna_eligible.tsv \
  --complexes data/screened/complex/complex_eligible.tsv \
  --out data/screened/rna_prior_candidates.tsv
```

This produces two recorded `source_view` classes:

1. `standalone_rna`;
2. `protein_rna_complex_chain_extracted`.

For class 2, the protein chain is never loaded by the RNA-prior dataloader. The exact source remains auditable through `source_complex_sample_id`.

---

## 5. Download Rfam and jointly annotate all candidates

```bash
pr-pilot download-rfam --out data/reference/rfam
```

Then jointly cluster **single-molecule candidates and complex chains in the same run**:

```bash
pr-pilot annotate \
  --proteins data/screened/protein/protein_eligible.tsv \
  --rnas data/screened/rna_prior_candidates.tsv \
  --complexes data/screened/complex/complex_eligible.tsv \
  --rfam-cm-gz data/reference/rfam/Rfam.cm.gz \
  --rfam-clanin data/reference/rfam/Rfam.clanin \
  --cpu 16 \
  --out data/annotated
```

Why joint annotation is mandatory:

- a P30 label generated in a separate clustering run is not directly comparable to another run;
- the same applies to RNA R80;
- final-test purge from structural-prior pools is only meaningful when labels share one clustering universe.

Generated hierarchy:

- protein P90 / P40 / P30;
- RNA R90 / R80;
- Rfam family labels from `cmscan --cut_ga --rfam`.

Multi-chain complexes retain all constituent cluster/family labels. The splitter treats **any** shared P30, **any** shared R80, or **any** shared Rfam as a link between samples.

---

## 6. Freeze the test set FIRST, then the prior pools

This ordering is non-negotiable.

```bash
rm -rf manifests/pilot_v1

pr-pilot freeze \
  --config configs/pilot.yaml \
  --proteins data/annotated/protein_annotated.tsv \
  --rnas data/annotated/rna_annotated.tsv \
  --complexes data/annotated/complex_annotated.tsv \
  --out manifests/pilot_v1
```

The freeze operation performs:

1. construct strict connected components under Protein P30 + RNA R80 + Rfam;
2. choose exactly 100 **whole-component** experimental complexes for final test;
3. remove every test-linked component from development candidates;
4. deterministically sample 1,000 complex development samples;
5. form strict 900/100 complex train/validation groups;
6. purge final-test exact sequences, P30, R80 and Rfam from the protein/RNA prior candidate pools;
7. only then sample 1,000 Protein and 1,000 RNA structural-prior structures;
8. freeze 900/100 single-molecule train/validation manifests;
9. write SHA256 metadata for every frozen manifest.

If an exact 100-sample strict component holdout is mathematically impossible, the program must fail. The response is to enlarge the eligible candidate universe or version an explicitly relaxed pilot; never split a homologous component silently.

---

## 7. Mandatory data audit before GPU training

```bash
pr-pilot audit-data \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/data_audit
```

This audit must pass before any baseline or DM-ICF training starts.

Minimum hard assertions:

- exactly 1,000 Protein prior structures = 900 train + 100 validation;
- exactly 1,000 RNA prior structures = 900 train + 100 validation;
- exactly 1,100 complexes = 1,000 development + 100 immutable final test;
- complex development = 900 train + 100 validation;
- final test is experimental only;
- no exact Protein/RNA sequence overlap between complex dev and test;
- no mother-sample overlap;
- no constituent P30 overlap;
- no constituent R80 overlap;
- no constituent Rfam overlap;
- no final-test P30/R80/Rfam/exact sequence in either structural-prior pool.

Also archive distributions for:

- Protein/RNA/total length;
- resolution/method;
- interface size and missing fraction;
- RNA source view (standalone vs extracted complex chain);
- P30/R80/Rfam family sizes;
- complex type/source where annotations are available.

Large train/test covariate shifts are reported, not hidden. The final 100 are intentionally strict OOD, so this pilot is **pseudo-random under bilateral novelty constraints**, not an IID random split.

---

## 8. Baseline preparation only after the same manifests are frozen

```bash
python tools/run_official_baselines.py \
  --repo-root . \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/baselines \
  --prepare-only
```

Both official converters verify that the canonicalized chain sequence is exactly the frozen manifest sequence. A method-specific conversion failure is an error and must be reported; the failed structure is not silently replaced for only one method.

---

## 9. Data versioning

Never overwrite a reported dataset version. A complete data version consists of:

```text
RCSB query JSON
+ download manifests and failures
+ screening eligible/rejected tables and summaries
+ Rfam checksums
+ MMseqs/Infernal logs
+ annotated candidate tables
+ frozen manifests and SHA256 metadata
+ data audit outputs
+ software versions
```

If any screening threshold, clustering universe, Rfam release or split seed changes, create `pilot_v2` rather than mutating `pilot_v1`.
