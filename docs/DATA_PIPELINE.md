# Mini-pilot data pipeline — audited v3

This is the only supported order for constructing the 1,000 Protein / 1,000 RNA / 1,100 Protein–RNA pilot. IDs are frozen before model results exist; rejected samples are never silently replaced after looking at performance.

## 0. External requirements

Required before annotation:

- MMseqs2;
- Infernal (`cmscan`, `cmpress`);
- official Rfam CM/clan files downloaded by the project.

Archive versions:

```bash
mkdir -p artifacts
mmseqs version > artifacts/software_versions.txt
cmscan -h | head -n 3 >> artifacts/software_versions.txt
python --version >> artifacts/software_versions.txt
```

## 1. Broad RCSB discovery

```bash
mkdir -p data/discovery
pr-pilot discover-rcsb --kind protein --out data/discovery/protein.tsv
pr-pilot discover-rcsb --kind rna --out data/discovery/rna.tsv
pr-pilot discover-rcsb --kind complex --out data/discovery/complex.tsv
```

Discovery is broad; scientific filtering occurs from coordinates and writes explicit rejection reasons.

Candidate intent:

- Protein: experimental Protein entries without RNA/DNA;
- standalone RNA: experimental RNA entries without Protein/DNA;
- complex: experimental entries containing Protein + RNA and no DNA.

## 2. Download oversized deterministic pools

Do not download exactly the target counts. QC, family grouping and test-family purge remove many candidates.

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

Single-polymer views use deposited coordinates. Complexes use biological assembly 1 in this pilot. Every successful file gets URL/path/SHA256/byte count; failures are archived.

If too few samples survive, enlarge the download universe. Do not weaken QC behind the scenes.

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

### Primary hard filters

All data:

- experimental only;
- no DNA/hybrid target;
- canonicalizable target residues only;
- resolution <= 4.0 A where applicable; NMR allowed and recorded separately;
- shared canonical residue vocabulary across screening, DM-ICF and baseline converters.

Protein prior:

- 30–1,000 residues;
- required backbone atoms;
- no RNA partner in the selected view.

Standalone RNA prior:

- 5–500 nucleotides;
- canonical A/U/G/C after legitimate parent mapping;
- valid C1'/C3'/C4' sugar frame;
- no Protein partner in the standalone view.

Protein–RNA complex:

- biological assembly 1;
- real full-heavy-atom Protein–RNA contact required;
- at least 3 contacting residue/nucleotide pairs at the 6-A screening threshold;
- only contact-connected Protein/RNA chains retained as the mother sample;
- 30 <= total selected Protein residues <= 1,000;
- 5 <= total selected RNA nucleotides <= 500;
- Protein + RNA <= 1,000 tokens;
- interface backbone missing-atom fraction <= 10%;
- ribosome/spliceosome keyword exclusion plus size control;
- reference atoms required by runtime present.

Every rejection remains in `*_rejected.tsv`.

### Important interface distinction

The **canonical supervised/reporting interface** is full-heavy-atom contact at 6 A on clean coordinates. It is independent of DM-ICF's wider/capped PR message graph. Training coordinate noise cannot change this label.

SPIR/design-time interface selection uses only the sequence-neutral PR graph, not native side-chain/base contact labels.

## 4. Build RNA structural-prior candidate views

High-quality standalone RNA alone may be insufficient after strict filtering. The paper design therefore allows RNA chains extracted from other experimental Protein–RNA complexes with the Protein partner removed.

```bash
python tools/build_rna_candidate_pool.py \
  --standalone data/screened/rna/rna_eligible.tsv \
  --complexes data/screened/complex/complex_eligible.tsv \
  --out data/screened/rna_prior_candidates.tsv
```

Two `source_view` classes are recorded:

1. `standalone_rna`;
2. `protein_rna_complex_chain_extracted`.

For class 2, Protein atoms are never loaded by the RNA-prior dataloader.

Additional v3 protection: after the exact 1,100 downstream complex pool is frozen, any extracted RNA view whose `source_complex_sample_id` belongs to that frozen pool is purged. Thus the RNA prior cannot see an exact downstream RNA backbone from the reported 1,100 complexes.

## 5. Joint clustering and Rfam annotation

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

Single-polymer candidates and complex chains are clustered in the **same MMseqs universe**. Otherwise P30/R80 labels from separate clustering runs are not comparable.

Outputs include Protein P90/P40/P30, RNA R90/R80 and Rfam family labels. Multi-chain complexes retain every constituent label; any shared P30, any shared R80 or any shared Rfam connects samples during strict splitting.

## 6. Freeze final test first

```bash
rm -rf manifests/pilot_v1

pr-pilot freeze \
  --config configs/pilot.yaml \
  --proteins data/annotated/protein_annotated.tsv \
  --rnas data/annotated/rna_annotated.tsv \
  --complexes data/annotated/complex_annotated.tsv \
  --out manifests/pilot_v1
```

Order is non-negotiable:

1. build connected complex components under Protein P30 + RNA R80 + Rfam;
2. choose exactly 100 whole-component experimental complexes for final test;
3. remove those components from development candidates;
4. sample 1,000 complex development samples;
5. split complex dev into strict 900/100 bilateral-disjoint train/validation;
6. purge final-test exact sequences/P30/R80/Rfam from Protein/RNA prior candidates;
7. purge RNA extracted-chain views sourced from the frozen 1,100 complex pool;
8. freeze 1,000 Protein prior structures with **P30-disjoint 900/100 validation**;
9. freeze 1,000 RNA prior structures with **R80-or-Rfam-component-disjoint 900/100 validation**;
10. write manifest SHA256 metadata.

If exact component sizes make a strict 100 impossible, fail and enlarge the candidate universe. Never split a homologous component to make the count fit.

## 7. Mandatory data audit

```bash
pr-pilot audit-data \
  --config configs/pilot.yaml \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/data_audit
```

Hard assertions include:

- exact 1,000 Protein / 1,000 RNA / 1,100 complex counts;
- final test = 100 experimental complexes;
- no exact complex dev/test sequence overlap;
- no mother-sample overlap;
- no P30, R80 or Rfam overlap across complex dev/test;
- no final-test exact/P30/R80/Rfam neighbour in either prior pool;
- no P30 overlap across Protein prior train/validation;
- no R80 or Rfam overlap across RNA prior train/validation.

Archive distributions for length, method/resolution, interface size/missingness, RNA source view, family sizes and complex source/type metadata when available. Strict OOD covariate shift is reported, not hidden.

## 8. Baseline preparation only after manifests are frozen

First verify pinned upstream contracts:

```bash
python tools/preflight_official_baselines.py \
  --repo-root . \
  --third-party-root third_party/checkouts \
  --out artifacts/preflight/baselines.json
```

Then convert exactly the frozen prior IDs:

```bash
python tools/run_official_baselines.py \
  --repo-root . \
  --manifest-root manifests/pilot_v1 \
  --out artifacts/baselines \
  --prepare-only
```

A method-specific conversion failure is an error. Never replace only the failed baseline sample while keeping a different ID for DM-ICF.

## 9. Versioning

A complete data version consists of:

```text
RCSB query JSON
+ download manifests/failures/checksums
+ screening eligible/rejected tables
+ MMseqs + Infernal/Rfam logs and versions
+ annotated candidate tables
+ frozen manifests and SHA256 metadata
+ data-audit outputs
```

Changing a screening threshold, Rfam release, clustering universe or split seed creates `pilot_v2` rather than mutating `pilot_v1`.
