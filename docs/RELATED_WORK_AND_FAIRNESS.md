# Related work and comparison contracts

**Primary-source check date:** 2026-09-05. This is a task/input comparison, not a
meta-analysis of incompatible published scores. Links below are evidence, not
confirmation that their software or weights ran in this review environment.

## 1. Position the contribution narrowly and accurately

| Work | Relevant capability from its own paper | Implication for this project |
| --- | --- | --- |
| ProteinMPNN [1] | fixed-backbone protein sequence design | a strong protein structural reference, not a partner-identity control by itself |
| LigandMPNN [2] | protein sequence design conditioned on nonprotein atomic context, including nucleotides | include a protein-target context-aware comparison; document any extra native atomic information |
| gRNAde [3] | geometric RNA inverse design, including multi-conformer inputs | compare the RNA prior; separate single- and multi-conformer regimes and atom views |
| RhoDesign [4] | structure-based RNA sequence design with aptamer-related functional validation | recovery alone is not the full standard for RNA design quality |
| NA-MPNN [5] | unified protein/DNA/RNA graph; its design model explicitly demonstrates protein-conditioned RNA inverse folding | the pilot's partner-hidden wrapper is NOT the full capability of NA-MPNN; a conditional RNA comparison is needed |
| RNAFlow [6] | protein-conditioned RNA sequence AND structure generation | relevant landscape, but not identical to fixed-dual-backbone sequence design |
| MMDiff [7] | joint protein/nucleic-acid sequence/structure generation | do not claim to be the first joint multimolecular generator |
| RFDpoly [8] | general biopolymer backbone generation with downstream design/validation | a complementary upstream geometry problem, not an interchangeable fixed-backbone baseline |

For gRNAde specifically, the paper's three-bead representation uses P, C4' and a
glycosidic nitrogen coordinate (N1/N9); this differs from the proposed sugar/
phosphate-only view. A coordinate at that position is not the same thing as
supplying a base identity label. Audit what each model actually receives rather
than automatically accusing another representation of leakage. Keep published
settings and a harmonized-input sensitivity study distinct.

NA-MPNN's design and DNA-specificity models have different supervision. Do not use
a specificity model trained against PPMs as an undeclared substitute for RNA
sequence design. Its Figure 2a and evaluation section retain protein as context
while designing RNA. Consequently, "NA-MPNN does not receive protein identity"
is true of this pilot's selected blind route, not a general statement about the
method. Its biochemical/structure validation also raises the evidentiary standard;
an unrun pilot cannot currently claim superiority.

## 2. Three comparisons, three different questions

### Track A: controlled small-data structural-prior comparison

Train the project's priors and official single-molecule references from random
initialization on the same frozen single-molecule IDs, with development selection
and full-pool refit explicitly documented. This asks whether the prior implementation
is competitive at the pilot data scale. It does not measure the best practical
performance of a large published checkpoint. Report parameter counts, usable
sample intersections and optimization opportunities.

### Track B: partner-aware conditional comparison

For RNA-known protein design, compare to LigandMPNN where a valid context adapter
can be constructed. For protein-known RNA design, compare to the design variant
of NA-MPNN WITH protein context. Record target atom view, partner atom view, partner
sequence access, chain fixation, training data and candidate budgets in a table
for every run. An atomic-context baseline may have information that the primary
sequence-neutral model does not. Report both the practical full-input route and,
when technically sound, a restricted-input sensitivity comparison. Do not hide
the advantage or silently alter an official model's semantics.

A fair external conditional benchmark is not a substitute for internal controls:
those are needed to isolate data, capacity and the explicit C/DeltaC/alpha factorization.

### Track C: practical published-weight comparison

Use pinned published checkpoints in a separately labeled track. Disclose known or
unknown pretraining overlap, intended/usable target pools, hardware and inference
budgets. A model trained on thousands more structures and a random-init 1000-sample
pilot do not have the same data opportunity. Conversely, weak small-data baselines
do not establish a state-of-the-art claim against those published systems.

The complete primary baseline software integration remains UNVERIFIED in this
review. No new official wrapper is claimed to be tested without executing its
pinned upstream code, preprocessing, checkpoint and output mapping.

## 3. What "excellent and complete" should mean here

The explicit decomposition and testable partner dependence can be a worthwhile
contribution even when a larger model is more accurate. But excellence requires
measured results on held-out families, correct probabilistic/metric semantics,
fair strong baselines, paired sequence/structure behavior, and a reproducible
release. A large experiment registry and many implementation comments do not
establish those outcomes. Interpretability must survive main-effects controls,
residual drift analysis and predictive held-out intervention tests.

Do not import published recovery percentages as directly comparable results:
datasets, atom views, templates, training corpora, temperatures and reporting units
differ. The literature here motivates comparison design, not a fabricated scorecard.

## 4. Primary sources

[1] Dauparas et al. Robust deep learning-based protein sequence design using
ProteinMPNN. Science (2022). DOI: 10.1126/science.add2187.
https://www.science.org/doi/10.1126/science.add2187

[2] Dauparas et al. Atomic context-conditioned protein sequence design using
LigandMPNN. Nature Methods (2025). DOI: 10.1038/s41592-025-02626-1.
https://www.nature.com/articles/s41592-025-02626-1

[3] Joshi et al. gRNAde: Geometric Deep Learning for 3D RNA inverse design.
ICLR (2025), official proceedings.
https://proceedings.iclr.cc/paper_files/paper/2025/file/1b96f01343ff10150e6719eb163e1536-Paper-Conference.pdf

[4] Wong et al. Deep generative design of RNA aptamers using structural predictions.
Nature Computational Science (2024). DOI: 10.1038/s43588-024-00720-6.
https://www.nature.com/articles/s43588-024-00720-6

[5] Kubaney et al. RNA sequence design and protein-DNA specificity prediction with
NA-MPNN. bioRxiv, version 2 consulted. DOI: 10.1101/2025.10.03.679414.
https://www.biorxiv.org/content/10.1101/2025.10.03.679414v2.full

[6] Nori and Jin. RNAFlow: RNA Structure & Sequence Design via Inverse Folding-Based
Flow Matching. ICML/PMLR 235 (2024).
https://proceedings.mlr.press/v235/nori24a.html

[7] Morehead et al. Towards Joint Sequence-Structure Generation of Nucleic Acid
and Protein Complexes with SE(3)-Discrete Diffusion. arXiv:2401.06151 (2024).
https://arxiv.org/abs/2401.06151

[8] RFDpoly primary preprint, DOI: 10.1101/2025.10.01.679929.
https://www.biorxiv.org/content/10.1101/2025.10.01.679929v1
Author-maintained documentation: https://rosettacommons.github.io/RFDpoly/

## 5. Bibliography maintenance limits

Freeze the exact manuscript version, venue, DOI and software checkpoint separately.
A retrieved preprint does not prove that a cited journal version does not exist.
In particular, this review did not independently resolve the final publisher
metadata for the working PDF's 2026 Townley reference; retain that as a bibliography
check, not a reason to silently rewrite it. New papers after the sources above may
change the comparison landscape; the list is a reviewed set, not a claim of an
exhaustive up-to-the-minute survey of every method.
