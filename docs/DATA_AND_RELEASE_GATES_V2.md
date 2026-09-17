# Data, integration and release gates, Review 4

**Status:** proposed acceptance checklist. No box is marked passed merely because
code or a README exists. The offline Review-4 bundle has synthetic tests only.

## Gate A: source and executable environment

Record project Git SHA and dirty-tree status, PDF SHA256, config SHA256, Python and
package versions, hardware, precision, operating system and seeds. Lock actual
upstream baseline SHAs/checkpoint hashes separately from this project. Confirm
third-party licenses and redistribution permissions. The Review-4 installer guards
three known source blobs; this is not certification of all other repository files.

Run the original complete test suite plus Review-4 tests. Compile all code and
run a critical-error linter. Run `tools/check_review4_integration.py`; migrate
legacy Brier calls and probability exports before it can pass. A static passing
scan does not replace execution. Record every command, exit code and stderr.
Do not treat absence of a CI status object as proof of either a passing or a
failing workflow; inspect actual run/job records when access is available.

## Gate B: enough independent real data, not enough database rows

The pilot targets 1000 Protein, 1000 RNA, 1000 development complexes and 100 final
complexes. These are desired counts, not an established supply of independent
usable structures. Build a funnel table: discovered entries -> experimental
assemblies -> valid polymer pairs -> quality-passing mother samples -> unique
biological groups -> bilateral eligible clusters -> final frozen pools. Report
why every exclusion occurs and how many independent groups remain.

A mother sample is a connected protein-RNA interacting subcomplex of a biological
assembly, with exact construct/chain identities. Preserve all conformers,
mutation-series members and close derivatives under the same biological group.
Avoid treating repeated determinations as independent data. Do not silently cut
a large assembly into many pseudoreplicates or relax P30/R80/Rfam to fill a quota.
If the target count is infeasible, revise scope prospectively and retain strictness.

The raw manifest should include at least:

```text
sample_id, mother_id, biological_pair_id, source_accession, source_url,
source_file_sha256, assembly_id, model_id, method, resolution_or_missing,
protein_chain_ids, rna_chain_ids, exact_construct_sequences,
residue_numbering_and_insertion_codes, canonical_parent_mapping,
missing_atom_and_occupancy_summary, total_tokens, quality_tier,
protein_cluster_ids, rna_cluster_ids, rfam_assignments,
clustering_tool_version_and_parameters, split, independence_group_id
```

Missing Rfam is not proof of a new family. Store unresolved/ambiguous mappings and
coverage. Record release/date/hash of annotation resources, including Rfam, instead
of writing only "latest". Modified nucleotides retain raw identity and a documented
canonical-parent mapping; unmappable residues are not automatically ordinary bases.
Report NMR separately from X-ray/cryo-EM; a missing resolution value must not be
interpreted as high resolution. Do not introduce predicted structures into the
experimental-only primary pilot without a new protocol version.

## Gate C: leakage and task definition

Audit exact constructs, every protein/RNA chain in multi-chain samples, protein
low-identity clusters, RNA sequence clusters, Rfam families, biological pairs,
mutation series and conformers. Both sides must pass the strict final holdout.
Test-linked neighbors must also be purged from both single-molecule prior pools.
Development train/validation splitting has its own family-disjoint audit; removing
final-test neighbors alone does not make development validation independent.

Supplement cluster IDs with nearest-training-neighbor identity/coverage and local
alignment diagnostics. Sequence clustering is not a guarantee of structural
novelty; report an RNA structural-similarity challenge analysis when feasible.
Freeze the strict test manifest and its intended denominator before training.
Never swap failed final targets for easier replacements after evaluation.

Maintain two explicitly different interface objects:

* canonical outcome/training labels: source, unperturbed full-heavy-atom 6-A contacts;
* model/design neighborhood: sequence-neutral atom geometry, 8-A cutoff and cap.

Changing graph cap, dropout or coordinate-noise realization must not change
canonical outcome labels. Native-side-chain/base atoms and canonical labels must
not enter structural encoder features or decide the primary SPIR reopen set.
Audit that whole information path, not only the node feature tensor.

## Gate D: integration contracts on tiny REAL structures

Synthetic graph tests supplied here must be followed by a small real-data smoke
covering: one protein, one RNA, a small complex, multi-chain/assembly mapping,
modified residues, insertion codes, missing atoms, no PR edges and singleton PR
neighborhoods. Check chain/residue round trips and baseline output-index maps.
A smoke fixture need not be a final-test target and should not expose final labels.

Required numerical/property checks include:

* proper rigid-rotation and translation invariance of the whole featurizer/model;
  do not impose reflection invariance on chiral molecular geometry;
* permutation equivariance under consistent node/edge reindexing, including output
  restoration to source residue IDs;
* no change from altering unknown own/partner token values, and no native-identity
  atoms or labels entering a geometry-only path;
* canonical-label stability under graph/noise changes; primary SPIR selection
  independent of canonical labels; fixed-site and non-design-site preservation;
* finite outputs/gradients for missing semantic loss groups, empty edges, mixed
  lengths and supported reduced precision;
* unchanged output between cached and uncached inference, including altered known
  masks and partner sequences, real wrappers and all intended control models;
* parameter ownership AND actual update checks through all six stages; frozen
  weights need deterministic eval semantics when used as fixed priors;
* wrong-token corruption must actually be visible to the intended context path.
  Replacing only known=False tokens that are multiplied by zero is a no-op;
* original/random-initialized scratch controls fully trainable from step zero;
* a miniature development-selection -> full-pool refit replay with the original
  schedule horizon and logged progress, learning rates, step counts and unfreezing.

Several items depend on full trainer/runtime files unavailable during this review.
They remain OPEN, even though the isolated model/sampler/statistical tests pass.

## Gate E: official baselines and fair inputs

Execute pinned upstream preprocessing, one short training step, checkpoint save/
load, inference and probability export. Confirm A/U/G/C column mapping with a
non-symmetric known probability vector, not a uniform vector that hides swaps.
Require same intended pool and a predeclared intersection report when parsers
exclude samples. Keep failed-target denominators and reasons. Test frozen positions
and all source-to-output maps. Do not infer successful upstream compatibility from
hand-written CLI strings alone.

A one-sided ProteinMPNN/NA-MPNN route answers a structural-prior question. Add
conditional-information references for partner-aware claims and distinguish
published-weight practical comparisons from small-data random-init comparisons.
No external baseline training or inference was executed in this offline release.

## Gate F: development and final protocol freeze

Store all development search choices and budgets, not only winners. Select stopping
rules without final100. Full-pool refit starts from the specified initialization
and replays the chosen prefix of the original schedule; it must not compress a
long cosine/curriculum into the selected number of epochs. A larger pool can mean
more optimizer steps: record whether schedule is parameterized by epoch, token or
step and preserve the intended semantics explicitly.

Before final evaluation freeze: all checkpoints, seeds, five hypotheses and
component definitions, independence-group roster, effect direction, eligibility,
missingness policy, temperatures, graph/interface definitions, SPIR scope/order,
candidate and predictor budgets, uncertainty metrics, plots and statistical code.
The exact group-balanced primary estimand proposed in Review 4 must be accepted
BEFORE use, not substituted after a complex-level result is unfavorable.

## Gate G: complete evidence and release

Every expected method/seed/target must produce either a valid artifact or a visible
failure record. NaN is not zero; a missing group is not a perfect score; a parser
exception must not silently shrink a benchmark. An incomplete primary roster
fails numeric analysis until the prespecified failure policy has been applied.

A result folder should be immutable and contain:

```text
provenance.json / environment.txt / commands.log
protocol_hashes.json / manifests_and_independence_roster/
training_histories/ / checkpoints_manifest.json
per_token_probabilities/ / per_complex_metrics/ / failures.jsonl
candidates_pre_post_spir/ / external_checks/
confirmatory_effects.csv / confirmatory_results.json
secondary_diagnostics/ / figures_with_input_hashes/
```

A genuine bug discovered after unblinding requires a dated amendment, preserved old
results, a description of affected comparisons and rerunning every affected method,
not just the method helped by the fix. This bundle changes some metric definitions
and SPIR information access; any old numbers from those paths are not directly
comparable without rerunning and labeling versions.
