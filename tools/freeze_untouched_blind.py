#!/usr/bin/env python3
"""Screen, annotate, audit, and freeze a new untouched blind complex set.

The blind set is built from local experimental CIF files that are not present
in any prior complex manifest found in the project workspace.  It is selected
without using model scores, holdout metrics, GC, length, or degree.  Cluster
and Rfam labels are computed jointly with the old development and holdout
rows, so P30/R80/Rfam leakage checks use a common label space.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pr_pilot.data.clustering import annotate_all_candidates
from pr_pilot.data.manifest import assert_no_test_leakage
from pr_pilot.data.screening import ScreenConfig, screen_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def known_pdb_ids(roots: list[Path]) -> set[str]:
    ids: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.tsv"):
            if "complex" not in path.name.lower():
                continue
            # Candidate/raw download registries and rejection logs describe
            # structures that were merely considered.  They are not evidence
            # of prior model use and would otherwise exclude the entire raw
            # source pool.  Keep only manifests/eligible/annotated records.
            if path.name in {"complex.tsv", "complex_remaining.tsv", "complex_rejected.tsv"} or "shard" in path.name.lower():
                continue
            try:
                frame = pd.read_csv(path, sep="\t", usecols=["pdb_id"], dtype=str)
            except Exception:
                continue
            ids.update(frame["pdb_id"].dropna().astype(str).str.upper())
    return ids


def screen_one(path_cfg: tuple[str, ScreenConfig]) -> tuple[str, dict | None, str]:
    path_text, cfg = path_cfg
    path = Path(path_text)
    record, reason = screen_file(path, "complex", cfg)
    return path_text, record, reason


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def chain_rows(complexes: pd.DataFrame, polymer: str) -> pd.DataFrame:
    rows = []
    key = "protein_chain_sequences" if polymer == "protein" else "rna_chain_sequences"
    for row in complexes.itertuples(index=False):
        chains = json.loads(str(getattr(row, key)))
        for chain, sequence in sorted(chains.items()):
            rows.append(
                {
                    "sample_id": f"{row.sample_id}::{chain}",
                    "sequence": str(sequence),
                    "sequence_hash": hashlib.sha256(str(sequence).encode()).hexdigest(),
                    "length": len(str(sequence)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rfam-cm-gz", type=Path, required=True)
    parser.add_argument("--rfam-clanin", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--annotation-cpu", type=int, default=12)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    existing_dev = pd.concat(
        [
            pd.read_csv(args.existing_manifest / "complex_train.tsv", sep="\t", dtype=str),
            pd.read_csv(args.existing_manifest / "complex_val.tsv", sep="\t", dtype=str),
        ],
        ignore_index=True,
    )
    existing_test = pd.read_csv(args.existing_manifest / "complex_test.tsv", sep="\t", dtype=str)
    known_ids = set(existing_dev["sample_id"].astype(str)) | set(existing_test["sample_id"].astype(str))
    exclude_pdb = known_pdb_ids(
        [
            args.existing_manifest,
            args.workspace_root / "workflow_20260905" / "data",
            args.workspace_root / "local_incremental_eval_20260911" / "manifests",
        ]
    )

    raw_files = sorted({path for root in args.raw_root for path in root.rglob("*.cif")})
    by_pdb: dict[str, Path] = {}
    for path in raw_files:
        pdb_id = path.name.split("-")[0].split(".")[0].upper()
        if pdb_id in exclude_pdb or pdb_id in by_pdb:
            continue
        by_pdb[pdb_id] = path
    download = pd.DataFrame(
        [{"pdb_id": pdb, "path": str(path), "kind": "complex", "source": "local_raw_cif"} for pdb, path in sorted(by_pdb.items())]
    )
    write_frame(out / "screen_input.tsv", download)

    cfg = ScreenConfig(
        protein_min_length=40,
        protein_max_length=2000,
        rna_min_length=10,
        rna_max_length=500,
        max_total_tokens=1000,
        max_resolution_angstrom=4.0,
        allow_nmr_without_resolution=True,
        apply_resolution_method_filter=False,
        interface_contact_angstrom=6.0,
        min_interfacial_residue_pairs=3,
        max_interface_missing_fraction=0.10,
        exclude_large_rnp_keywords=True,
    )
    screened_dir = out / "screened"
    eligible_path = screened_dir / "complex_eligible.tsv"
    rejected_path = screened_dir / "complex_rejected.tsv"
    if eligible_path.exists() and rejected_path.exists():
        # The screen is deterministic for a fixed raw-root and config. Reuse
        # it after an annotation-only failure instead of parsing 4k CIFs again.
        eligible_frame = pd.read_csv(eligible_path, sep="\t", dtype=str)
        rejected_frame = pd.read_csv(rejected_path, sep="\t", dtype=str)
    else:
        eligible: list[dict] = []
        rejected: list[dict] = []
        jobs = [(str(path), cfg) for path in by_pdb.values()]
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [executor.submit(screen_one, job) for job in jobs]
            for future in as_completed(futures):
                path_text, record, reason = future.result()
                if record is None:
                    rejected.append({"path": path_text, "reason": reason})
                else:
                    eligible.append(record)
        eligible_frame = pd.DataFrame(sorted(eligible, key=lambda row: str(row["sample_id"])))
        rejected_frame = pd.DataFrame(rejected)
        write_frame(eligible_path, eligible_frame)
        write_frame(rejected_path, rejected_frame)
        (screened_dir / "summary.json").write_text(
            json.dumps(
                {
                    "raw_files_considered": len(raw_files),
                    "pdb_candidates_after_prior_manifest_exclusion": len(download),
                    "eligible": len(eligible_frame),
                    "rejected": len(rejected_frame),
                    "config": cfg.__dict__,
                    "excluded_pdb_count": len(exclude_pdb),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if eligible_frame.empty:
        raise RuntimeError("No eligible blind candidates after structural screening")

    known = pd.concat([existing_dev, existing_test], ignore_index=True)
    combined = pd.concat([known, eligible_frame], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
    annotation_input = out / "annotation_input"
    write_frame(annotation_input / "complex.tsv", combined)
    write_frame(annotation_input / "protein.tsv", chain_rows(combined, "protein"))
    write_frame(annotation_input / "rna.tsv", chain_rows(combined, "rna"))
    annotated_dir = out / "annotated"
    annotated_complex_path = annotated_dir / "complex_annotated.tsv"
    if not annotated_complex_path.exists():
        _, _, annotated_complex_path = annotate_all_candidates(
            annotation_input / "protein.tsv",
            annotation_input / "rna.tsv",
            annotation_input / "complex.tsv",
        annotated_dir,
        rfam_cm_gz=args.rfam_cm_gz,
        rfam_clanin=args.rfam_clanin,
        cmscan_cpu=int(args.annotation_cpu),
        rfam_query_sample_ids=set(eligible_frame["sample_id"].astype(str)),
    )
    annotated = pd.read_csv(annotated_complex_path, sep="\t", dtype=str)
    blind_candidates = annotated[~annotated["sample_id"].isin(known_ids)].copy()
    ref_dev = annotated[annotated["sample_id"].isin(set(existing_dev["sample_id"]))].copy()
    ref_test = annotated[annotated["sample_id"].isin(set(existing_test["sample_id"]))].copy()
    if len(ref_dev) != len(existing_dev) or len(ref_test) != len(existing_test):
        raise RuntimeError("Combined annotation did not retain every existing reference row")

    # Audit every candidate first; selection below uses only stable sample ID
    # hashing and the requested count, never model or holdout metrics.
    safe_rows: list[dict] = []
    rejected_leakage: list[dict] = []
    for row in blind_candidates.to_dict(orient="records"):
        one = pd.DataFrame([row])
        try:
            assert_no_test_leakage(ref_dev, ref_test, one, strict_cluster_check=True)
        except AssertionError as exc:
            rejected_leakage.append({"sample_id": row["sample_id"], "reason": str(exc)})
        else:
            safe_rows.append(row)
    safe = pd.DataFrame(safe_rows)
    if safe.empty:
        # Lock an explicit empty blind manifest rather than silently relaxing
        # the leakage contract or promoting a contaminated candidate.
        safe = pd.DataFrame(columns=annotated.columns)
    safe["selection_hash"] = safe["sample_id"].map(lambda x: hashlib.sha256(f"{args.seed}|{x}".encode()).hexdigest())
    safe = safe.sort_values("selection_hash", kind="stable").reset_index(drop=True)
    selected = safe.head(min(int(args.target), len(safe))).copy()
    selected["blind_selection_rank"] = range(1, len(selected) + 1)
    selected["mother_sample_id"] = selected["sample_id"]
    blind_dir = out / "blind_manifest"
    blind_dir.mkdir(parents=True, exist_ok=True)
    write_frame(blind_dir / "complex_test.tsv", selected)
    write_frame(blind_dir / "complex_pool.tsv", selected)
    write_frame(blind_dir / "complex_dev.tsv", selected.iloc[:0].copy())
    write_frame(blind_dir / "complex_train.tsv", selected.iloc[:0].copy())
    write_frame(blind_dir / "complex_val.tsv", selected.iloc[:0].copy())
    (out / "leakage_audit.json").write_text(
        json.dumps(
            {
                "reference_development_rows": len(ref_dev),
                "reference_old_holdout_rows": len(ref_test),
                "candidate_rows_after_annotation": len(blind_candidates),
                "p30_r80_rfam_rejected": len(rejected_leakage),
                "p30_r80_rfam_clean": len(safe),
                "selected": len(selected),
                "rejected_examples": rejected_leakage[:20],
                "checks": ["sample_id", "mother_sample_id", "protein_hash", "rna_hash", "protein_cluster_p30", "rna_cluster_r80", "rfam_family"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files = {
        "complex_test.tsv": sha256_file(blind_dir / "complex_test.tsv"),
        "complex_pool.tsv": sha256_file(blind_dir / "complex_pool.tsv"),
    }
    (blind_dir / "manifest_meta.json").write_text(
        json.dumps(
            {
                "seed": int(args.seed),
                "source_raw_roots": [str(root) for root in args.raw_root],
                "source_raw_root_file_count": len(raw_files),
                "structure_filters": cfg.__dict__,
                "selection_rule": "stable SHA-256 sample_id order after P30/R80/Rfam audit; no metrics used",
                "target_requested": int(args.target),
                "selected_count": len(selected),
                "files": files,
                "blind_locked_before_training": True,
                "metrics_read": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"raw": len(raw_files), "screen_candidates": len(download), "eligible": len(eligible_frame), "clean": len(safe), "selected": len(selected), "blind_dir": str(blind_dir)}, indent=2))


if __name__ == "__main__":
    main()
