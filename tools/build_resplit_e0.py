#!/usr/bin/env python3
"""Jointly annotate the screened raw pool and freeze a retrospective E0 split.

The candidate pool is screened before this script and contains only structures
that satisfy the locked length/quality/contact protocol.  Historical
development and holdout rows are included only to put every sequence in one
fresh P30/R80/Rfam label space; only the screened candidates are assigned to
the new development or blind manifests.  No model metrics are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from freeze_untouched_blind import chain_rows  # noqa: E402
from pr_pilot.data.clustering import annotate_all_candidates  # noqa: E402
from pr_pilot.data.manifest import assert_no_test_leakage, bilateral_components, sha256_file  # noqa: E402


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)


def stable_key(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()


def choose_component_subset(components: list[list[str]], candidate_ids: set[str], target: int, seed: int) -> set[str]:
    """Choose a deterministic component union closest to target candidate rows."""
    records = []
    for component in components:
        ids = sorted(set(component) & candidate_ids)
        if ids:
            records.append((stable_key(seed, "|".join(sorted(component))), ids))
    records.sort(key=lambda item: item[0])

    # 0/1 subset-sum over candidate counts.  1,075 rows keeps this small and
    # guarantees exact target whenever the component sizes permit it.
    states: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, ids) in enumerate(records):
        size = len(ids)
        for total, choice in list(states.items()):
            new_total = total + size
            if new_total not in states:
                states[new_total] = choice + (index,)
    possible = sorted(states, key=lambda n: (abs(n - target), n > target, n))
    selected_total = possible[0]
    selected: set[str] = set()
    for index in states[selected_total]:
        selected.update(records[index][1])
    if not selected:
        raise RuntimeError("component split selected no blind candidates")
    return selected


def split_dev_val(dev: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/val manifest views without breaking components.

    The Adapter runner treats train+val as the development universe and makes
    its own grouped CV folds.  These views exist for cache compatibility and
    are themselves component-disjoint for auditability.
    """
    components = bilateral_components(dev)
    target = max(1, round(len(dev) * 0.2))
    val_ids = choose_component_subset(components, set(dev["sample_id"].astype(str)), target, seed)
    val = dev[dev["sample_id"].astype(str).isin(val_ids)].copy().reset_index(drop=True)
    train = dev[~dev["sample_id"].astype(str).isin(val_ids)].copy().reset_index(drop=True)
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rfam-cm", type=Path, required=True)
    parser.add_argument("--rfam-clanin", type=Path, required=True)
    parser.add_argument("--tool-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--annotation-cpu", type=int, default=12)
    args = parser.parse_args()

    # The local wrappers use WSL-backed mmseqs/cmscan/cmpress.  Put them first
    # so this run cannot accidentally resolve a different installation.
    os.environ["PATH"] = str(args.tool_dir) + os.pathsep + os.environ.get("PATH", "")
    for tool in ("mmseqs", "cmscan", "cmpress"):
        import shutil
        if shutil.which(tool) is None:
            raise RuntimeError(f"missing required local wrapper: {tool}")

    old_train = pd.read_csv(args.old_manifest / "complex_train.tsv", sep="\t", dtype=str)
    old_val = pd.read_csv(args.old_manifest / "complex_val.tsv", sep="\t", dtype=str)
    old_test = pd.read_csv(args.old_manifest / "complex_test.tsv", sep="\t", dtype=str)
    old = pd.concat([old_train, old_val, old_test], ignore_index=True)
    eligible = pd.read_csv(args.eligible, sep="\t", dtype=str)
    candidate_ids = set(eligible["sample_id"].astype(str))
    if len(candidate_ids) != len(eligible):
        raise ValueError("eligible candidate sample_id values are not unique")

    combined = pd.concat([eligible, old], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
    annotation_input = args.out / "annotation_input"
    write_frame(annotation_input / "complex.tsv", combined)
    write_frame(annotation_input / "protein.tsv", chain_rows(combined, "protein"))
    write_frame(annotation_input / "rna.tsv", chain_rows(combined, "rna"))

    annotated_dir = args.out / "annotated"
    annotated_path = annotated_dir / "complex_annotated.tsv"
    if not annotated_path.exists():
        _, _, annotated_path = annotate_all_candidates(
            annotation_input / "protein.tsv",
            annotation_input / "rna.tsv",
            annotation_input / "complex.tsv",
            annotated_dir,
            rfam_cm_gz=args.rfam_cm,
            rfam_clanin=args.rfam_clanin,
            cmscan_cpu=args.annotation_cpu,
            rfam_query_sample_ids=None,
        )
    annotated = pd.read_csv(annotated_path, sep="\t", dtype=str)
    if set(annotated["sample_id"].astype(str)) != set(combined["sample_id"].astype(str)):
        raise RuntimeError("joint annotation lost or added sample IDs")

    components = bilateral_components(annotated)
    blind_ids = choose_component_subset(components, candidate_ids, round(len(candidate_ids) * 0.20), args.seed)
    dev_ids = candidate_ids - blind_ids
    dev = annotated[annotated["sample_id"].astype(str).isin(dev_ids)].copy().reset_index(drop=True)
    blind = annotated[annotated["sample_id"].astype(str).isin(blind_ids)].copy().reset_index(drop=True)
    train, val = split_dev_val(dev, args.seed + 1)

    manifest = args.out / "manifest"
    write_frame(manifest / "complex_pool.tsv", pd.concat([dev, blind], ignore_index=True))
    write_frame(manifest / "complex_dev.tsv", dev)
    write_frame(manifest / "complex_train.tsv", train)
    write_frame(manifest / "complex_val.tsv", val)
    write_frame(manifest / "complex_test.tsv", blind)

    # Explicitly verify the cross-partition hash/P30/R80/Rfam contract.  The
    # component construction should already imply this, but the independent
    # assertion is retained as an auditable safety check.
    assert_no_test_leakage(dev, pd.DataFrame(columns=dev.columns), blind, strict_cluster_check=True)

    meta = {
        "protocol": "retrospective_resplit_e0",
        "retrospective_resplit": True,
        "historical_data_may_be_reassigned": True,
        "blind_locked_before_new_training": True,
        "metrics_read": False,
        "seed": args.seed,
        "candidate_count": len(candidate_ids),
        "development_count": len(dev),
        "blind_count": len(blind),
        "requested_blind_fraction": 0.20,
        "actual_blind_fraction": len(blind) / max(len(candidate_ids), 1),
        "joint_annotation_rows": len(annotated),
        "grouping": "fresh bilateral connected components over Protein P30, RNA R80, and Rfam",
        "length_protocol": {"protein": [40, 2000], "rna": [10, 500]},
        "resolution_method_filter": "disabled for this round; resolution/method retained for stratified reporting",
        "source_eligible_sha256": sha256_file(args.eligible),
        "files": {name: sha256_file(path) for name, path in {
            "complex_pool.tsv": manifest / "complex_pool.tsv",
            "complex_dev.tsv": manifest / "complex_dev.tsv",
            "complex_train.tsv": manifest / "complex_train.tsv",
            "complex_val.tsv": manifest / "complex_val.tsv",
            "complex_test.tsv": manifest / "complex_test.tsv",
        }.items()},
        "old_reference_counts": {"train": len(old_train), "val": len(old_val), "test": len(old_test)},
    }
    (manifest / "manifest_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "resplit_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
