"""Audited manifest freezing with strict prior validation and complex-view separation.

The strict complex splitter/purge lives in ``manifest_legacy.py``. This wrapper adds:

1. P30-disjoint Protein prior validation;
2. RNA validation disjoint under R80 OR Rfam connected components;
3. final-test exact/family purge before prior sampling;
4. mandatory removal of RNA extracted-chain views sourced from the frozen 1,100
   downstream complex pool.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pr_pilot.data import manifest_legacy as _legacy
from pr_pilot.data.manifest_legacy import *  # noqa: F401,F403


def _single_components(df: pd.DataFrame, molecule: str) -> list[list[str]]:
    frame = df.reset_index(drop=True)
    uf = _legacy._UnionFind(len(frame))
    last: dict[str, int] = {}
    for index, row in frame.iterrows():
        if molecule == "protein":
            labels = {f"p30:{x}" for x in _legacy._labels(row["protein_cluster_p30"])}
        elif molecule == "rna":
            labels = {f"r80:{x}" for x in _legacy._labels(row["rna_cluster_r80"])}
            labels.update({f"rfam:{x}" for x in _legacy._labels(row["rfam_family"])})
        else:
            raise ValueError("molecule must be 'protein' or 'rna'")
        if not labels:
            raise ValueError(
                f"{molecule} sample {row['sample_id']} has no strict validation label"
            )
        for label in labels:
            if label in last:
                uf.union(index, last[label])
            else:
                last[label] = index
    groups: dict[int, list[str]] = {}
    for index, sample_id in enumerate(frame["sample_id"].astype(str)):
        groups.setdefault(uf.find(index), []).append(sample_id)
    return list(groups.values())


def _purge_frozen_complex_views(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, int]:
    """Remove extracted RNA views whose source complex is in the frozen 1,100.

    If extracted views exist, absence of ``complex_pool.tsv`` is a protocol error:
    prior sampling must occur only after the downstream complex pool is frozen.
    """
    if "source_complex_sample_id" not in df.columns:
        return df, 0
    source = df["source_complex_sample_id"].fillna("").astype(str)
    has_extracted = source.str.len().gt(0).any()
    pool_path = out_dir / "complex_pool.tsv"
    if has_extracted and not pool_path.exists():
        raise RuntimeError(
            "RNA extracted-chain candidates are present but complex_pool.tsv is not frozen yet; "
            "freeze complexes first, then prior pools"
        )
    if not pool_path.exists():
        return df, 0
    frozen = set(pd.read_csv(pool_path, sep="\t")["sample_id"].astype(str))
    remove = source.isin(frozen)
    return df.loc[~remove].copy().reset_index(drop=True), int(remove.sum())


def freeze_single_molecule_pool(
    eligible_path: Path,
    out_dir: Path,
    molecule: str,
    seed: int,
    forbidden_test_path: Path,
    counts: FrozenCounts = FrozenCounts(),  # noqa: F405
) -> dict[str, Path]:
    """Freeze a strict 900/100 prior pool after all downstream-holdout protections."""
    df = pd.read_csv(eligible_path, sep=None, engine="python")
    test = pd.read_csv(forbidden_test_path, sep="\t")
    _legacy._require_columns(test, _legacy.REQUIRED_COMPLEX, "complex_test")
    required = _legacy.REQUIRED_PROTEIN if molecule == "protein" else _legacy.REQUIRED_RNA
    _legacy._require_columns(df, required, molecule)

    df, purge_stats = _legacy._purge_pretraining_against_test(df, molecule, test)
    frozen_complex_view_purged = 0
    if molecule == "rna":
        df, frozen_complex_view_purged = _purge_frozen_complex_views(df, out_dir)

    train_n = counts.protein_train if molecule == "protein" else counts.rna_train
    val_n = counts.protein_val if molecule == "protein" else counts.rna_val
    pool_n = train_n + val_n
    components = _single_components(df, molecule)
    val_ids = _legacy._exact_component_subset(components, val_n, seed + 7001)
    val = df[df["sample_id"].astype(str).isin(val_ids)].copy().reset_index(drop=True)
    remainder = df[~df["sample_id"].astype(str).isin(val_ids)].copy().reset_index(drop=True)
    train = _legacy.deterministic_sample(remainder, train_n, seed + 7003)
    pool = pd.concat([train, val], ignore_index=True)
    if len(pool) != pool_n:
        raise AssertionError("Single-molecule frozen pool size mismatch")

    if molecule == "protein":
        overlap = _legacy._flatten_column(train, "protein_cluster_p30") & _legacy._flatten_column(
            val, "protein_cluster_p30"
        )
        split_rule = "P30 whole-component validation"
    else:
        r80 = _legacy._flatten_column(train, "rna_cluster_r80") & _legacy._flatten_column(
            val, "rna_cluster_r80"
        )
        rfam = _legacy._flatten_column(train, "rfam_family") & _legacy._flatten_column(
            val, "rfam_family"
        )
        overlap = r80 | rfam
        split_rule = "R80-or-Rfam connected-component validation"
    if overlap:
        raise AssertionError(
            f"{molecule} prior train/validation family leakage: {sorted(overlap)[:5]}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, frame in [("pool", pool), ("train", train), ("val", val)]:
        path = out_dir / f"{molecule}_{label}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[label] = path
    _legacy._write_manifest_meta(
        out_dir / f"{molecule}_manifest_meta.json",
        seed,
        paths,
        extra={
            "final_test_purge": purge_stats,
            "frozen_complex_chain_views_purged": frozen_complex_view_purged,
            "forbidden_test_sha256": _legacy.sha256_file(forbidden_test_path),
            "prior_validation_split_rule": split_rule,
            "prior_validation_cluster_disjoint": True,
        },
    )
    return paths


def assert_prior_train_val_disjoint(
    protein_train: pd.DataFrame,
    protein_val: pd.DataFrame,
    rna_train: pd.DataFrame,
    rna_val: pd.DataFrame,
) -> None:
    p = _legacy._flatten_column(protein_train, "protein_cluster_p30") & _legacy._flatten_column(
        protein_val, "protein_cluster_p30"
    )
    r80 = _legacy._flatten_column(rna_train, "rna_cluster_r80") & _legacy._flatten_column(
        rna_val, "rna_cluster_r80"
    )
    rfam = _legacy._flatten_column(rna_train, "rfam_family") & _legacy._flatten_column(
        rna_val, "rfam_family"
    )
    if p or r80 or rfam:
        raise AssertionError(
            f"Prior validation leakage: P30={len(p)} R80={len(r80)} Rfam={len(rfam)}"
        )
