"""Manifest freezing, deterministic sampling and leakage guards.

This module is intentionally strict. The final 100-complex holdout must be frozen
before any training manifests are consumed. Training code should call
`assert_no_test_leakage` at startup and fail hard on overlap.

The actual upstream dataset builder is project-specific, so this module defines
and validates the normalized TSV/CSV contract rather than pretending to know the
user's local PDB/RNAsolo paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import hashlib
import json
import random

import pandas as pd


REQUIRED_COMMON = {
    "sample_id",
    "structure_path",
    "sequence",
    "sequence_hash",
}

REQUIRED_COMPLEX = {
    "sample_id",
    "structure_path",
    "protein_sequence",
    "rna_sequence",
    "protein_hash",
    "rna_hash",
    "protein_cluster_p30",
    "rna_cluster_r80",
    "rfam_family",
    "mother_sample_id",
    "experimental",
}


@dataclass(frozen=True)
class FrozenCounts:
    protein_pool: int = 1000
    protein_train: int = 900
    protein_val: int = 100
    rna_pool: int = 1000
    rna_train: int = 900
    rna_val: int = 100
    complex_pool: int = 1100
    complex_dev: int = 1000
    complex_test: int = 100
    complex_train: int = 900
    complex_val: int = 100


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} manifest missing required columns: {missing}")


def normalize_sequence(seq: str, alphabet: str) -> str:
    seq = str(seq).upper().replace(" ", "").replace("T", "U" if alphabet == "rna" else "T")
    valid = set("ACDEFGHIKLMNPQRSTVWY") if alphabet == "protein" else set("AUGC")
    bad = sorted(set(seq) - valid)
    if bad:
        raise ValueError(f"Non-canonical symbols for {alphabet}: {bad}")
    return seq


def deterministic_sample(df: pd.DataFrame, n: int, seed: int, key: str = "sample_id") -> pd.DataFrame:
    """Sample without dependence on row order.

    We hash `(seed, sample_id)` and take the smallest hashes. This is preferable
    to `df.sample` because input row order or pandas version changes cannot alter
    the selected pilot set.
    """
    if len(df) < n:
        raise ValueError(f"Requested {n} rows but only {len(df)} eligible rows exist")
    ranked = df.copy()
    ranked["__pilot_rank"] = ranked[key].astype(str).map(lambda x: sha256_text(f"{seed}|{x}"))
    ranked = ranked.sort_values(["__pilot_rank", key], kind="mergesort").head(n)
    return ranked.drop(columns=["__pilot_rank"]).reset_index(drop=True)


def split_deterministic(df: pd.DataFrame, n_train: int, n_val: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_train + n_val != len(df):
        raise ValueError("Split counts must exactly consume the frozen pool")
    ranked = deterministic_sample(df, len(df), seed=seed)
    return ranked.iloc[:n_train].copy(), ranked.iloc[n_train:].copy()


def freeze_single_molecule_pool(
    eligible_path: Path,
    out_dir: Path,
    molecule: str,
    seed: int,
    counts: FrozenCounts = FrozenCounts(),
) -> dict[str, Path]:
    """Freeze protein or RNA 1000-pool and 900/100 dev split."""
    df = pd.read_csv(eligible_path, sep=None, engine="python")
    _require_columns(df, REQUIRED_COMMON, molecule)
    pool_n = counts.protein_pool if molecule == "protein" else counts.rna_pool
    train_n = counts.protein_train if molecule == "protein" else counts.rna_train
    val_n = counts.protein_val if molecule == "protein" else counts.rna_val
    pool = deterministic_sample(df, pool_n, seed)
    train, val = split_deterministic(pool, train_n, val_n, seed + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for label, frame in [("pool", pool), ("train", train), ("val", val)]:
        path = out_dir / f"{molecule}_{label}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[label] = path
    _write_manifest_meta(out_dir / f"{molecule}_manifest_meta.json", seed, paths)
    return paths


def freeze_complex_pool(
    eligible_path: Path,
    out_dir: Path,
    seed: int,
    counts: FrozenCounts = FrozenCounts(),
    require_strict_bilateral: bool = True,
) -> dict[str, Path]:
    """Freeze 1100 experimental complexes, then 100 final holdout and 900/100 dev.

    Expected input should already contain biological-assembly mother samples and
    quality filters. The function refuses predicted structures.
    """
    df = pd.read_csv(eligible_path, sep=None, engine="python")
    _require_columns(df, REQUIRED_COMPLEX, "complex")
    if not df["experimental"].astype(bool).all():
        raise ValueError("Pilot complex pool is experimental-only; predicted structures are forbidden")
    if df["mother_sample_id"].duplicated().any():
        raise ValueError("Duplicate mother_sample_id values must be resolved before freezing")

    pool = deterministic_sample(df, counts.complex_pool, seed)

    # Prefer a test subset whose P30 and Rfam labels are disjoint from the remaining dev pool.
    # A deterministic greedy construction is used so selection can be reproduced exactly.
    candidates = deterministic_sample(pool, len(pool), seed + 17)
    test_rows: list[int] = []
    for idx in candidates.index:
        if len(test_rows) >= counts.complex_test:
            break
        candidate = candidates.loc[idx]
        remaining = pool.drop(index=test_rows + [idx], errors="ignore")
        p_ok = candidate["protein_cluster_p30"] not in set(remaining["protein_cluster_p30"])
        rfam = str(candidate.get("rfam_family", ""))
        r_ok = (not rfam or rfam.lower() in {"nan", "none", "unknown"}) or rfam not in set(remaining["rfam_family"].astype(str))
        if p_ok and r_ok:
            test_rows.append(idx)

    relaxations = []
    if len(test_rows) < counts.complex_test:
        if require_strict_bilateral:
            raise RuntimeError(
                f"Only {len(test_rows)} strict bilateral candidates found; need {counts.complex_test}. "
                "Do not silently relax. Re-run with require_strict_bilateral=False only after documenting it."
            )
        relaxations.append({"rule": "strict_bilateral", "strict_count": len(test_rows)})
        for idx in candidates.index:
            if idx not in test_rows:
                test_rows.append(idx)
            if len(test_rows) == counts.complex_test:
                break

    test = pool.loc[test_rows].copy().reset_index(drop=True)
    dev = pool.drop(index=test_rows).copy().reset_index(drop=True)
    if len(dev) != counts.complex_dev:
        raise AssertionError("Complex dev size mismatch")
    dev_train, dev_val = split_deterministic(dev, counts.complex_train, counts.complex_val, seed + 19)

    assert_no_test_leakage(dev_train, dev_val, test, strict_cluster_check=require_strict_bilateral)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "pool": pool,
        "dev": dev,
        "train": dev_train,
        "val": dev_val,
        "test": test,
    }
    paths: dict[str, Path] = {}
    for label, frame in frames.items():
        path = out_dir / f"complex_{label}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[label] = path
    _write_manifest_meta(out_dir / "complex_manifest_meta.json", seed, paths, extra={"relaxations": relaxations})
    return paths


def assert_no_test_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    strict_cluster_check: bool = True,
) -> None:
    dev = pd.concat([train, val], ignore_index=True)
    for key in ["sample_id", "mother_sample_id", "protein_hash", "rna_hash"]:
        overlap = set(dev[key].astype(str)) & set(test[key].astype(str))
        if overlap:
            raise AssertionError(f"TEST LEAKAGE via {key}: {sorted(list(overlap))[:5]}")
    if strict_cluster_check:
        p_overlap = set(dev["protein_cluster_p30"].astype(str)) & set(test["protein_cluster_p30"].astype(str))
        if p_overlap:
            raise AssertionError(f"P30 leakage into final test: {sorted(list(p_overlap))[:5]}")
        known_test_rfam = {x for x in test["rfam_family"].astype(str) if x.lower() not in {"", "nan", "none", "unknown"}}
        r_overlap = set(dev["rfam_family"].astype(str)) & known_test_rfam
        if r_overlap:
            raise AssertionError(f"Rfam leakage into final test: {sorted(list(r_overlap))[:5]}")


def _write_manifest_meta(path: Path, seed: int, paths: dict[str, Path], extra: dict | None = None) -> None:
    payload = {
        "seed": seed,
        "files": {name: {"path": str(p), "sha256": sha256_file(p)} for name, p in paths.items()},
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
