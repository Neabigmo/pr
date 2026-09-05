"""Manifest freezing, deterministic sampling and leakage guards.

Scientific invariants
---------------------
1. The final 100-complex holdout is frozen *before* the 1,000-complex
   development set and before both single-molecule pretraining pools.
2. Protein P30 and RNA Rfam/R80 overlap is checked at the *constituent chain*
   level. Semicolon-joined multi-chain labels are never treated as one opaque
   string.
3. Test homologues/families are purged from protein/RNA structural-prior pools.
4. All randomisation is stable hash-based and independent of dataframe row order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import pandas as pd


REQUIRED_COMMON = {"sample_id", "structure_path", "sequence", "sequence_hash"}
REQUIRED_PROTEIN = REQUIRED_COMMON | {"protein_cluster_p30"}
REQUIRED_RNA = REQUIRED_COMMON | {"rna_cluster_r80", "rfam_family"}
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
UNKNOWN_LABELS = {"", "nan", "none", "unknown", "na", "n/a", ".", "?"}


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


def _labels(value: object) -> set[str]:
    """Parse a semicolon-joined label field into non-empty constituent labels."""
    if value is None:
        return set()
    text = str(value).strip()
    if text.lower() in UNKNOWN_LABELS:
        return set()
    return {
        part.strip()
        for part in text.split(";")
        if part.strip() and part.strip().lower() not in UNKNOWN_LABELS
    }


def _rna_holdout_labels(row: pd.Series) -> set[str]:
    """Return every known Rfam AND every R80 chain cluster.

    We deliberately use the union rather than preferring Rfam. A pair of complexes
    sharing either an Rfam family or an R80 sequence cluster belongs to one connected
    component, so the subsequent audit cannot discover an overlap the splitter itself
    ignored.
    """
    labels = {f"rfam:{x}" for x in _labels(row.get("rfam_family", ""))}
    labels.update({f"r80:{x}" for x in _labels(row.get("rna_cluster_r80", ""))})
    return labels


def normalize_sequence(seq: str, alphabet: str) -> str:
    seq = str(seq).upper().replace(" ", "")
    if alphabet == "rna":
        seq = seq.replace("T", "U")
        valid = set("AUGC")
    elif alphabet == "protein":
        valid = set("ACDEFGHIKLMNPQRSTVWY")
    else:
        raise ValueError(f"Unknown alphabet {alphabet!r}")
    bad = sorted(set(seq) - valid)
    if bad:
        raise ValueError(f"Non-canonical symbols for {alphabet}: {bad}")
    return seq


def deterministic_sample(df: pd.DataFrame, n: int, seed: int, key: str = "sample_id") -> pd.DataFrame:
    """Stable pseudo-random sample independent of input row order."""
    if key not in df.columns:
        raise ValueError(f"Sampling key {key!r} is missing")
    if len(df) < n:
        raise ValueError(f"Requested {n} rows but only {len(df)} eligible rows exist")
    if df[key].astype(str).duplicated().any():
        raise ValueError(f"Sampling key {key!r} must be unique")
    ranked = df.copy()
    ranked["__pilot_rank"] = ranked[key].astype(str).map(lambda x: sha256_text(f"{seed}|{x}"))
    ranked = ranked.sort_values(["__pilot_rank", key], kind="mergesort").head(n)
    return ranked.drop(columns=["__pilot_rank"]).reset_index(drop=True)


def split_deterministic(
    df: pd.DataFrame,
    n_train: int,
    n_val: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_train + n_val != len(df):
        raise ValueError("Split counts must exactly consume the frozen pool")
    ranked = deterministic_sample(df, len(df), seed=seed)
    return (
        ranked.iloc[:n_train].copy().reset_index(drop=True),
        ranked.iloc[n_train:].copy().reset_index(drop=True),
    )


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def bilateral_components(df: pd.DataFrame) -> list[list[str]]:
    """Connected components under any shared P30, Rfam or R80 label.

    This catches transitive leakage and multi-chain partial overlap. Example:
    ``P30_A;P30_B`` and ``P30_A`` belong to the same component. Likewise two RNA
    chains with different Rfam labels but the same R80 cluster cannot be separated.
    """
    _require_columns(df, REQUIRED_COMPLEX, "complex")
    frame = df.reset_index(drop=True).copy()
    uf = _UnionFind(len(frame))
    last_p: dict[str, int] = {}
    last_r: dict[str, int] = {}
    for i, row in frame.iterrows():
        p_labels = _labels(row["protein_cluster_p30"])
        r_labels = _rna_holdout_labels(row)
        if not p_labels:
            raise ValueError(f"Complex {row['sample_id']} has no usable P30 label")
        if not r_labels:
            raise ValueError(f"Complex {row['sample_id']} has neither usable Rfam nor R80 label")
        for label in p_labels:
            if label in last_p:
                uf.union(i, last_p[label])
            else:
                last_p[label] = i
        for label in r_labels:
            if label in last_r:
                uf.union(i, last_r[label])
            else:
                last_r[label] = i

    groups: dict[int, list[str]] = {}
    for i, sid in enumerate(frame["sample_id"].astype(str)):
        groups.setdefault(uf.find(i), []).append(sid)
    return list(groups.values())


def _exact_component_subset(components: list[list[str]], target_n: int, seed: int) -> set[str]:
    """Choose whole bilateral components totalling exactly ``target_n`` rows."""
    ordered = sorted(
        components,
        key=lambda comp: sha256_text(f"{seed}|{'|'.join(sorted(comp))}"),
    )
    dp: dict[int, tuple[int, ...]] = {0: ()}
    for ci, comp in enumerate(ordered):
        size = len(comp)
        for total, choice in list(dp.items()):
            new_total = total + size
            if new_total <= target_n and new_total not in dp:
                dp[new_total] = choice + (ci,)
        if target_n in dp:
            break
    if target_n not in dp:
        sizes = sorted(len(c) for c in components)
        raise RuntimeError(
            f"Cannot form an exact strict component holdout of {target_n}. "
            f"Component sizes={sizes[:40]}{'...' if len(sizes) > 40 else ''}. "
            "Enlarge the eligible candidate set or change the frozen seed; never split a component silently."
        )
    selected: set[str] = set()
    for ci in dp[target_n]:
        selected.update(ordered[ci])
    if len(selected) != target_n:
        raise AssertionError("Strict component selection size mismatch")
    return selected


def bilateral_group_split(
    df: pd.DataFrame,
    n_holdout: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return remainder and a strict bilateral holdout of exactly ``n_holdout``."""
    hold_ids = _exact_component_subset(bilateral_components(df), n_holdout, seed)
    hold = df[df["sample_id"].astype(str).isin(hold_ids)].copy().reset_index(drop=True)
    remain = df[~df["sample_id"].astype(str).isin(hold_ids)].copy().reset_index(drop=True)
    assert_no_test_leakage(
        remain,
        pd.DataFrame(columns=remain.columns),
        hold,
        strict_cluster_check=True,
    )
    return remain, hold


def _flatten_column(df: pd.DataFrame, column: str) -> set[str]:
    result: set[str] = set()
    for value in df[column]:
        result.update(_labels(value))
    return result


def _purge_pretraining_against_test(
    df: pd.DataFrame,
    molecule: str,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)
    if molecule == "protein":
        _require_columns(df, REQUIRED_PROTEIN, molecule)
        forbidden_hash = set(test["protein_hash"].astype(str))
        forbidden_p30 = _flatten_column(test, "protein_cluster_p30")
        keep = ~df["sequence_hash"].astype(str).isin(forbidden_hash)
        keep &= ~df["protein_cluster_p30"].astype(str).map(
            lambda x: bool(_labels(x) & forbidden_p30)
        )
    elif molecule == "rna":
        _require_columns(df, REQUIRED_RNA, molecule)
        forbidden_hash = set(test["rna_hash"].astype(str))
        forbidden_r80 = _flatten_column(test, "rna_cluster_r80")
        forbidden_rfam = _flatten_column(test, "rfam_family")
        keep = ~df["sequence_hash"].astype(str).isin(forbidden_hash)
        keep &= ~df["rna_cluster_r80"].astype(str).map(
            lambda x: bool(_labels(x) & forbidden_r80)
        )
        if forbidden_rfam:
            keep &= ~df["rfam_family"].astype(str).map(
                lambda x: bool(_labels(x) & forbidden_rfam)
            )
    else:
        raise ValueError("molecule must be 'protein' or 'rna'")
    out = df[keep].copy().reset_index(drop=True)
    return out, {"before": before, "after": len(out), "purged": before - len(out)}


def freeze_single_molecule_pool(
    eligible_path: Path,
    out_dir: Path,
    molecule: str,
    seed: int,
    forbidden_test_path: Path,
    counts: FrozenCounts = FrozenCounts(),
) -> dict[str, Path]:
    """Freeze a 1,000-structure prior pool after final-test purge."""
    df = pd.read_csv(eligible_path, sep=None, engine="python")
    test = pd.read_csv(forbidden_test_path, sep="\t")
    _require_columns(test, REQUIRED_COMPLEX, "complex_test")
    required = REQUIRED_PROTEIN if molecule == "protein" else REQUIRED_RNA
    _require_columns(df, required, molecule)
    df, purge_stats = _purge_pretraining_against_test(df, molecule, test)
    pool_n = counts.protein_pool if molecule == "protein" else counts.rna_pool
    train_n = counts.protein_train if molecule == "protein" else counts.rna_train
    val_n = counts.protein_val if molecule == "protein" else counts.rna_val
    pool = deterministic_sample(df, pool_n, seed)
    train, val = split_deterministic(pool, train_n, val_n, seed + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for label, frame in [("pool", pool), ("train", train), ("val", val)]:
        path = out_dir / f"{molecule}_{label}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[label] = path
    _write_manifest_meta(
        out_dir / f"{molecule}_manifest_meta.json",
        seed,
        paths,
        extra={
            "final_test_purge": purge_stats,
            "forbidden_test_sha256": sha256_file(forbidden_test_path),
        },
    )
    return paths


def freeze_complex_pool(
    eligible_path: Path,
    out_dir: Path,
    seed: int,
    counts: FrozenCounts = FrozenCounts(),
    require_strict_bilateral: bool = True,
    strict_validation: bool = True,
) -> dict[str, Path]:
    """Freeze strict test first, then sample a non-overlapping 1,000-complex dev set."""
    df = pd.read_csv(eligible_path, sep=None, engine="python")
    _require_columns(df, REQUIRED_COMPLEX, "complex")
    if not df["experimental"].astype(bool).all():
        raise ValueError("Pilot complex pool is experimental-only; predicted structures are forbidden")
    if df["sample_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate sample_id values must be resolved before freezing")
    if df["mother_sample_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate mother_sample_id values must be grouped before freezing")

    relaxations: list[dict] = []
    if require_strict_bilateral:
        remainder, test = bilateral_group_split(df, counts.complex_test, seed + 17)
        dev = deterministic_sample(remainder, counts.complex_dev, seed + 19)
    else:
        pool_random = deterministic_sample(df, counts.complex_pool, seed)
        ranked = deterministic_sample(pool_random, len(pool_random), seed + 17)
        test = ranked.iloc[: counts.complex_test].copy().reset_index(drop=True)
        dev = ranked.iloc[counts.complex_test :].copy().reset_index(drop=True)
        relaxations.append({"rule": "strict_bilateral", "reason": "explicitly_disabled"})

    if len(test) != counts.complex_test or len(dev) != counts.complex_dev:
        raise AssertionError("Frozen complex counts are inconsistent")

    if strict_validation:
        train, val = bilateral_group_split(dev, counts.complex_val, seed + 23)
        if len(train) != counts.complex_train:
            raise AssertionError("Strict train/validation split size mismatch")
    else:
        train, val = split_deterministic(
            dev,
            counts.complex_train,
            counts.complex_val,
            seed + 23,
        )

    assert_no_test_leakage(
        train,
        val,
        test,
        strict_cluster_check=require_strict_bilateral,
    )
    pool = pd.concat([dev, test], ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {"pool": pool, "dev": dev, "train": train, "val": val, "test": test}
    paths: dict[str, Path] = {}
    for label, frame in frames.items():
        path = out_dir / f"complex_{label}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[label] = path
    _write_manifest_meta(
        out_dir / "complex_manifest_meta.json",
        seed,
        paths,
        extra={"relaxations": relaxations, "strict_validation": strict_validation},
    )
    return paths


def assert_no_test_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    strict_cluster_check: bool = True,
) -> None:
    dev = pd.concat([train, val], ignore_index=True)
    for key in ["sample_id", "mother_sample_id", "protein_hash", "rna_hash"]:
        if key not in dev or key not in test:
            continue
        overlap = set(dev[key].astype(str)) & set(test[key].astype(str))
        if overlap:
            raise AssertionError(f"TEST LEAKAGE via {key}: {sorted(overlap)[:5]}")
    if strict_cluster_check and len(dev):
        p_overlap = _flatten_column(dev, "protein_cluster_p30") & _flatten_column(
            test,
            "protein_cluster_p30",
        )
        if p_overlap:
            raise AssertionError(f"P30 leakage into final test: {sorted(p_overlap)[:5]}")
        dev_rfam = _flatten_column(dev, "rfam_family")
        test_rfam = _flatten_column(test, "rfam_family")
        if test_rfam:
            r_overlap = dev_rfam & test_rfam
            if r_overlap:
                raise AssertionError(f"Rfam leakage into final test: {sorted(r_overlap)[:5]}")
        r80_overlap = _flatten_column(dev, "rna_cluster_r80") & _flatten_column(
            test,
            "rna_cluster_r80",
        )
        if r80_overlap:
            raise AssertionError(f"R80 leakage into final test: {sorted(r80_overlap)[:5]}")


def assert_pretraining_disjoint(
    protein_pool: pd.DataFrame,
    rna_pool: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Hard audit that final-test exact sequences and families never enter priors."""
    _require_columns(protein_pool, REQUIRED_PROTEIN, "protein_pool")
    _require_columns(rna_pool, REQUIRED_RNA, "rna_pool")
    _require_columns(test, REQUIRED_COMPLEX, "complex_test")
    p_hash = set(protein_pool["sequence_hash"].astype(str)) & set(test["protein_hash"].astype(str))
    p30 = _flatten_column(protein_pool, "protein_cluster_p30") & _flatten_column(
        test,
        "protein_cluster_p30",
    )
    r_hash = set(rna_pool["sequence_hash"].astype(str)) & set(test["rna_hash"].astype(str))
    r80 = _flatten_column(rna_pool, "rna_cluster_r80") & _flatten_column(
        test,
        "rna_cluster_r80",
    )
    rfam = _flatten_column(rna_pool, "rfam_family") & _flatten_column(test, "rfam_family")
    if p_hash or p30 or r_hash or r80 or rfam:
        raise AssertionError(
            "FINAL TEST leaked into structural-prior pools: "
            f"p_hash={len(p_hash)} p30={len(p30)} r_hash={len(r_hash)} "
            f"r80={len(r80)} rfam={len(rfam)}"
        )


def _write_manifest_meta(
    path: Path,
    seed: int,
    paths: dict[str, Path],
    extra: dict | None = None,
) -> None:
    payload = {
        "seed": seed,
        "files": {
            name: {"path": str(file_path), "sha256": sha256_file(file_path)}
            for name, file_path in paths.items()
        },
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
