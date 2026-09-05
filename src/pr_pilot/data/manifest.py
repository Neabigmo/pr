"""Manifest freezing, deterministic sampling and leakage guards.

Scientific invariant: the final complex holdout is frozen *before* the protein
and RNA pretraining pools. Test families are then purged from every pretraining
pool. All selections use stable sample/group identifiers, never transient row
indices.
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
    "sample_id", "structure_path", "protein_sequence", "rna_sequence",
    "protein_hash", "rna_hash", "protein_cluster_p30", "rna_cluster_r80",
    "rfam_family", "mother_sample_id", "experimental",
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


def _known_rfam(value: object) -> bool:
    s = str(value).strip().lower()
    return s not in {"", "nan", "none", "unknown", "na"}


def _rna_holdout_key(row: pd.Series) -> str:
    """Use biological family when known, otherwise sequence cluster."""
    if _known_rfam(row.get("rfam_family", "")):
        return f"rfam:{row['rfam_family']}"
    return f"r80:{row['rna_cluster_r80']}"


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
    """Stable hash sampling independent of input row order or pandas RNG."""
    if len(df) < n:
        raise ValueError(f"Requested {n} rows but only {len(df)} eligible rows exist")
    if df[key].astype(str).duplicated().any():
        raise ValueError(f"Sampling key {key!r} must be unique")
    ranked = df.copy()
    ranked["__pilot_rank"] = ranked[key].astype(str).map(lambda x: sha256_text(f"{seed}|{x}"))
    ranked = ranked.sort_values(["__pilot_rank", key], kind="mergesort").head(n)
    return ranked.drop(columns=["__pilot_rank"]).reset_index(drop=True)


def split_deterministic(df: pd.DataFrame, n_train: int, n_val: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_train + n_val != len(df):
        raise ValueError("Split counts must exactly consume the frozen pool")
    ranked = deterministic_sample(df, len(df), seed=seed)
    return ranked.iloc[:n_train].copy().reset_index(drop=True), ranked.iloc[n_train:].copy().reset_index(drop=True)


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
    """Connected components under shared P30 OR RNA-family/R80 key.

    A whole component must remain in one split. This is stricter and safer than
    greedily testing row indices, and it prevents transitive leakage such as
    P-family A -- RNA-family X -- P-family B across train/test.
    """
    _require_columns(df, REQUIRED_COMPLEX, "complex")
    frame = df.reset_index(drop=True).copy()
    uf = _UnionFind(len(frame))
    last_p: dict[str, int] = {}
    last_r: dict[str, int] = {}
    for i, row in frame.iterrows():
        p = str(row["protein_cluster_p30"])
        r = _rna_holdout_key(row)
        if p in last_p:
            uf.union(i, last_p[p])
        else:
            last_p[p] = i
        if r in last_r:
            uf.union(i, last_r[r])
        else:
            last_r[r] = i
    groups: dict[int, list[str]] = {}
    for i, sid in enumerate(frame["sample_id"].astype(str)):
        groups.setdefault(uf.find(i), []).append(sid)
    return list(groups.values())


def _exact_component_subset(components: list[list[str]], target_n: int, seed: int) -> set[str]:
    """Deterministically choose whole bilateral components totaling target_n.

    Dynamic programming avoids the old bug where a shuffled candidate row index
    was later interpreted as an index into a different dataframe.
    """
    ordered = sorted(
        components,
        key=lambda comp: sha256_text(f"{seed}|{'|'.join(sorted(comp))}"),
    )
    # sum -> component indices. Stop states above target_n.
    dp: dict[int, tuple[int, ...]] = {0: ()}
    for ci, comp in enumerate(ordered):
        size = len(comp)
        for total, choice in list(dp.items())[::-1]:
            new_total = total + size
            if new_total <= target_n and new_total not in dp:
                dp[new_total] = choice + (ci,)
        if target_n in dp:
            break
    if target_n not in dp:
        sizes = sorted(len(c) for c in components)
        raise RuntimeError(
            f"Cannot form an exact bilateral component split of {target_n} complexes. "
            f"Component sizes={sizes[:30]}{'...' if len(sizes) > 30 else ''}. "
            "Change the pilot seed or enlarge the eligible/frozen pool; never split a component."
        )
    selected: set[str] = set()
    for ci in dp[target_n]:
        selected.update(ordered[ci])
    return selected


def bilateral_group_split(df: pd.DataFrame, n_holdout: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return remainder, strict bilateral holdout with exactly n_holdout rows."""
    components = bilateral_components(df)
    hold_ids = _exact_component_subset(components, n_holdout, seed)
    hold = df[df["sample_id"].astype(str).isin(hold_ids)].copy().reset_index(drop=True)
    remain = df[~df["sample_id"].astype(str).isin(hold_ids)].copy().reset_index(drop=True)
    if len(hold) != n_holdout:
        raise AssertionError("Bilateral component split size mismatch")
    assert_no_test_leakage(remain, pd.DataFrame(columns=remain.columns), hold, strict_cluster_check=True)
    return remain, hold


def _purge_pretraining_against_test(df: pd.DataFrame, molecule: str, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)
    if molecule == "protein":
        _require_columns(df, REQUIRED_PROTEIN, molecule)
        forbidden_hash = set(test["protein_hash"].astype(str))
        forbidden_cluster = set(test["protein_cluster_p30"].astype(str))
        keep = (~df["sequence_hash"].astype(str).isin(forbidden_hash)) & (~df["protein_cluster_p30"].astype(str).isin(forbidden_cluster))
    elif molecule == "rna":
        _require_columns(df, REQUIRED_RNA, molecule)
        forbidden_hash = set(test["rna_hash"].astype(str))
        forbidden_r80 = set(test["rna_cluster_r80"].astype(str))
        known_rfam = {str(x) for x in test["rfam_family"] if _known_rfam(x)}
        keep = (~df["sequence_hash"].astype(str).isin(forbidden_hash)) & (~df["rna_cluster_r80"].astype(str).isin(forbidden_r80))
        if known_rfam:
            keep &= ~df["rfam_family"].astype(str).isin(known_rfam)
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
    """Freeze a 1000-pool only after purging all final-test homologues/families."""
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
    paths = {}
    for label, frame in [("pool", pool), ("train", train), ("val", val)]:
        path = out_dir / f"{molecule}_{label}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[label] = path
    _write_manifest_meta(
        out_dir / f"{molecule}_manifest_meta.json", seed, paths,
        extra={"final_test_purge": purge_stats, "forbidden_test_sha256": sha256_file(forbidden_test_path)},
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
    """Freeze 1100 experimental complexes with stable bilateral group splits."""
    df = pd.read_csv(eligible_path, sep=None, engine="python")
    _require_columns(df, REQUIRED_COMPLEX, "complex")
    if not df["experimental"].astype(bool).all():
        raise ValueError("Pilot complex pool is experimental-only; predicted structures are forbidden")
    if df["sample_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate sample_id values must be resolved before freezing")
    if df["mother_sample_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate mother_sample_id values must be grouped before freezing")

    pool = deterministic_sample(df, counts.complex_pool, seed)
    relaxations: list[dict] = []
    if require_strict_bilateral:
        dev, test = bilateral_group_split(pool, counts.complex_test, seed + 17)
    else:
        ranked = deterministic_sample(pool, len(pool), seed + 17)
        test = ranked.iloc[: counts.complex_test].copy().reset_index(drop=True)
        dev = ranked.iloc[counts.complex_test :].copy().reset_index(drop=True)
        relaxations.append({"rule": "strict_bilateral", "reason": "explicitly_disabled"})

    if len(dev) != counts.complex_dev:
        raise AssertionError("Complex dev size mismatch")
    if strict_validation:
        train, val = bilateral_group_split(dev, counts.complex_val, seed + 19)
        if len(train) != counts.complex_train:
            raise AssertionError("Complex train size mismatch")
    else:
        train, val = split_deterministic(dev, counts.complex_train, counts.complex_val, seed + 19)

    assert_no_test_leakage(train, val, test, strict_cluster_check=require_strict_bilateral)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {"pool": pool, "dev": dev, "train": train, "val": val, "test": test}
    paths: dict[str, Path] = {}
    for label, frame in frames.items():
        path = out_dir / f"complex_{label}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        paths[label] = path
    _write_manifest_meta(
        out_dir / "complex_manifest_meta.json", seed, paths,
        extra={"relaxations": relaxations, "strict_validation": strict_validation},
    )
    return paths


def assert_no_test_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, strict_cluster_check: bool = True) -> None:
    dev = pd.concat([train, val], ignore_index=True)
    for key in ["sample_id", "mother_sample_id", "protein_hash", "rna_hash"]:
        if key not in dev or key not in test:
            continue
        overlap = set(dev[key].astype(str)) & set(test[key].astype(str))
        if overlap:
            raise AssertionError(f"TEST LEAKAGE via {key}: {sorted(overlap)[:5]}")
    if strict_cluster_check and len(dev):
        p_overlap = set(dev["protein_cluster_p30"].astype(str)) & set(test["protein_cluster_p30"].astype(str))
        if p_overlap:
            raise AssertionError(f"P30 leakage into final test: {sorted(p_overlap)[:5]}")
        dev_r = {_rna_holdout_key(row) for _, row in dev.iterrows()}
        test_r = {_rna_holdout_key(row) for _, row in test.iterrows()}
        r_overlap = dev_r & test_r
        if r_overlap:
            raise AssertionError(f"RNA family/R80 leakage into final test: {sorted(r_overlap)[:5]}")


def assert_pretraining_disjoint(protein_pool: pd.DataFrame, rna_pool: pd.DataFrame, test: pd.DataFrame) -> None:
    """Hard audit that final-test exact sequences and families never enter priors."""
    _require_columns(protein_pool, REQUIRED_PROTEIN, "protein_pool")
    _require_columns(rna_pool, REQUIRED_RNA, "rna_pool")
    _require_columns(test, REQUIRED_COMPLEX, "complex_test")
    p_hash = set(protein_pool["sequence_hash"].astype(str)) & set(test["protein_hash"].astype(str))
    p_cluster = set(protein_pool["protein_cluster_p30"].astype(str)) & set(test["protein_cluster_p30"].astype(str))
    r_hash = set(rna_pool["sequence_hash"].astype(str)) & set(test["rna_hash"].astype(str))
    r80 = set(rna_pool["rna_cluster_r80"].astype(str)) & set(test["rna_cluster_r80"].astype(str))
    known_test_rfam = {str(x) for x in test["rfam_family"] if _known_rfam(x)}
    rfam = set(rna_pool["rfam_family"].astype(str)) & known_test_rfam
    if p_hash or p_cluster or r_hash or r80 or rfam:
        raise AssertionError(
            "FINAL TEST leaked into structural-prior pools: "
            f"p_hash={len(p_hash)} p30={len(p_cluster)} r_hash={len(r_hash)} r80={len(r80)} rfam={len(rfam)}"
        )


def _write_manifest_meta(path: Path, seed: int, paths: dict[str, Path], extra: dict | None = None) -> None:
    payload = {"seed": seed, "files": {name: {"path": str(p), "sha256": sha256_file(p)} for name, p in paths.items()}}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
