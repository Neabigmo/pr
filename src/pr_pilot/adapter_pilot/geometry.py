"""SE(3)-invariant cross-chain geometry for the conditional Adapter.

The implementation deliberately keeps geometry independent of residue identity.
Only backbone atoms and deterministic virtual anchors are used.  A cross edge
is admitted only when at least one complete anchor pair has finite coordinates;
missing atoms therefore cannot become a zero-coordinate fake contact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np

from pr_pilot.data.features import rbf, virtual_cb
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter


P_ANCHORS = ("CA", "VCB")
R_ANCHORS = ("C1'", "P", "VN")
FULL_GEOMETRY_DIM = 114  # 6*16 RBF + 6 masks + 3+3 directions + 6D rotation


@dataclass(frozen=True)
class CrossEdgeSet:
    protein_index: np.ndarray  # [E]
    rna_index: np.ndarray  # [E]
    distance: np.ndarray  # [E], minimum valid anchor distance
    features: np.ndarray  # [E, D], raw geometry for G0/G1/G2
    feature_dim: int
    radius_angstrom: float
    max_neighbors: int


def _norm(vector: np.ndarray) -> np.ndarray:
    value = float(np.linalg.norm(vector))
    return vector / value if value > 1e-8 else np.zeros_like(vector)


def _frame(origin: np.ndarray, x_point: np.ndarray | None, y_point: np.ndarray | None) -> tuple[np.ndarray, bool]:
    if x_point is None or y_point is None:
        return np.eye(3, dtype=np.float32), False
    x = _norm(x_point - origin)
    y0 = y_point - origin
    y = _norm(y0 - np.dot(y0, x) * x)
    z = _norm(np.cross(x, y))
    if min(np.linalg.norm(x), np.linalg.norm(y), np.linalg.norm(z)) < 0.5:
        return np.eye(3, dtype=np.float32), False
    y = _norm(np.cross(z, x))
    return np.stack([x, y, z], axis=1).astype(np.float32), True


def _virtual_na_n(atoms: dict[str, np.ndarray]) -> tuple[np.ndarray | None, bool]:
    required = ("O4'", "C1'", "C2'")
    if not all(name in atoms for name in required):
        return None, False
    o4, c1, c2 = (atoms[name] for name in required)
    b = c1 - o4
    c = c2 - c1
    a = np.cross(b, c)
    value = -0.56967352 * a + 0.51055973 * b - 0.53122153 * c + c1
    return value.astype(np.float32), bool(np.isfinite(value).all())


def _jitter_atoms(record, rng: np.random.Generator | None, noise: float) -> dict[str, np.ndarray]:
    atoms = {name: np.asarray(value, dtype=np.float32).copy() for name, value in record.atoms.items()}
    if rng is not None and noise > 0:
        for name, value in list(atoms.items()):
            if not name.startswith("V") and np.isfinite(value).all():
                atoms[name] = value + rng.normal(0.0, noise, size=3).astype(np.float32)
    return atoms


def _protein_anchors(record, rng: np.random.Generator | None = None, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray]:
    atoms = _jitter_atoms(record, rng, noise)
    if all(name in atoms for name in ("N", "CA", "C")):
        atoms["VCB"] = virtual_cb(atoms["N"], atoms["CA"], atoms["C"])
    values: list[np.ndarray] = []
    valid: list[bool] = []
    for name in ("CA", "VCB"):
        value = atoms.get(name)
        ok = value is not None and np.isfinite(value).all()
        values.append(np.asarray(value if ok else np.zeros(3), dtype=np.float32))
        valid.append(bool(ok))
    frame, frame_ok = _frame(atoms.get("CA", np.zeros(3)), atoms.get("C"), atoms.get("N"))
    return np.stack(values), np.asarray(valid, dtype=bool), frame, frame_ok, atoms["CA"]


def _rna_anchors(record, rng: np.random.Generator | None = None, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray]:
    atoms = _jitter_atoms(record, rng, noise)
    vn, vn_ok = _virtual_na_n(atoms)
    values: list[np.ndarray] = []
    valid: list[bool] = []
    for name, value, ok in (
        ("C1'", atoms.get("C1'"), atoms.get("C1'") is not None),
        ("P", atoms.get("P"), atoms.get("P") is not None),
        ("VN", vn, vn_ok),
    ):
        finite = bool(ok and value is not None and np.isfinite(value).all())
        values.append(np.asarray(value if finite else np.zeros(3), dtype=np.float32))
        valid.append(finite)
    origin = atoms.get("C1'", np.zeros(3))
    frame, frame_ok = _frame(origin, atoms.get("O4'"), atoms.get("C2'"))
    return np.stack(values), np.asarray(valid, dtype=bool), frame, frame_ok, atoms["C1'"]


def _one_feature(
    p_anchor: np.ndarray,
    p_valid: np.ndarray,
    p_frame: np.ndarray,
    p_frame_ok: bool,
    r_anchor: np.ndarray,
    r_valid: np.ndarray,
    r_frame: np.ndarray,
    r_frame_ok: bool,
    p_ref: np.ndarray,
    r_ref: np.ndarray,
    bins: int,
    mode: str,
) -> np.ndarray:
    mask = p_valid[:, None] & r_valid[None, :]
    distances = np.linalg.norm(p_anchor[:, None, :] - r_anchor[None, :, :], axis=-1)
    distance_blocks: list[np.ndarray] = []
    for distance, valid in zip(distances.reshape(-1), mask.reshape(-1)):
        values = rbf(np.asarray([distance if valid else 20.0], dtype=np.float32), bins=bins).reshape(-1)
        distance_blocks.append(values * float(valid))
    if mode == "G0":
        return np.concatenate([distance_blocks[0], np.asarray([float(mask[0, 0])], dtype=np.float32)])
    base = np.concatenate(distance_blocks + [mask.astype(np.float32).reshape(-1)])
    if mode == "G1":
        return base
    if mode != "G2":
        raise ValueError(f"unknown geometry mode: {mode}")
    if p_frame_ok and r_frame_ok:
        delta = r_ref - p_ref
        distance = float(np.linalg.norm(delta))
        if distance > 1e-8:
            forward = p_frame.T @ (delta / distance)
            reverse = r_frame.T @ (-delta / distance)
        else:
            forward = np.zeros(3, dtype=np.float32)
            reverse = np.zeros(3, dtype=np.float32)
        relative = p_frame.T @ r_frame
        rotation6 = relative[:, :2].reshape(-1)
    else:
        forward = np.zeros(3, dtype=np.float32)
        reverse = np.zeros(3, dtype=np.float32)
        rotation6 = np.zeros(6, dtype=np.float32)
    return np.concatenate([base, forward, reverse, rotation6]).astype(np.float32)


def _records(adapter: GemmiStructureAdapter, structure_path, sample_id: str, polymer: str, chains: Sequence[str]):
    # The pinned runtime parser already enforces canonical residues and required
    # reference atoms.  It does not expose this as public API, but using its
    # record representation keeps the cross geometry aligned with the runtime
    # residue order and avoids a second, potentially divergent parser.
    return adapter._read_records(structure_path, sample_id, polymer, chains)


def build_cross_edges(
    structure_path,
    sample_id: str,
    protein_chains: Sequence[str],
    rna_chains: Sequence[str],
    radius_angstrom: float = 8.0,
    max_neighbors: int = 12,
    bins: int = 16,
    geometry_mode: str = "G2",
    candidate_neighbors: int = 32,
    coordinate_noise_angstrom: float = 0.0,
    noise_seed: int = 0,
) -> tuple[CrossEdgeSet, list, list]:
    """Build a symmetric capped PR graph and return its aligned records.

    The union of per-protein and per-RNA top-K neighborhoods is used.  The
    larger candidate cap is only an internal cache allowance; callers may
    select a smaller K without reparsing coordinates.
    """
    adapter = GemmiStructureAdapter(rbf_bins=bins, pr_cutoff_angstrom=radius_angstrom, pr_max_neighbors=max_neighbors)
    p_records = _records(adapter, structure_path, sample_id, "protein", protein_chains)
    r_records = _records(adapter, structure_path, sample_id, "rna", rna_chains)
    digest = hashlib.sha256(f"{sample_id}|{noise_seed}|{coordinate_noise_angstrom:.6f}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))
    p_values = [_protein_anchors(record, rng, float(coordinate_noise_angstrom)) for record in p_records]
    r_values = [_rna_anchors(record, rng, float(coordinate_noise_angstrom)) for record in r_records]
    candidates: list[tuple[int, int, float]] = []
    pair_features: dict[tuple[int, int], np.ndarray] = {}
    for i, (pa, pm, pf, pf_ok, p_ref) in enumerate(p_values):
        for j, (ra, rm, rf, rf_ok, r_ref) in enumerate(r_values):
            valid = pm[:, None] & rm[None, :]
            if not valid.any():
                continue
            distances = np.linalg.norm(pa[:, None, :] - ra[None, :, :], axis=-1)
            effective = float(distances[valid].min())
            if effective <= float(radius_angstrom):
                candidates.append((i, j, effective))
                pair_features[(i, j)] = _one_feature(
                    pa, pm, pf, pf_ok, ra, rm, rf, rf_ok,
                    p_ref, r_ref, bins, geometry_mode,
                )
    keep: set[tuple[int, int]] = set()
    for i in range(len(p_records)):
        vals = sorted((item for item in candidates if item[0] == i), key=lambda item: item[2])[:candidate_neighbors]
        keep.update((a, b) for a, b, _ in vals)
    for j in range(len(r_records)):
        vals = sorted((item for item in candidates if item[1] == j), key=lambda item: item[2])[:candidate_neighbors]
        keep.update((a, b) for a, b, _ in vals)
    # Re-apply the requested K after the cache allowance.  This makes the
    # resulting graph exactly reproducible for each P3 neighborhood setting.
    final: set[tuple[int, int]] = set()
    for i in range(len(p_records)):
        vals = sorted((item for item in candidates if item[0] == i), key=lambda item: item[2])[:max_neighbors]
        final.update((a, b) for a, b, _ in vals)
    for j in range(len(r_records)):
        vals = sorted((item for item in candidates if item[1] == j), key=lambda item: item[2])[:max_neighbors]
        final.update((a, b) for a, b, _ in vals)
    ordered = sorted(final)
    if not ordered:
        raise ValueError(f"No valid cross edge for {sample_id} under radius={radius_angstrom}, K={max_neighbors}")
    distance_map = {(i, j): d for i, j, d in candidates}
    features = np.stack([pair_features[(i, j)] for i, j in ordered]).astype(np.float32)
    dims = {"G0": bins + 1, "G1": 6 * bins + 6, "G2": FULL_GEOMETRY_DIM}
    if features.shape[1] != dims[geometry_mode]:
        raise AssertionError(f"cross geometry dimension drift: {features.shape[1]} != {dims[geometry_mode]}")
    result = CrossEdgeSet(
        protein_index=np.asarray([i for i, _ in ordered], dtype=np.int64),
        rna_index=np.asarray([j for _, j in ordered], dtype=np.int64),
        distance=np.asarray([distance_map[item] for item in ordered], dtype=np.float32),
        features=features,
        feature_dim=features.shape[1],
        radius_angstrom=float(radius_angstrom),
        max_neighbors=int(max_neighbors),
    )
    return result, p_records, r_records


def heavy_contact_audit(p_records: Iterable, r_records: Iterable, thresholds: Sequence[float] = (5.0,)) -> dict:
    """Return true full-heavy-atom contact-pair counts for one complex."""
    values = {float(threshold): 0 for threshold in thresholds}
    distances: list[float] = []
    for p in p_records:
        p_atoms = [xyz for name, xyz in p.atoms.items() if not name.startswith("V") and np.isfinite(xyz).all()]
        if not p_atoms:
            continue
        p_xyz = np.stack(p_atoms)
        for r in r_records:
            r_atoms = [xyz for name, xyz in r.atoms.items() if not name.startswith("V") and np.isfinite(xyz).all()]
            if not r_atoms:
                continue
            d = np.linalg.norm(p_xyz[:, None, :] - np.stack(r_atoms)[None, :, :], axis=-1)
            minimum = float(d.min())
            distances.append(minimum)
            for threshold in values:
                if minimum < threshold:
                    values[threshold] += 1
    values["min"] = float(min(distances)) if distances else float("nan")
    values["pair_count"] = int(len(distances))
    return values
