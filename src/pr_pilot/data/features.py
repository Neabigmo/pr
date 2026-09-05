"""Sequence-neutral structural feature contracts.

The scientific purpose of this module is to make leakage impossible by default.
Full raw structures may contain native side chains/base atoms, but the DM-ICF
model view is built only from explicitly allowed atom names.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


PROTEIN_ALLOWED = ("N", "CA", "C", "O")
RNA_ALLOWED = (
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"
)
RNA_FORBIDDEN_IDENTITY_ATOMS = {
    "N1", "N2", "N3", "N4", "N6", "N7", "N9",
    "C2", "C4", "C5", "C6", "C8", "O2", "O4", "O6",
}


@dataclass
class AtomView:
    names: list[str]
    coords: np.ndarray  # [A,3]
    present: np.ndarray # [A]


def build_fixed_atom_view(atom_coordinates: Mapping[str, np.ndarray], allowed: tuple[str, ...]) -> AtomView:
    coords = np.zeros((len(allowed), 3), dtype=np.float32)
    present = np.zeros(len(allowed), dtype=bool)
    for i, atom in enumerate(allowed):
        if atom in atom_coordinates:
            xyz = np.asarray(atom_coordinates[atom], dtype=np.float32)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"Invalid coordinate for {atom}: {xyz}")
            coords[i] = xyz
            present[i] = True
    return AtomView(list(allowed), coords, present)


def protein_backbone_view(atom_coordinates: Mapping[str, np.ndarray]) -> AtomView:
    return build_fixed_atom_view(atom_coordinates, PROTEIN_ALLOWED)


def rna_backbone_view(atom_coordinates: Mapping[str, np.ndarray]) -> AtomView:
    forbidden = RNA_FORBIDDEN_IDENTITY_ATOMS & set(atom_coordinates)
    # Presence in raw input is allowed, but this builder must prove it does not expose them.
    view = build_fixed_atom_view(atom_coordinates, RNA_ALLOWED)
    if any(name in RNA_FORBIDDEN_IDENTITY_ATOMS for name in view.names):
        raise AssertionError(f"RNA identity-leaking atoms entered model view: {sorted(forbidden)}")
    return view


def virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute sequence-neutral virtual C-beta from backbone N/CA/C.

    Formula is a standard deterministic local-frame construction; it does not use
    native side-chain coordinates.
    """
    n = np.asarray(n, dtype=np.float32)
    ca = np.asarray(ca, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    b = ca - n
    d = c - ca
    b /= np.linalg.norm(b) + 1e-8
    d /= np.linalg.norm(d) + 1e-8
    a = np.cross(b, d)
    a /= np.linalg.norm(a) + 1e-8
    # Coefficients approximate tetrahedral geometry; exact values should be kept
    # identical across train/val/test and audited against the chosen baseline view.
    direction = -0.58273431 * b + 0.56802827 * d + 0.54067466 * a
    return ca + 1.522 * direction


def rbf(distances: np.ndarray, bins: int = 24, dmin: float = 2.0, dmax: float = 20.0) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)[..., None]
    centers = np.linspace(dmin, dmax, bins, dtype=np.float32)
    width = (dmax - dmin) / max(bins - 1, 1)
    return np.exp(-((distances - centers) / max(width, 1e-6)) ** 2)


def pairwise_allowed_distances(protein_view: AtomView, rna_view: AtomView) -> tuple[np.ndarray, np.ndarray]:
    """Return all allowed protein/RNA atom-pair distances plus a presence mask."""
    diff = protein_view.coords[:, None, :] - rna_view.coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = protein_view.present[:, None] & rna_view.present[None, :]
    return dist.astype(np.float32), mask


def assert_rna_model_view_has_no_identity_atoms(view: AtomView) -> None:
    bad = set(view.names) & RNA_FORBIDDEN_IDENTITY_ATOMS
    if bad:
        raise AssertionError(f"Fatal nucleotide identity leakage: {sorted(bad)}")
