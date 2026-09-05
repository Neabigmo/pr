"""Generic Gemmi adapter for PDB/mmCIF structures.

The adapter is the single source of truth for tensor shapes used by training,
inference and evaluation. It exposes sequence-neutral geometry only and shares
canonical residue mapping with the data screener.

Rich Protein-RNA edges implement the agreed 5 x 12 atom geometry:
protein N/CA/C/O/virtual-CB against RNA
P/OP1/OP2/O5'/C5'/C4'/O4'/C3'/O3'/C2'/O2'/C1', plus missing-atom masks,
two local-frame displacements and relative frame rotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import hashlib
import math

import gemmi
import numpy as np
import torch

from pr_pilot.data.features import PROTEIN_ALLOWED, RNA_ALLOWED, rbf, virtual_cb
from pr_pilot.data.residue_vocab import PROTEIN_ALPHABET, RNA_ALPHABET, classify_residue
from pr_pilot.model.dmicf import PRBatch
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample, PolymerGraph

PROTEIN_TO_INDEX = {aa: i for i, aa in enumerate(PROTEIN_ALPHABET)}
RNA_TO_INDEX = {b: i for i, b in enumerate(RNA_ALPHABET)}

P_ATOMS = ("N", "CA", "C", "O", "VCB")
R_ATOMS = tuple(RNA_ALLOWED)
PROTEIN_NODE_DIM = len(PROTEIN_ALLOWED) + 3 + 6 + 2  # presence + local lengths + torsions + relpos/bias = 15
RNA_NODE_DIM = len(RNA_ALLOWED) + 4 + 4  # presence + local lengths + relpos/prev/next/bias = 20
INTRA_EDGE_EXTRA_DIM = 3 + 9 + 3  # displacement + relative rotation + same/sep/covalent
PR_RELATIVE_EXTRA_DIM = 3 + 3 + 9  # P-frame disp + R-frame disp + frame rotation


def feature_dimensions(rbf_bins: int, rich_pr_geometry: bool) -> dict[str, int]:
    """Return exact adapter feature dimensions; model code must never duplicate them."""
    pair_count = len(P_ATOMS) * len(R_ATOMS)
    return {
        "protein_node": PROTEIN_NODE_DIM,
        "rna_node": RNA_NODE_DIM,
        "protein_edge": int(rbf_bins) + INTRA_EDGE_EXTRA_DIM,
        "rna_edge": int(rbf_bins) + INTRA_EDGE_EXTRA_DIM,
        "pr_edge": (
            pair_count * int(rbf_bins) + pair_count + PR_RELATIVE_EXTRA_DIM
            if rich_pr_geometry
            else int(rbf_bins)
        ),
        "pr_atom_pairs": pair_count,
    }


@dataclass
class _Record:
    token: int
    chain: str
    chain_index: int
    order_in_chain: int
    residue_id: str
    atoms: dict[str, np.ndarray]
    present: dict[str, bool]
    ref: np.ndarray
    frame: np.ndarray


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else np.zeros_like(v)


def _make_frame(origin: np.ndarray, x_point: np.ndarray | None, y_point: np.ndarray | None) -> tuple[np.ndarray, bool]:
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


def _dihedral(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = _norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def _atom_dict(residue: gemmi.Residue) -> dict[str, np.ndarray]:
    """Choose the highest-occupancy conformer for each atom name."""
    chosen: dict[str, tuple[float, np.ndarray]] = {}
    for atom in residue:
        name = atom.name.strip()
        xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
        occ = float(atom.occ)
        if not np.isfinite(xyz).all():
            continue
        if name not in chosen or occ > chosen[name][0]:
            chosen[name] = (occ, xyz)
    return {k: v[1] for k, v in chosen.items()}


def _noise_atoms(atoms: dict[str, np.ndarray], rng: np.random.Generator, sigma: float) -> dict[str, np.ndarray]:
    if sigma <= 0:
        return {k: v.copy() for k, v in atoms.items()}
    return {k: v + rng.normal(0.0, sigma, size=3).astype(np.float32) for k, v in atoms.items()}


def _seed_for(base_seed: int, sample_id: str) -> int:
    h = hashlib.sha256(f"{base_seed}|{sample_id}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**32)


def _chain_filter(value: Sequence[str] | None) -> set[str] | None:
    if value is None:
        return None
    return {str(x).strip() for x in value if str(x).strip()}


def parse_chain_list(value: object) -> list[str] | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


class GemmiStructureAdapter:
    def __init__(
        self,
        rbf_bins: int = 24,
        intra_max_neighbors: int = 32,
        pr_cutoff_angstrom: float = 8.0,
        pr_max_neighbors: int = 12,
        coordinate_noise_angstrom: float = 0.0,
        seed: int = 20260905,
        rich_pr_geometry: bool = True,
    ):
        self.rbf_bins = int(rbf_bins)
        self.intra_max_neighbors = int(intra_max_neighbors)
        self.pr_cutoff = float(pr_cutoff_angstrom)
        self.pr_max_neighbors = int(pr_max_neighbors)
        self.noise = float(coordinate_noise_angstrom)
        self.seed = int(seed)
        self.rich_pr_geometry = bool(rich_pr_geometry)

    @property
    def dims(self) -> dict[str, int]:
        return feature_dimensions(self.rbf_bins, self.rich_pr_geometry)

    def _read_records(self, path: Path, sample_id: str, polymer: str, chains: Sequence[str] | None) -> list[_Record]:
        structure = gemmi.read_structure(str(path))
        if len(structure) == 0:
            raise ValueError(f"No model in structure {path}")
        model = structure[0]
        keep_chains = _chain_filter(chains)
        rng = np.random.default_rng(_seed_for(self.seed, sample_id + "|" + polymer))
        records: list[_Record] = []
        chain_counter = 0
        for chain in model:
            chain_name = str(chain.name)
            if keep_chains is not None and chain_name not in keep_chains:
                continue
            local_order = 0
            any_kept = False
            for residue in chain:
                name = residue.name.strip().upper()
                cls = classify_residue(name)
                if cls.polymer == "unsupported_polymer":
                    # An unsupported monomer inside an explicitly selected chain
                    # would shorten the supervised sequence silently; fail instead.
                    if keep_chains is not None:
                        raise ValueError(
                            f"Unsupported polymer residue {name} in selected chain {chain_name} of {sample_id}"
                        )
                    continue
                if cls.polymer != polymer or cls.token is None:
                    continue
                token = PROTEIN_TO_INDEX[cls.token] if polymer == "protein" else RNA_TO_INDEX[cls.token]
                atoms = _noise_atoms(_atom_dict(residue), rng, self.noise)
                if polymer == "protein":
                    if "CA" not in atoms:
                        raise ValueError(f"Missing CA in selected protein residue {sample_id}:{chain_name}:{residue.seqid}")
                    if all(x in atoms for x in ("N", "CA", "C")):
                        atoms["VCB"] = virtual_cb(atoms["N"], atoms["CA"], atoms["C"])
                    ref = atoms["CA"]
                    frame, _ = _make_frame(ref, atoms.get("C"), atoms.get("N"))
                    present = {a: a in atoms for a in (*PROTEIN_ALLOWED, "VCB")}
                else:
                    if "C4'" not in atoms or "C1'" not in atoms or "C3'" not in atoms:
                        raise ValueError(
                            f"Missing RNA frame atom C4'/C1'/C3' in selected residue {sample_id}:{chain_name}:{residue.seqid}"
                        )
                    ref = atoms["C4'"]
                    frame, ok = _make_frame(ref, atoms["C1'"], atoms["C3'"])
                    if not ok:
                        raise ValueError(f"Degenerate RNA local frame in {sample_id}:{chain_name}:{residue.seqid}")
                    present = {a: a in atoms for a in RNA_ALLOWED}
                rid = f"{chain_name}:{residue.seqid.num}{residue.seqid.icode.strip()}:{name}"
                records.append(
                    _Record(
                        token,
                        chain_name,
                        chain_counter,
                        local_order,
                        rid,
                        atoms,
                        present,
                        ref.astype(np.float32),
                        frame,
                    )
                )
                local_order += 1
                any_kept = True
            if any_kept:
                chain_counter += 1
        if not records:
            raise ValueError(f"No canonical {polymer} residues found in {path} for sample {sample_id}")
        return records

    def _protein_node_features(self, recs: list[_Record]) -> np.ndarray:
        out = []
        for idx, r in enumerate(recs):
            presence = [float(r.present.get(a, False)) for a in PROTEIN_ALLOWED]

            def d(a: str, b: str) -> float:
                return float(np.linalg.norm(r.atoms[a] - r.atoms[b]) / 2.0) if a in r.atoms and b in r.atoms else 0.0

            local = [d("N", "CA"), d("CA", "C"), d("C", "O")]
            phi = psi = omg = 0.0
            prev = recs[idx - 1] if idx > 0 and recs[idx - 1].chain_index == r.chain_index else None
            nxt = recs[idx + 1] if idx + 1 < len(recs) and recs[idx + 1].chain_index == r.chain_index else None
            try:
                if prev is not None:
                    phi = _dihedral(prev.atoms["C"], r.atoms["N"], r.atoms["CA"], r.atoms["C"])
                    omg = _dihedral(prev.atoms["CA"], prev.atoms["C"], r.atoms["N"], r.atoms["CA"])
                if nxt is not None:
                    psi = _dihedral(r.atoms["N"], r.atoms["CA"], r.atoms["C"], nxt.atoms["N"])
            except KeyError:
                pass
            tors = [math.sin(phi), math.cos(phi), math.sin(psi), math.cos(psi), math.sin(omg), math.cos(omg)]
            relpos = r.order_in_chain / max(
                1.0,
                max(x.order_in_chain for x in recs if x.chain_index == r.chain_index),
            )
            out.append(presence + local + tors + [relpos, 1.0])
        arr = np.asarray(out, dtype=np.float32)
        if arr.shape[1] != PROTEIN_NODE_DIM:
            raise AssertionError(f"Protein node feature contract drift: {arr.shape[1]} != {PROTEIN_NODE_DIM}")
        return arr

    def _rna_node_features(self, recs: list[_Record]) -> np.ndarray:
        out = []
        for idx, r in enumerate(recs):
            presence = [float(r.present.get(a, False)) for a in RNA_ALLOWED]

            def d(a: str, b: str, scale: float = 3.0) -> float:
                return float(np.linalg.norm(r.atoms[a] - r.atoms[b]) / scale) if a in r.atoms and b in r.atoms else 0.0

            local = [d("P", "C4'"), d("C4'", "C1'"), d("C3'", "O3'"), d("O5'", "C5'")]
            prev = recs[idx - 1] if idx > 0 and recs[idx - 1].chain_index == r.chain_index else None
            nxt = recs[idx + 1] if idx + 1 < len(recs) and recs[idx + 1].chain_index == r.chain_index else None
            prev_d = float(np.linalg.norm(r.ref - prev.ref) / 10.0) if prev is not None else 0.0
            next_d = float(np.linalg.norm(r.ref - nxt.ref) / 10.0) if nxt is not None else 0.0
            relpos = r.order_in_chain / max(
                1.0,
                max(x.order_in_chain for x in recs if x.chain_index == r.chain_index),
            )
            out.append(presence + local + [relpos, prev_d, next_d, 1.0])
        arr = np.asarray(out, dtype=np.float32)
        if arr.shape[1] != RNA_NODE_DIM:
            raise AssertionError(f"RNA node feature contract drift: {arr.shape[1]} != {RNA_NODE_DIM}")
        return arr

    def _intra_graph(self, recs: list[_Record]) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.stack([r.ref for r in recs])
        frames = np.stack([r.frame for r in recs])
        dist = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
        edges: set[tuple[int, int]] = set()
        outgoing = [0] * len(recs)
        for i in range(len(recs)):
            for j in np.argsort(dist[i]):
                if i == j:
                    continue
                edges.add((i, int(j)))
                outgoing[i] += 1
                if outgoing[i] >= self.intra_max_neighbors:
                    break
        for i in range(len(recs) - 1):
            if recs[i].chain_index == recs[i + 1].chain_index:
                edges.add((i, i + 1))
                edges.add((i + 1, i))
        ordered = sorted(edges)
        feats = []
        for i, j in ordered:
            dij = float(dist[i, j])
            disp = frames[i].T @ (xyz[j] - xyz[i])
            rot = frames[i].T @ frames[j]
            same = float(recs[i].chain_index == recs[j].chain_index)
            sep = (
                float(np.clip(recs[j].order_in_chain - recs[i].order_in_chain, -32, 32) / 32.0)
                if same
                else 0.0
            )
            cov = float(same and abs(recs[j].order_in_chain - recs[i].order_in_chain) == 1)
            feat = np.concatenate(
                [
                    rbf(np.array([dij]), self.rbf_bins).reshape(-1),
                    disp / 20.0,
                    rot.reshape(-1),
                    np.array([same, sep, cov], np.float32),
                ]
            )
            feats.append(feat.astype(np.float32))
        edge_dim = self.rbf_bins + INTRA_EDGE_EXTRA_DIM
        if not ordered:
            return np.zeros((2, 0), dtype=np.int64), np.zeros((0, edge_dim), dtype=np.float32)
        edge_x = np.stack(feats)
        if edge_x.shape[1] != edge_dim:
            raise AssertionError(f"Intra-edge feature contract drift: {edge_x.shape[1]} != {edge_dim}")
        return np.asarray(ordered, dtype=np.int64).T, edge_x

    def _polymer_graph(self, recs: list[_Record], polymer: str) -> PolymerGraph:
        node_x = self._protein_node_features(recs) if polymer == "protein" else self._rna_node_features(recs)
        edge_index, edge_x = self._intra_graph(recs)
        n = len(recs)
        graph = PolymerGraph(
            node_x=torch.from_numpy(node_x),
            edge_index=torch.from_numpy(edge_index).long(),
            edge_x=torch.from_numpy(edge_x),
            sequence=torch.tensor([r.token for r in recs], dtype=torch.long),
            interface=torch.zeros(n, dtype=torch.bool),
            valid=torch.ones(n, dtype=torch.bool),
            fixed=torch.zeros(n, dtype=torch.bool),
            reference_xyz=torch.from_numpy(np.stack([r.ref for r in recs]).astype(np.float32)),
            chain_index=torch.tensor([r.chain_index for r in recs], dtype=torch.long),
            residue_ids=[r.residue_id for r in recs],
        )
        graph.validate()
        return graph

    @staticmethod
    def _pr_atom_arrays(rec: _Record, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        coords = np.zeros((len(names), 3), dtype=np.float32)
        mask = np.zeros(len(names), dtype=bool)
        for k, name in enumerate(names):
            if name in rec.atoms:
                coords[k] = rec.atoms[name]
                mask[k] = True
        return coords, mask

    def _pr_edges(self, p: list[_Record], r: list[_Record]) -> PRBatch:
        candidates: list[tuple[int, int, float]] = []
        pair_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]] = {}
        for i, pr in enumerate(p):
            pc, pm = self._pr_atom_arrays(pr, P_ATOMS)
            for j, rr in enumerate(r):
                rc, rm = self._pr_atom_arrays(rr, R_ATOMS)
                mask = pm[:, None] & rm[None, :]
                dd = np.linalg.norm(pc[:, None, :] - rc[None, :, :], axis=-1)
                eff = float(dd[mask].min()) if mask.any() else float(np.linalg.norm(pr.ref - rr.ref))
                pair_cache[(i, j)] = (dd, mask, eff)
                if eff <= self.pr_cutoff:
                    candidates.append((i, j, eff))

        keep: set[tuple[int, int]] = set()
        for i in range(len(p)):
            vals = sorted((x for x in candidates if x[0] == i), key=lambda x: x[2])[: self.pr_max_neighbors]
            keep.update((a, b) for a, b, _ in vals)
        for j in range(len(r)):
            vals = sorted((x for x in candidates if x[1] == j), key=lambda x: x[2])[: self.pr_max_neighbors]
            keep.update((a, b) for a, b, _ in vals)
        ordered = sorted(keep)
        if not ordered:
            raise ValueError("No protein-RNA edge under configured cutoff")

        feats: list[np.ndarray] = []
        effs: list[float] = []
        for i, j in ordered:
            dd, mask, eff = pair_cache[(i, j)]
            effs.append(eff)
            if self.rich_pr_geometry:
                distance_blocks = []
                for value, valid in zip(dd.reshape(-1), mask.reshape(-1)):
                    block = rbf(
                        np.array([value if valid else 20.0], dtype=np.float32),
                        self.rbf_bins,
                    ).reshape(-1)
                    distance_blocks.append(block * float(valid))
                disp_p = p[i].frame.T @ (r[j].ref - p[i].ref)
                disp_r = r[j].frame.T @ (p[i].ref - r[j].ref)
                rot = p[i].frame.T @ r[j].frame
                feat = np.concatenate(
                    distance_blocks
                    + [
                        mask.astype(np.float32).reshape(-1),
                        disp_p / 20.0,
                        disp_r / 20.0,
                        rot.reshape(-1),
                    ]
                )
            else:
                feat = rbf(np.array([eff], dtype=np.float32), self.rbf_bins).reshape(-1)
            feats.append(feat.astype(np.float32))

        edge_features = np.stack(feats)
        expected = self.dims["pr_edge"]
        if edge_features.shape[1] != expected:
            raise AssertionError(f"PR-edge feature contract drift: {edge_features.shape[1]} != {expected}")
        return PRBatch(
            protein_index=torch.tensor([i for i, _ in ordered], dtype=torch.long),
            rna_index=torch.tensor([j for _, j in ordered], dtype=torch.long),
            edge_features=torch.from_numpy(edge_features),
            effective_distance=torch.tensor(effs, dtype=torch.float32),
        )

    def load_protein(
        self,
        structure_path: Path,
        sample_id: str,
        chains: Sequence[str] | None = None,
    ) -> PolymerGraph:
        return self._polymer_graph(self._read_records(structure_path, sample_id, "protein", chains), "protein")

    def load_rna(
        self,
        structure_path: Path,
        sample_id: str,
        chains: Sequence[str] | None = None,
    ) -> PolymerGraph:
        return self._polymer_graph(self._read_records(structure_path, sample_id, "rna", chains), "rna")

    def load_complex(
        self,
        structure_path: Path,
        sample_id: str,
        protein_chains: Sequence[str] | None = None,
        rna_chains: Sequence[str] | None = None,
    ) -> ComplexTensorSample:
        p_rec = self._read_records(structure_path, sample_id, "protein", protein_chains)
        r_rec = self._read_records(structure_path, sample_id, "rna", rna_chains)
        p_graph = self._polymer_graph(p_rec, "protein")
        r_graph = self._polymer_graph(r_rec, "rna")
        pr = self._pr_edges(p_rec, r_rec)
        p_graph.interface[torch.unique(pr.protein_index)] = True
        r_graph.interface[torch.unique(pr.rna_index)] = True
        sample = ComplexTensorSample(
            sample_id,
            p_graph,
            r_graph,
            pr,
            metadata={"structure_path": str(structure_path)},
        )
        sample.validate()
        return sample
