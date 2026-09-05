"""Generic Gemmi adapter for PDB/mmCIF structures.

The manifest is expected to point to the exact structure/biological-assembly file
used for the experiment. Chain IDs can be supplied explicitly; otherwise chains
are inferred from canonical residue names. All model features are sequence-neutral.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import hashlib
import math

import gemmi
import numpy as np
import torch

from pr_pilot.data.features import PROTEIN_ALLOWED, RNA_ALLOWED, rbf, virtual_cb
from pr_pilot.model.dmicf import PRBatch
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample, PolymerGraph

PROTEIN_TO_INDEX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
RNA_TO_INDEX = {b: i for i, b in enumerate("AUGC")}
AA3_TO_1 = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
}
RNA_NAME_TO_1 = {
    "A":"A","U":"U","G":"G","C":"C","RA":"A","RU":"U","RG":"G","RC":"C",
    "ADE":"A","URA":"U","GUA":"G","CYT":"C",
}

P_ATOMS = ("N", "CA", "C", "O", "VCB")
R_ATOMS = ("P", "C4'", "C1'", "O3'", "O5'", "C3'", "C5'")


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
    """Choose highest-occupancy conformer for every atom name."""
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
    return {k: (v + rng.normal(0.0, sigma, size=3).astype(np.float32)) for k, v in atoms.items()}


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
                if polymer == "protein":
                    one = AA3_TO_1.get(name)
                    if one is None:
                        continue
                    token = PROTEIN_TO_INDEX[one]
                else:
                    one = RNA_NAME_TO_1.get(name)
                    if one is None:
                        continue
                    token = RNA_TO_INDEX[one]
                atoms = _noise_atoms(_atom_dict(residue), rng, self.noise)
                if polymer == "protein":
                    if "CA" not in atoms:
                        continue
                    if all(x in atoms for x in ("N", "CA", "C")):
                        atoms["VCB"] = virtual_cb(atoms["N"], atoms["CA"], atoms["C"])
                    ref = atoms["CA"]
                    frame, _ = _make_frame(ref, atoms.get("C"), atoms.get("N"))
                    present = {a: a in atoms for a in (*PROTEIN_ALLOWED, "VCB")}
                else:
                    ref_name = "C4'" if "C4'" in atoms else ("C1'" if "C1'" in atoms else None)
                    if ref_name is None:
                        continue
                    ref = atoms[ref_name]
                    frame, _ = _make_frame(ref, atoms.get("C1'"), atoms.get("C3'"))
                    present = {a: a in atoms for a in RNA_ALLOWED}
                rid = f"{chain_name}:{residue.seqid.num}{residue.seqid.icode.strip()}:{name}"
                records.append(_Record(token, chain_name, chain_counter, local_order, rid, atoms, present, ref.astype(np.float32), frame))
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
            p = [float(r.present.get(a, False)) for a in PROTEIN_ALLOWED]
            def d(a: str, b: str) -> float:
                return float(np.linalg.norm(r.atoms[a] - r.atoms[b]) / 2.0) if a in r.atoms and b in r.atoms else 0.0
            local = [d("N","CA"), d("CA","C"), d("C","O")]
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
            relpos = r.order_in_chain / max(1.0, max(x.order_in_chain for x in recs if x.chain_index == r.chain_index))
            out.append(p + local + tors + [relpos, 1.0])
        return np.asarray(out, dtype=np.float32)

    def _rna_node_features(self, recs: list[_Record]) -> np.ndarray:
        out = []
        for idx, r in enumerate(recs):
            p = [float(r.present.get(a, False)) for a in RNA_ALLOWED]
            def d(a: str, b: str, scale: float = 3.0) -> float:
                return float(np.linalg.norm(r.atoms[a] - r.atoms[b]) / scale) if a in r.atoms and b in r.atoms else 0.0
            local = [d("P","C4'"), d("C4'","C1'"), d("C3'","O3'"), d("O5'","C5'")]
            prev = recs[idx - 1] if idx > 0 and recs[idx - 1].chain_index == r.chain_index else None
            nxt = recs[idx + 1] if idx + 1 < len(recs) and recs[idx + 1].chain_index == r.chain_index else None
            prev_d = float(np.linalg.norm(r.ref - prev.ref) / 10.0) if prev is not None else 0.0
            next_d = float(np.linalg.norm(r.ref - nxt.ref) / 10.0) if nxt is not None else 0.0
            relpos = r.order_in_chain / max(1.0, max(x.order_in_chain for x in recs if x.chain_index == r.chain_index))
            out.append(p + local + [relpos, prev_d, next_d, 1.0])
        return np.asarray(out, dtype=np.float32)

    def _intra_graph(self, recs: list[_Record]) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.stack([r.ref for r in recs])
        frames = np.stack([r.frame for r in recs])
        dist = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
        edges: set[tuple[int, int]] = set()
        for i in range(len(recs)):
            order = np.argsort(dist[i])
            for j in order:
                if i == j:
                    continue
                edges.add((i, int(j)))
                if sum(1 for a, _ in edges if a == i) >= self.intra_max_neighbors:
                    break
        for i in range(len(recs) - 1):
            if recs[i].chain_index == recs[i + 1].chain_index:
                edges.add((i, i + 1)); edges.add((i + 1, i))
        ordered = sorted(edges)
        feats = []
        for i, j in ordered:
            dij = float(dist[i, j])
            disp = frames[i].T @ (xyz[j] - xyz[i])
            rot = frames[i].T @ frames[j]
            same = float(recs[i].chain_index == recs[j].chain_index)
            sep = float(np.clip(recs[j].order_in_chain - recs[i].order_in_chain, -32, 32) / 32.0) if same else 0.0
            cov = float(same and abs(recs[j].order_in_chain - recs[i].order_in_chain) == 1)
            feat = np.concatenate([rbf(np.array([dij]), self.rbf_bins).reshape(-1), disp / 20.0, rot.reshape(-1), np.array([same, sep, cov], np.float32)])
            feats.append(feat.astype(np.float32))
        if not ordered:
            return np.zeros((2,0), dtype=np.int64), np.zeros((0, self.rbf_bins + 15), dtype=np.float32)
        return np.asarray(ordered, dtype=np.int64).T, np.stack(feats)

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

    def _pr_atom_arrays(self, rec: _Record, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        coords = np.zeros((len(names),3), dtype=np.float32)
        mask = np.zeros(len(names), dtype=bool)
        for k, name in enumerate(names):
            if name in rec.atoms:
                coords[k] = rec.atoms[name]
                mask[k] = True
        return coords, mask

    def _pr_edges(self, p: list[_Record], r: list[_Record]) -> PRBatch:
        candidates: list[tuple[int,int,float]] = []
        pair_cache: dict[tuple[int,int], tuple[np.ndarray,np.ndarray,float]] = {}
        for i, pr in enumerate(p):
            pc, pm = self._pr_atom_arrays(pr, P_ATOMS)
            for j, rr in enumerate(r):
                rc, rm = self._pr_atom_arrays(rr, R_ATOMS)
                mask = pm[:,None] & rm[None,:]
                dd = np.linalg.norm(pc[:,None,:] - rc[None,:,:], axis=-1)
                eff = float(dd[mask].min()) if mask.any() else float(np.linalg.norm(pr.ref - rr.ref))
                pair_cache[(i,j)] = (dd, mask, eff)
                if eff <= self.pr_cutoff:
                    candidates.append((i,j,eff))
        keep: set[tuple[int,int]] = set()
        for i in range(len(p)):
            vals = sorted((x for x in candidates if x[0] == i), key=lambda x: x[2])[:self.pr_max_neighbors]
            keep.update((a,b) for a,b,_ in vals)
        for j in range(len(r)):
            vals = sorted((x for x in candidates if x[1] == j), key=lambda x: x[2])[:self.pr_max_neighbors]
            keep.update((a,b) for a,b,_ in vals)
        ordered = sorted(keep)
        if not ordered:
            raise ValueError("No protein-RNA edge under configured cutoff")
        feats, effs = [], []
        for i,j in ordered:
            dd, mask, eff = pair_cache[(i,j)]
            effs.append(eff)
            if self.rich_pr_geometry:
                distance_blocks = []
                for value, valid in zip(dd.reshape(-1), mask.reshape(-1)):
                    block = rbf(np.array([value if valid else 20.0], dtype=np.float32), self.rbf_bins).reshape(-1)
                    distance_blocks.append(block * float(valid))
                disp_p = p[i].frame.T @ (r[j].ref - p[i].ref)
                disp_r = r[j].frame.T @ (p[i].ref - r[j].ref)
                rot = p[i].frame.T @ r[j].frame
                feat = np.concatenate(distance_blocks + [mask.astype(np.float32).reshape(-1), disp_p/20.0, disp_r/20.0, rot.reshape(-1)])
            else:
                feat = rbf(np.array([eff], dtype=np.float32), self.rbf_bins).reshape(-1)
            feats.append(feat.astype(np.float32))
        return PRBatch(
            protein_index=torch.tensor([i for i,_ in ordered], dtype=torch.long),
            rna_index=torch.tensor([j for _,j in ordered], dtype=torch.long),
            edge_features=torch.from_numpy(np.stack(feats)),
            effective_distance=torch.tensor(effs, dtype=torch.float32),
        )

    def load_protein(self, structure_path: Path, sample_id: str, chains: Sequence[str] | None = None) -> PolymerGraph:
        return self._polymer_graph(self._read_records(structure_path, sample_id, "protein", chains), "protein")

    def load_rna(self, structure_path: Path, sample_id: str, chains: Sequence[str] | None = None) -> PolymerGraph:
        return self._polymer_graph(self._read_records(structure_path, sample_id, "rna", chains), "rna")

    def load_complex(self, structure_path: Path, sample_id: str, protein_chains: Sequence[str] | None = None, rna_chains: Sequence[str] | None = None) -> ComplexTensorSample:
        p_rec = self._read_records(structure_path, sample_id, "protein", protein_chains)
        r_rec = self._read_records(structure_path, sample_id, "rna", rna_chains)
        p_graph = self._polymer_graph(p_rec, "protein")
        r_graph = self._polymer_graph(r_rec, "rna")
        pr = self._pr_edges(p_rec, r_rec)
        p_graph.interface[torch.unique(pr.protein_index)] = True
        r_graph.interface[torch.unique(pr.rna_index)] = True
        sample = ComplexTensorSample(sample_id, p_graph, r_graph, pr, metadata={"structure_path": str(structure_path)})
        sample.validate()
        return sample
