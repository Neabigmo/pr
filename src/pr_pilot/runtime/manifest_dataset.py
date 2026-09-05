"""Manifest-backed structure datasets with a model-independent interface mask.

The canonical biological interface is deliberately decoupled from the DM-ICF
message-passing graph. It is defined once from the unperturbed source coordinates
as any Protein/RNA residue pair with a heavy-atom distance <= 6.0 A. The PR graph
may use a wider cutoff and neighbour cap, but changing those receptive-field
hyperparameters must never change interface/non-interface labels used in losses,
baseline mapping or evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import gemmi
import numpy as np
import pandas as pd
import torch

from pr_pilot.data.residue_vocab import classify_residue
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample, PolymerGraph, StructureAdapter
from pr_pilot.runtime.gemmi_adapter import parse_chain_list


CANONICAL_INTERFACE_CUTOFF_ANGSTROM = 6.0


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    structure_path: Path
    raw: dict


class ManifestTable:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.df = pd.read_csv(self.path, sep=None, engine="python")
        if "sample_id" not in self.df or "structure_path" not in self.df:
            raise ValueError(f"Manifest {path} requires sample_id and structure_path")
        if self.df["sample_id"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate sample_id in {path}")

    def __len__(self) -> int:
        return len(self.df)

    def rows(self) -> Iterator[ManifestRow]:
        for _, row in self.df.iterrows():
            raw = row.to_dict()
            yield ManifestRow(str(row["sample_id"]), Path(str(row["structure_path"])), raw)


def _rid(chain_name: str, residue: gemmi.Residue) -> str:
    name = residue.name.strip().upper()
    return f"{chain_name}:{residue.seqid.num}{residue.seqid.icode.strip()}:{name}"


def _heavy_xyz(residue: gemmi.Residue) -> np.ndarray:
    chosen: dict[str, tuple[float, np.ndarray]] = {}
    for atom in residue:
        if atom.element.name == "H":
            continue
        xyz = np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
        if not np.isfinite(xyz).all():
            continue
        name = atom.name.strip()
        occ = float(atom.occ)
        if name not in chosen or occ > chosen[name][0]:
            chosen[name] = (occ, xyz)
    if not chosen:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([v[1] for v in chosen.values()]).astype(np.float32)


@lru_cache(maxsize=4096)
def _canonical_interface_ids_cached(
    structure_path: str,
    protein_chains_key: tuple[str, ...],
    rna_chains_key: tuple[str, ...],
    cutoff: float,
) -> tuple[frozenset[str], frozenset[str]]:
    structure = gemmi.read_structure(structure_path)
    if len(structure) == 0:
        raise ValueError(f"No model in structure {structure_path}")
    psel, rsel = set(protein_chains_key), set(rna_chains_key)
    proteins: list[tuple[str, np.ndarray]] = []
    rnas: list[tuple[str, np.ndarray]] = []
    seen_p: set[str] = set()
    seen_r: set[str] = set()

    for chain in structure[0]:
        chain_name = str(chain.name)
        if chain_name not in psel and chain_name not in rsel:
            continue
        for residue in chain:
            cls = classify_residue(residue.name)
            if chain_name in psel and cls.polymer == "protein" and cls.token is not None:
                rid = _rid(chain_name, residue)
                proteins.append((rid, _heavy_xyz(residue)))
                seen_p.add(chain_name)
            elif chain_name in rsel and cls.polymer == "rna" and cls.token is not None:
                rid = _rid(chain_name, residue)
                rnas.append((rid, _heavy_xyz(residue)))
                seen_r.add(chain_name)

    if seen_p != psel:
        raise ValueError(f"Canonical interface: missing Protein chains {sorted(psel - seen_p)} in {structure_path}")
    if seen_r != rsel:
        raise ValueError(f"Canonical interface: missing RNA chains {sorted(rsel - seen_r)} in {structure_path}")
    if not proteins or not rnas:
        raise ValueError(f"Canonical interface requires both polymers in {structure_path}")

    p_interface: set[str] = set()
    r_interface: set[str] = set()
    cutoff2 = float(cutoff) ** 2
    for prid, pxyz in proteins:
        if pxyz.size == 0:
            continue
        for rrid, rxyz in rnas:
            if rxyz.size == 0:
                continue
            # Squared distances avoid an unnecessary sqrt and use every heavy atom,
            # independent of the DM-ICF PR atom subset/neighbour cap.
            delta = pxyz[:, None, :] - rxyz[None, :, :]
            if np.any(np.sum(delta * delta, axis=-1) <= cutoff2):
                p_interface.add(prid)
                r_interface.add(rrid)

    if not p_interface or not r_interface:
        raise ValueError(f"No canonical heavy-atom Protein-RNA contact <= {cutoff:.2f} A in {structure_path}")
    return frozenset(p_interface), frozenset(r_interface)


def canonical_interface_ids(
    structure_path: Path,
    protein_chains: list[str],
    rna_chains: list[str],
    cutoff: float = CANONICAL_INTERFACE_CUTOFF_ANGSTROM,
) -> tuple[set[str], set[str]]:
    p, r = _canonical_interface_ids_cached(
        str(Path(structure_path).resolve()),
        tuple(sorted(protein_chains)),
        tuple(sorted(rna_chains)),
        float(cutoff),
    )
    return set(p), set(r)


def load_protein_row(adapter: StructureAdapter, row: ManifestRow) -> PolymerGraph:
    chains = parse_chain_list(row.raw.get("protein_chains", row.raw.get("chains")))
    return adapter.load_protein(row.structure_path, row.sample_id, chains=chains)


def load_rna_row(adapter: StructureAdapter, row: ManifestRow) -> PolymerGraph:
    chains = parse_chain_list(row.raw.get("rna_chains", row.raw.get("chains")))
    return adapter.load_rna(row.structure_path, row.sample_id, chains=chains)


def load_complex_row(adapter: StructureAdapter, row: ManifestRow) -> ComplexTensorSample:
    pchains = parse_chain_list(row.raw.get("protein_chains")) or []
    rchains = parse_chain_list(row.raw.get("rna_chains")) or []
    if not pchains or not rchains:
        raise ValueError(f"Complex manifest row {row.sample_id} requires explicit protein_chains and rna_chains")
    sample = adapter.load_complex(
        row.structure_path,
        row.sample_id,
        protein_chains=pchains,
        rna_chains=rchains,
    )

    cutoff = float(row.raw.get("canonical_interface_cutoff_angstrom", CANONICAL_INTERFACE_CUTOFF_ANGSTROM))
    p_ids, r_ids = canonical_interface_ids(row.structure_path, pchains, rchains, cutoff)
    p_mask = torch.tensor([rid in p_ids for rid in sample.protein.residue_ids], dtype=torch.bool)
    r_mask = torch.tensor([rid in r_ids for rid in sample.rna.residue_ids], dtype=torch.bool)
    if not p_mask.any() or not r_mask.any():
        raise ValueError(f"Canonical interface mapping produced an empty mask for {row.sample_id}")
    # Ensure every canonical contact residue that survived screening is represented
    # by the runtime graph. Silent residue-ID drift would invalidate all interface metrics.
    missing_p = p_ids - set(sample.protein.residue_ids)
    missing_r = r_ids - set(sample.rna.residue_ids)
    if missing_p or missing_r:
        raise ValueError(
            f"Canonical interface residue mapping drift for {row.sample_id}: "
            f"missing_protein={sorted(missing_p)[:5]} missing_rna={sorted(missing_r)[:5]}"
        )
    sample.protein.interface = p_mask
    sample.rna.interface = r_mask
    sample.metadata.update(
        {
            "canonical_interface_cutoff_angstrom": cutoff,
            "canonical_interface_definition": "full-heavy-atom",
            "pr_graph_interface_is_not_metric_interface": True,
        }
    )
    sample.validate()
    return sample
