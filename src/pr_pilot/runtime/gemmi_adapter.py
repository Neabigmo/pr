"""Audited Gemmi adapter with reporting-interface/message-graph separation.

``graph.interface`` is the canonical full-heavy-atom contact label at a fixed
biological cutoff, computed on unaugmented coordinates. The PR message graph is a
separate sequence-neutral receptive field. Inference refinement uses PR graph
indices and never native side-chain/base contact labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gemmi
import numpy as np
import torch

from pr_pilot.data.residue_vocab import classify_residue
from pr_pilot.runtime import gemmi_adapter_legacy as _legacy
from pr_pilot.runtime.gemmi_adapter_legacy import *  # noqa: F401,F403
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample, PolymerGraph


def _looks_hydrogen(atom_name: str) -> bool:
    letters = "".join(ch for ch in str(atom_name).upper() if ch.isalpha())
    return bool(letters) and letters[0] in {"H", "D"}


def _heavy_coords(record) -> np.ndarray:
    coords = [
        xyz
        for name, xyz in record.atoms.items()
        if not _looks_hydrogen(name) and np.isfinite(xyz).all()
    ]
    return np.asarray(coords, dtype=np.float32) if coords else np.zeros((0, 3), np.float32)


def _selected_chain_set(chains: Sequence[str] | None) -> set[str] | None:
    if chains is None:
        return None
    return {str(x).strip() for x in chains if str(x).strip()}


class GemmiStructureAdapter(_legacy.GemmiStructureAdapter):
    def __init__(
        self,
        rbf_bins: int = 24,
        intra_max_neighbors: int = 32,
        pr_cutoff_angstrom: float = 8.0,
        pr_max_neighbors: int = 12,
        coordinate_noise_angstrom: float = 0.0,
        seed: int = 20260905,
        rich_pr_geometry: bool = True,
        canonical_interface_cutoff_angstrom: float = 6.0,
    ):
        super().__init__(
            rbf_bins,
            intra_max_neighbors,
            pr_cutoff_angstrom,
            pr_max_neighbors,
            coordinate_noise_angstrom,
            seed,
            rich_pr_geometry,
        )
        if canonical_interface_cutoff_angstrom <= 0:
            raise ValueError("canonical_interface_cutoff_angstrom must be positive")
        self.canonical_interface_cutoff = float(canonical_interface_cutoff_angstrom)

    @staticmethod
    def _assert_no_unknown_selected_polymer(
        structure_path: Path,
        sample_id: str,
        chains: Sequence[str] | None,
    ) -> None:
        structure = gemmi.read_structure(str(structure_path))
        if len(structure) == 0:
            raise ValueError(f"No model in structure {structure_path}")
        selected = _selected_chain_set(chains)
        for chain in structure[0]:
            if selected is not None and str(chain.name) not in selected:
                continue
            for residue in chain:
                cls = classify_residue(residue.name)
                if cls.polymer == "other" and getattr(residue, "entity_type", None) == gemmi.EntityType.Polymer:
                    raise ValueError(
                        "Unknown polymer component would be silently dropped: "
                        f"{sample_id}:{chain.name}:{residue.seqid}:{residue.name}"
                    )

    def _canonical_interface_masks(self, p_records, r_records) -> tuple[torch.Tensor, torch.Tensor]:
        p_mask = np.zeros(len(p_records), dtype=bool)
        r_mask = np.zeros(len(r_records), dtype=bool)
        p_heavy = [_heavy_coords(record) for record in p_records]
        r_heavy = [_heavy_coords(record) for record in r_records]
        for i, pc in enumerate(p_heavy):
            if len(pc) == 0:
                continue
            for j, rc in enumerate(r_heavy):
                if len(rc) == 0:
                    continue
                d = float(np.linalg.norm(pc[:, None, :] - rc[None, :, :], axis=-1).min())
                if d <= self.canonical_interface_cutoff:
                    p_mask[i] = True
                    r_mask[j] = True
        if not p_mask.any() or not r_mask.any():
            raise ValueError("No canonical Protein-RNA heavy-atom interface under configured cutoff")
        return torch.from_numpy(p_mask), torch.from_numpy(r_mask)

    def _clean_records(
        self,
        structure_path: Path,
        sample_id: str,
        polymer: str,
        chains: Sequence[str] | None,
    ):
        clean = _legacy.GemmiStructureAdapter(
            self.rbf_bins,
            self.intra_max_neighbors,
            self.pr_cutoff,
            self.pr_max_neighbors,
            0.0,
            self.seed,
            self.rich_pr_geometry,
        )
        return clean._read_records(structure_path, sample_id, polymer, chains)

    def load_protein(
        self,
        structure_path: Path,
        sample_id: str,
        chains: Sequence[str] | None = None,
    ) -> PolymerGraph:
        self._assert_no_unknown_selected_polymer(structure_path, sample_id, chains)
        return super().load_protein(structure_path, sample_id, chains)

    def load_rna(
        self,
        structure_path: Path,
        sample_id: str,
        chains: Sequence[str] | None = None,
    ) -> PolymerGraph:
        self._assert_no_unknown_selected_polymer(structure_path, sample_id, chains)
        return super().load_rna(structure_path, sample_id, chains)

    def load_complex(
        self,
        structure_path: Path,
        sample_id: str,
        protein_chains: Sequence[str] | None = None,
        rna_chains: Sequence[str] | None = None,
    ) -> ComplexTensorSample:
        selected = None
        if protein_chains is not None or rna_chains is not None:
            selected = list(protein_chains or []) + list(rna_chains or [])
        self._assert_no_unknown_selected_polymer(structure_path, sample_id, selected)

        p_records = self._read_records(structure_path, sample_id, "protein", protein_chains)
        r_records = self._read_records(structure_path, sample_id, "rna", rna_chains)
        p_graph = self._polymer_graph(p_records, "protein")
        r_graph = self._polymer_graph(r_records, "rna")
        pr = self._pr_edges(p_records, r_records)

        clean_p = self._clean_records(structure_path, sample_id + "|clean", "protein", protein_chains)
        clean_r = self._clean_records(structure_path, sample_id + "|clean", "rna", rna_chains)
        if [x.residue_id for x in clean_p] != [x.residue_id for x in p_records]:
            raise AssertionError("Protein residue order drift between clean and augmented views")
        if [x.residue_id for x in clean_r] != [x.residue_id for x in r_records]:
            raise AssertionError("RNA residue order drift between clean and augmented views")
        p_graph.interface, r_graph.interface = self._canonical_interface_masks(clean_p, clean_r)

        sample = ComplexTensorSample(
            sample_id,
            p_graph,
            r_graph,
            pr,
            metadata={
                "structure_path": str(structure_path),
                "canonical_interface_cutoff_angstrom": self.canonical_interface_cutoff,
                "pr_graph_cutoff_angstrom": self.pr_cutoff,
                "pr_graph_max_neighbors": self.pr_max_neighbors,
            },
        )
        sample.validate()
        return sample
