"""Audited Gemmi adapter with reporting-interface/message-graph separation.

``graph.interface`` is a canonical full-heavy-atom contact label at a fixed
biological cutoff, computed on *unaugmented* coordinates and used for supervised
loss grouping/reporting. The PR message graph remains a separate sequence-neutral
receptive field. Inference-time interface selection must use PR graph indices
rather than this reporting label (see inference/sampler.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from pr_pilot.runtime import gemmi_adapter_legacy as _legacy
from pr_pilot.runtime.gemmi_adapter_legacy import *  # noqa: F403,F401
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample


def _looks_hydrogen(atom_name: str) -> bool:
    letters = "".join(ch for ch in str(atom_name).upper() if ch.isalpha())
    return bool(letters) and letters[0] in {"H", "D"}


def _heavy_coords(record) -> np.ndarray:
    coords = [
        xyz
        for name, xyz in record.atoms.items()
        if not _looks_hydrogen(name) and np.isfinite(xyz).all()
    ]
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


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
            rbf_bins=rbf_bins,
            intra_max_neighbors=intra_max_neighbors,
            pr_cutoff_angstrom=pr_cutoff_angstrom,
            pr_max_neighbors=pr_max_neighbors,
            coordinate_noise_angstrom=coordinate_noise_angstrom,
            seed=seed,
            rich_pr_geometry=rich_pr_geometry,
        )
        cutoff = float(canonical_interface_cutoff_angstrom)
        if cutoff <= 0:
            raise ValueError("canonical_interface_cutoff_angstrom must be positive")
        self.canonical_interface_cutoff = cutoff

    def _canonical_interface_masks(
        self, protein_records, rna_records
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_mask = np.zeros(len(protein_records), dtype=bool)
        r_mask = np.zeros(len(rna_records), dtype=bool)
        p_heavy = [_heavy_coords(record) for record in protein_records]
        r_heavy = [_heavy_coords(record) for record in rna_records]
        cutoff = self.canonical_interface_cutoff
        for i, pc in enumerate(p_heavy):
            if len(pc) == 0:
                continue
            for j, rc in enumerate(r_heavy):
                if len(rc) == 0:
                    continue
                distance = np.linalg.norm(pc[:, None, :] - rc[None, :, :], axis=-1).min()
                if float(distance) <= cutoff:
                    p_mask[i] = True
                    r_mask[j] = True
        if not p_mask.any() or not r_mask.any():
            raise ValueError(
                "No canonical Protein-RNA heavy-atom interface under configured cutoff"
            )
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

    def load_complex(
        self,
        structure_path: Path,
        sample_id: str,
        protein_chains: Sequence[str] | None = None,
        rna_chains: Sequence[str] | None = None,
    ) -> ComplexTensorSample:
        # Noisy records are used only for model features and the training-time PR graph.
        protein_records = self._read_records(
            structure_path, sample_id, "protein", protein_chains
        )
        rna_records = self._read_records(structure_path, sample_id, "rna", rna_chains)
        protein_graph = self._polymer_graph(protein_records, "protein")
        rna_graph = self._polymer_graph(rna_records, "rna")
        pr = self._pr_edges(protein_records, rna_records)

        # Canonical interface is always derived from original deposited coordinates.
        clean_p = self._clean_records(
            structure_path, sample_id + "|clean", "protein", protein_chains
        )
        clean_r = self._clean_records(
            structure_path, sample_id + "|clean", "rna", rna_chains
        )
        if [x.residue_id for x in clean_p] != [x.residue_id for x in protein_records]:
            raise AssertionError("Protein residue order drift between clean and augmented views")
        if [x.residue_id for x in clean_r] != [x.residue_id for x in rna_records]:
            raise AssertionError("RNA residue order drift between clean and augmented views")
        p_interface, r_interface = self._canonical_interface_masks(clean_p, clean_r)
        protein_graph.interface = p_interface
        rna_graph.interface = r_interface

        sample = ComplexTensorSample(
            sample_id,
            protein_graph,
            rna_graph,
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
