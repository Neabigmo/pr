"""Structure-adapter contracts shared by Gemmi parsing, training and inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from torch import Tensor

from pr_pilot.model.dmicf import PRBatch


@dataclass
class PolymerGraph:
    node_x: Tensor
    edge_index: Tensor
    edge_x: Tensor
    sequence: Tensor
    interface: Tensor
    valid: Tensor
    fixed: Tensor
    reference_xyz: Tensor
    chain_index: Tensor
    residue_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        n = int(self.node_x.shape[0])
        for name, value in {
            "sequence": self.sequence,
            "interface": self.interface,
            "valid": self.valid,
            "fixed": self.fixed,
            "reference_xyz": self.reference_xyz,
            "chain_index": self.chain_index,
        }.items():
            if value.shape[0] != n:
                raise ValueError(f"{name} length {value.shape[0]} != node count {n}")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E]")
        if self.edge_x.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_x must align with edge_index")
        if self.reference_xyz.shape != (n, 3):
            raise ValueError("reference_xyz must be [N,3]")


@dataclass
class ComplexTensorSample:
    sample_id: str
    protein: PolymerGraph
    rna: PolymerGraph
    pr: PRBatch
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        self.protein.validate()
        self.rna.validate()
        e = int(self.pr.protein_index.shape[0])
        if self.pr.rna_index.shape[0] != e or self.pr.edge_features.shape[0] != e or self.pr.effective_distance.shape[0] != e:
            raise ValueError("PR tensors must share edge dimension")
        if e:
            if int(self.pr.protein_index.max()) >= self.protein.node_x.shape[0]:
                raise ValueError("PR protein index out of range")
            if int(self.pr.rna_index.max()) >= self.rna.node_x.shape[0]:
                raise ValueError("PR RNA index out of range")


class StructureAdapter(Protocol):
    def load_protein(self, structure_path: Path, sample_id: str, chains: Sequence[str] | None = None) -> PolymerGraph: ...
    def load_rna(self, structure_path: Path, sample_id: str, chains: Sequence[str] | None = None) -> PolymerGraph: ...
    def load_complex(
        self,
        structure_path: Path,
        sample_id: str,
        protein_chains: Sequence[str] | None = None,
        rna_chains: Sequence[str] | None = None,
    ) -> ComplexTensorSample: ...
