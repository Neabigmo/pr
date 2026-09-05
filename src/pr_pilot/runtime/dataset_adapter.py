"""Local structure adapter boundary.

This is the only intentionally installation-specific module. The pilot must not
hard-code paths from one machine into scientific code. Implementations should
parse the frozen manifest's `structure_path` and return tensors matching
`docs/IMPLEMENTATION_CONTRACT.md`.

A correct adapter should be able to support:
  - ordinary protein-only samples;
  - ordinary RNA-only samples;
  - protein-RNA biological-assembly mother samples;
  - deterministic coordinate noise;
  - sparse PP/RR/PR graphs;
  - interface masks;
  - rich PR geometry;
  - fixed/designable masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
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


@dataclass
class ComplexTensorSample:
    sample_id: str
    protein: PolymerGraph
    rna: PolymerGraph
    pr: PRBatch
    metadata: dict


class StructureAdapter(Protocol):
    def load_protein(self, structure_path: Path, sample_id: str) -> PolymerGraph: ...
    def load_rna(self, structure_path: Path, sample_id: str) -> PolymerGraph: ...
    def load_complex(self, structure_path: Path, sample_id: str) -> ComplexTensorSample: ...


class UnconfiguredLocalAdapter:
    """Deliberate fail-fast default.

    Replace this class with the user's actual mmCIF/PDB parser integration. Do not
    modify model code to accommodate ad-hoc local file formats; normalize here.
    """

    def _fail(self, kind: str, structure_path: Path, sample_id: str):
        raise NotImplementedError(
            f"Local {kind} structure adapter is not configured for sample {sample_id} at {structure_path}. "
            "Implement StructureAdapter using Gemmi and the contracts in docs/IMPLEMENTATION_CONTRACT.md."
        )

    def load_protein(self, structure_path: Path, sample_id: str) -> PolymerGraph:
        self._fail("protein", structure_path, sample_id)

    def load_rna(self, structure_path: Path, sample_id: str) -> PolymerGraph:
        self._fail("RNA", structure_path, sample_id)

    def load_complex(self, structure_path: Path, sample_id: str) -> ComplexTensorSample:
        self._fail("complex", structure_path, sample_id)
