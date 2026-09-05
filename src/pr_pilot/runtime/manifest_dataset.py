"""Manifest-backed structure datasets.

No structure is cached across epochs by default because coordinate-noise augmentation
must be allowed to change deterministically with epoch. For a 1k pilot this simple
loader is preferable to a hidden preprocessing cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from pr_pilot.runtime.dataset_adapter import ComplexTensorSample, PolymerGraph, StructureAdapter
from pr_pilot.runtime.gemmi_adapter import parse_chain_list


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


def load_protein_row(adapter: StructureAdapter, row: ManifestRow) -> PolymerGraph:
    chains = parse_chain_list(row.raw.get("protein_chains", row.raw.get("chains")))
    return adapter.load_protein(row.structure_path, row.sample_id, chains=chains)


def load_rna_row(adapter: StructureAdapter, row: ManifestRow) -> PolymerGraph:
    chains = parse_chain_list(row.raw.get("rna_chains", row.raw.get("chains")))
    return adapter.load_rna(row.structure_path, row.sample_id, chains=chains)


def load_complex_row(adapter: StructureAdapter, row: ManifestRow) -> ComplexTensorSample:
    pchains = parse_chain_list(row.raw.get("protein_chains"))
    rchains = parse_chain_list(row.raw.get("rna_chains"))
    return adapter.load_complex(row.structure_path, row.sample_id, protein_chains=pchains, rna_chains=rchains)
