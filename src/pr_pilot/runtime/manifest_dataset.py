"""Manifest-backed structure datasets.

The complex manifest is the single source of truth for the *canonical biological
interface* (full-heavy-atom contact mask frozen during screening).  The runtime
PR graph remains an independent message-passing receptive field and must never
redefine interface NLL/recovery labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import json

import pandas as pd
import torch

from pr_pilot.runtime.dataset_adapter import (
    ComplexTensorSample,
    PolymerGraph,
    StructureAdapter,
)
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
            yield ManifestRow(
                str(row["sample_id"]), Path(str(row["structure_path"])), raw
            )


def load_protein_row(adapter: StructureAdapter, row: ManifestRow) -> PolymerGraph:
    chains = parse_chain_list(row.raw.get("protein_chains", row.raw.get("chains")))
    return adapter.load_protein(row.structure_path, row.sample_id, chains=chains)


def load_rna_row(adapter: StructureAdapter, row: ManifestRow) -> PolymerGraph:
    chains = parse_chain_list(row.raw.get("rna_chains", row.raw.get("chains")))
    return adapter.load_rna(row.structure_path, row.sample_id, chains=chains)


def _canonical_ids(raw: dict, key: str) -> set[str]:
    if key not in raw:
        raise ValueError(
            f"Complex manifest is missing {key!r}. Re-run coordinate screening with "
            "the canonical heavy-atom interface schema; do not fall back to the model PR graph."
        )
    value = raw[key]
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            items = []
        else:
            items = json.loads(text)
    if not isinstance(items, list):
        raise ValueError(f"{key} must be a JSON array")
    return {str(x) for x in items}


def _apply_canonical_interface(
    graph: PolymerGraph, expected_ids: set[str], label: str
) -> None:
    observed = set(graph.residue_ids)
    missing = expected_ids - observed
    if missing:
        raise ValueError(
            f"Canonical {label} interface IDs are absent from runtime graph: "
            f"{sorted(missing)[:5]}"
        )
    graph.interface = torch.tensor(
        [rid in expected_ids for rid in graph.residue_ids], dtype=torch.bool
    )
    if expected_ids and not graph.interface.any():
        raise AssertionError(f"Non-empty canonical {label} interface became empty")
    graph.validate()


def load_complex_row(
    adapter: StructureAdapter, row: ManifestRow
) -> ComplexTensorSample:
    pchains = parse_chain_list(row.raw.get("protein_chains"))
    rchains = parse_chain_list(row.raw.get("rna_chains"))
    sample = adapter.load_complex(
        row.structure_path,
        row.sample_id,
        protein_chains=pchains,
        rna_chains=rchains,
    )
    p_ids = _canonical_ids(row.raw, "protein_interface_residue_ids")
    r_ids = _canonical_ids(row.raw, "rna_interface_residue_ids")
    _apply_canonical_interface(sample.protein, p_ids, "protein")
    _apply_canonical_interface(sample.rna, r_ids, "RNA")
    sample.metadata.update(
        {
            "canonical_interface": True,
            "canonical_interface_cutoff_angstrom": row.raw.get(
                "canonical_interface_cutoff_angstrom"
            ),
            "canonical_interface_definition": row.raw.get(
                "canonical_interface_definition", "full_heavy_atom_min_distance"
            ),
        }
    )
    sample.validate()
    return sample
