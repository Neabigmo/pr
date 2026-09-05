"""Post-freeze schema and canonical-interface audit before any GPU training."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = {
    "protein_interface_residue_ids",
    "rna_interface_residue_ids",
    "canonical_interface_cutoff_angstrom",
    "canonical_interface_definition",
}


def _parse_ids(value: object, label: str) -> list[str]:
    try:
        items = json.loads(str(value)) if not isinstance(value, list) else value
    except Exception as exc:
        raise ValueError(f"Invalid JSON interface list for {label}: {value!r}") from exc
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError(f"Interface list {label} must be a JSON array of strings")
    if not items:
        raise ValueError(f"Canonical interface list {label} is empty")
    if len(items) != len(set(items)):
        raise ValueError(f"Canonical interface list {label} contains duplicates")
    return items


def _method_class(method: object) -> str:
    text = str(method).upper()
    if "NMR" in text:
        return "NMR"
    if "ELECTRON" in text or "CRYO" in text:
        return "cryo-EM"
    if "X-RAY" in text or "DIFFRACTION" in text:
        return "X-ray"
    return "other"


def audit_frozen_complexes(
    manifest_root: Path,
    expected_cutoff: float,
    out_dir: Path,
) -> dict:
    splits = ["dev", "train", "val", "test"]
    report: dict = {"expected_cutoff_angstrom": float(expected_cutoff), "splits": {}}
    errors: list[str] = []
    for split in splits:
        path = manifest_root / f"complex_{split}.tsv"
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        frame = pd.read_csv(path, sep="\t")
        missing = sorted(CANONICAL_COLUMNS - set(frame.columns))
        if missing:
            errors.append(f"{path.name} missing canonical columns {missing}")
            continue
        p_sizes, r_sizes = [], []
        for row in frame.itertuples(index=False):
            try:
                p_sizes.append(
                    len(_parse_ids(row.protein_interface_residue_ids, f"{row.sample_id}:protein"))
                )
                r_sizes.append(
                    len(_parse_ids(row.rna_interface_residue_ids, f"{row.sample_id}:rna"))
                )
            except ValueError as exc:
                errors.append(str(exc))
            cutoff = float(row.canonical_interface_cutoff_angstrom)
            if not np.isclose(cutoff, expected_cutoff, atol=1e-8):
                errors.append(
                    f"{row.sample_id}: canonical interface cutoff {cutoff} != frozen {expected_cutoff}"
                )
            if str(row.canonical_interface_definition) != "full_heavy_atom_min_distance":
                errors.append(
                    f"{row.sample_id}: unexpected interface definition {row.canonical_interface_definition!r}"
                )
        methods = frame.get("method", pd.Series(["unknown"] * len(frame))).map(_method_class)
        report["splits"][split] = {
            "n": int(len(frame)),
            "method_counts": {str(k): int(v) for k, v in methods.value_counts().items()},
            "protein_length": {
                "median": float(frame.protein_length.median()) if "protein_length" in frame else None,
                "p95": float(frame.protein_length.quantile(0.95)) if "protein_length" in frame else None,
            },
            "rna_length": {
                "median": float(frame.rna_length.median()) if "rna_length" in frame else None,
                "p95": float(frame.rna_length.quantile(0.95)) if "rna_length" in frame else None,
            },
            "canonical_protein_interface_residues_median": float(np.median(p_sizes)) if p_sizes else None,
            "canonical_rna_interface_residues_median": float(np.median(r_sizes)) if r_sizes else None,
            "nmr_policy": "model 1 only; Gemmi structure[0] is used consistently in screening/runtime",
        }
    report["errors"] = errors
    report["passed"] = not errors
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frozen_complex_schema_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if errors:
        raise ValueError("Frozen complex audit failed: " + "; ".join(errors[:10]))
    return report
