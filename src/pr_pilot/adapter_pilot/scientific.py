"""Protocol contracts shared by the single-seed scientific pilot."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SINGLE_SEED = 20260917
PROTEIN_LENGTH_BOUNDS = (40, 2000)
RNA_LENGTH_BOUNDS = (10, 500)
K_CONFIGS: tuple[tuple[int, int], ...] = ((4, 8), (8, 8), (8, 12), (8, 16), (12, 16))
GEOMETRY_MODES = ("G0", "G1", "G2", "G3")
AGGREGATIONS = ("A0", "A1", "A2")
INTERACTIONS = ("concat", "centered", "multiplicative", "film")
RESIDUALS = ("direct", "partner_centered", "scalar_gate", "confidence_gate")


@dataclass(frozen=True)
class ScientificProtocol:
    """Immutable run-level choices recorded in every report."""

    seed: int = SINGLE_SEED
    protein_min_length: int = PROTEIN_LENGTH_BOUNDS[0]
    protein_max_length: int = PROTEIN_LENGTH_BOUNDS[1]
    rna_min_length: int = RNA_LENGTH_BOUNDS[0]
    rna_max_length: int = RNA_LENGTH_BOUNDS[1]
    test_policy: str = "legacy_holdout_final_only"
    resolution_method_filter: bool = False
    bootstrap_resamples: int = 10_000
    shuffle_repeats: int = 20

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_length_bounds(frame: pd.DataFrame, protocol: ScientificProtocol) -> dict[str, Any]:
    """Validate a manifest without silently dropping records."""
    required = {"protein_length", "rna_length"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest missing length columns: {missing}")
    p = pd.to_numeric(frame["protein_length"], errors="coerce")
    r = pd.to_numeric(frame["rna_length"], errors="coerce")
    p_bad = ~(p.between(protocol.protein_min_length, protocol.protein_max_length, inclusive="both"))
    r_bad = ~(r.between(protocol.rna_min_length, protocol.rna_max_length, inclusive="both"))
    return {
        "rows": int(len(frame)),
        "protein_out_of_range": int(p_bad.sum()),
        "rna_out_of_range": int(r_bad.sum()),
        "both_out_of_range": int((p_bad & r_bad).sum()),
        "protein_range": [protocol.protein_min_length, protocol.protein_max_length],
        "rna_range": [protocol.rna_min_length, protocol.rna_max_length],
        "resolution_method_filter": protocol.resolution_method_filter,
    }


def normalize_metric(value: float, prior: float) -> float:
    if not np.isfinite(value) or not np.isfinite(prior) or prior <= 0:
        return float("nan")
    return float(value / prior)


def selection_key(metrics: dict[str, float], require_ratio_gate: bool = True) -> tuple[int, float, float]:
    """Compare candidates without an unregistered specificity weight."""
    protein_ratio = float(metrics["protein_interface_ratio"])
    rna_ratio = float(metrics["rna_interface_ratio"])
    ratio_gate = protein_ratio < 1.0 and rna_ratio < 1.0
    specificity = float(metrics.get("specificity_mean", float("nan")))
    if not np.isfinite(specificity):
        specificity = -float("inf")
    gate_rank = 0 if (ratio_gate or not require_ratio_gate) else 1
    return gate_rank, max(protein_ratio, rna_ratio), -specificity


def choose_candidate(candidates: list[dict[str, Any]], require_ratio_gate: bool = True) -> dict[str, Any]:
    if not candidates:
        raise ValueError("cannot choose from no candidates")
    return min(candidates, key=lambda item: selection_key(item["metrics"], require_ratio_gate))


def write_protocol(path: Path, protocol: ScientificProtocol, extra: dict[str, Any] | None = None) -> None:
    import json

    payload = {"protocol": protocol.as_dict(), **(extra or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
