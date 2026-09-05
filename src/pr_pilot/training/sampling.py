"""Family-aware samplers for the 1k pilot.

The scientific goal is to reduce domination by recurrent Protein/RNA families
without exploding the contribution of a family represented by only one sample.
For each row we estimate a conservative family frequency and sample with
replacement using weight ``frequency ** exponent`` (default exponent -0.5).

Because sampling is with replacement, every epoch records both draws and unique
sample count. Validation/refit selection never uses sampling statistics as a
metric; they are an audit trail only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from pr_pilot.runtime.manifest_dataset import ManifestRow, ManifestTable
from pr_pilot.training.stages import Stage


UNKNOWN = {"", "nan", "none", "unknown", "na", "n/a", ".", "?"}


def _labels(value: object) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip()
    if text.lower() in UNKNOWN:
        return set()
    return {part.strip() for part in text.split(";") if part.strip() and part.strip().lower() not in UNKNOWN}


def _stable_seed(seed: int, epoch: int, stage: Stage) -> int:
    digest = hashlib.sha256(f"{seed}|{epoch}|{stage.value}|family-sampler".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _row_labels(row: ManifestRow, stage: Stage) -> set[str]:
    raw = row.raw
    if stage == Stage.PROTEIN_PRIOR:
        labels = {f"p30:{x}" for x in _labels(raw.get("protein_cluster_p30"))}
    elif stage == Stage.RNA_PRIOR:
        labels = {f"r80:{x}" for x in _labels(raw.get("rna_cluster_r80"))}
        labels.update({f"rfam:{x}" for x in _labels(raw.get("rfam_family"))})
    else:
        labels = {f"p30:{x}" for x in _labels(raw.get("protein_cluster_p30"))}
        labels.update({f"r80:{x}" for x in _labels(raw.get("rna_cluster_r80"))})
        labels.update({f"rfam:{x}" for x in _labels(raw.get("rfam_family"))})
    # If annotation is missing, make the sample its own singleton family instead
    # of assigning all unknowns to one giant artificial family.
    return labels or {f"singleton:{row.sample_id}"}


@dataclass(frozen=True)
class SamplingAudit:
    draws: int
    unique_samples: int
    effective_sample_size: float
    min_probability: float
    max_probability: float


def family_sampling_probabilities(rows: list[ManifestRow], stage: Stage, exponent: float = -0.5) -> np.ndarray:
    if not rows:
        raise ValueError("Cannot sample an empty manifest")
    memberships = [_row_labels(row, stage) for row in rows]
    counts: dict[str, int] = {}
    for labels in memberships:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    # Conservative frequency = largest family in which the row participates.
    # A multichain complex cannot get a huge oversampling boost merely because one
    # constituent chain happens to be rare while another belongs to a common family.
    frequencies = np.asarray([max(counts[label] for label in labels) for labels in memberships], dtype=np.float64)
    weights = frequencies ** float(exponent)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid family-sampling weights")
    probabilities = weights / weights.sum()
    return probabilities


def epoch_rows(
    table: ManifestTable,
    stage: Stage,
    seed: int,
    epoch: int,
    *,
    enabled: bool = True,
    exponent: float = -0.5,
    draws: int | None = None,
) -> tuple[list[ManifestRow], SamplingAudit]:
    rows = list(table.rows())
    n_draws = int(draws if draws is not None else len(rows))
    if n_draws <= 0:
        raise ValueError("Epoch draw count must be positive")
    rng = np.random.default_rng(_stable_seed(seed, epoch, stage))
    if enabled:
        probabilities = family_sampling_probabilities(rows, stage, exponent)
        indices = rng.choice(len(rows), size=n_draws, replace=True, p=probabilities)
    else:
        probabilities = np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
        # No replacement when ordinary sampling is requested and draw count matches.
        if n_draws == len(rows):
            indices = rng.permutation(len(rows))
        else:
            indices = rng.choice(len(rows), size=n_draws, replace=True)
    selected = [rows[int(i)] for i in indices]
    ess = float(1.0 / np.square(probabilities).sum())
    audit = SamplingAudit(
        draws=n_draws,
        unique_samples=len({row.sample_id for row in selected}),
        effective_sample_size=ess,
        min_probability=float(probabilities.min()),
        max_probability=float(probabilities.max()),
    )
    return selected, audit
