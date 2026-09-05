"""Audited screening wrapper.

Adds v3 postconditions to the legacy coordinate parser:
- complex individual Protein/RNA maximum lengths are enforced;
- an unrecognized CCD component annotated by Gemmi as a polymer residue is fatal
  instead of being silently skipped and shortening the supervised sequence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
import json

import gemmi
import pandas as pd

from pr_pilot.data import screening_legacy as _legacy
from pr_pilot.data.residue_vocab import classify_residue
from pr_pilot.data.screening_legacy import *  # noqa: F401,F403


def _unknown_polymer_components(path: Path) -> list[str]:
    try:
        _, _, _, structure = _legacy._metadata(path)
    except Exception:
        return []
    if len(structure) == 0:
        return []
    bad: list[str] = []
    for chain in structure[0]:
        for residue in chain:
            cls = classify_residue(residue.name)
            entity_type = getattr(residue, "entity_type", None)
            if cls.polymer == "other" and entity_type == gemmi.EntityType.Polymer:
                bad.append(f"{chain.name}:{residue.seqid}:{residue.name}")
    return bad


def screen_file(
    path: Path,
    kind: Literal["protein", "rna", "complex"],
    cfg: ScreenConfig,  # noqa: F405
) -> tuple[dict | None, str]:
    unknown = _unknown_polymer_components(path)
    if unknown:
        return None, f"unknown_polymer_component:{unknown[0]}"

    record, reason = _legacy.screen_file(path, kind, cfg)
    if record is None or kind != "complex":
        return record, reason
    p_len = int(record["protein_length"])
    r_len = int(record["rna_length"])
    if not (cfg.protein_min_length <= p_len <= cfg.protein_max_length):
        return None, "complex_protein_length"
    if not (cfg.rna_min_length <= r_len <= cfg.rna_max_length):
        return None, "complex_rna_length"
    return record, reason


def screen_download_manifest(
    download_manifest: Path,
    kind: Literal["protein", "rna", "complex"],
    out_dir: Path,
    cfg: ScreenConfig,  # noqa: F405
) -> tuple[pd.DataFrame, pd.DataFrame]:
    downloads = pd.read_csv(download_manifest, sep="\t")
    eligible: list[dict] = []
    rejected: list[dict] = []
    for row in downloads.itertuples(index=False):
        record, reason = screen_file(Path(row.path), kind, cfg)
        if record is None:
            rejected.append({"pdb_id": row.pdb_id, "path": row.path, "reason": reason})
        else:
            eligible.append(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    elig_df = pd.DataFrame(eligible)
    rej_df = pd.DataFrame(rejected)
    elig_df.to_csv(out_dir / f"{kind}_eligible.tsv", sep="\t", index=False)
    rej_df.to_csv(out_dir / f"{kind}_rejected.tsv", sep="\t", index=False)
    summary = {
        "kind": kind,
        "config": _legacy.asdict(cfg),
        "downloaded": len(downloads),
        "eligible": len(elig_df),
        "rejected": len(rej_df),
        "rejection_counts": rej_df["reason"].value_counts().to_dict() if len(rej_df) else {},
        "complex_individual_length_limits_enforced": kind == "complex",
        "unknown_polymer_components_rejected": True,
    }
    (out_dir / f"{kind}_screen_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return elig_df, rej_df
