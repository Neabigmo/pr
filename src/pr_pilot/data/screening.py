"""Audited screening wrapper.

The legacy coordinate parser remains in ``screening_legacy.py``.  This wrapper
adds postconditions that were missing from the original complex path: the same
individual Protein/RNA maximum lengths used for the single-polymer pools must also
hold for a complex mother sample.  Screening and runtime continue to share the
6-A full-heavy-atom biological-interface concept.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
import json

import pandas as pd

from pr_pilot.data import screening_legacy as _legacy
from pr_pilot.data.screening_legacy import *  # noqa: F401,F403


def screen_file(
    path: Path,
    kind: Literal["protein", "rna", "complex"],
    cfg: ScreenConfig,  # noqa: F405
) -> tuple[dict | None, str]:
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
    }
    (out_dir / f"{kind}_screen_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return elig_df, rej_df
