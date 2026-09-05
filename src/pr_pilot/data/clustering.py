"""Audited clustering/Rfam annotation wrapper.

The original MMseqs/Rfam orchestration lives in ``clustering_legacy.py``. This
module fixes the Infernal ``cmscan --fmt 2`` parser: format 2 prepends a hit index
to the ordinary target/accession/query fields, so target name/accession/query name
are columns 1/2/3 rather than 0/1/2.
"""
from __future__ import annotations

from pathlib import Path

from pr_pilot.data import clustering_legacy as _legacy
from pr_pilot.data.clustering_legacy import *  # noqa: F401,F403


def parse_rfam_tbl_fmt2(lines: list[str]) -> dict[str, set[str]]:
    """Parse Infernal cmscan ``--tblout --fmt 2`` into query -> Rfam families."""
    hits: dict[str, set[str]] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        # fmt2: idx, target name, target accession, query name, query accession,
        # clan name, ...
        if len(fields) < 6:
            raise ValueError(f"Malformed cmscan fmt2 row: {line!r}")
        target_name = fields[1]
        target_accession = fields[2]
        query_name = fields[3]
        family = target_accession if target_accession not in {"-", "."} else target_name
        hits.setdefault(query_name, set()).add(family)
    return hits


def run_rfam_cmscan(
    fasta: Path,
    rfam_cm: Path,
    clanin: Path,
    out_dir: Path,
    cpu: int = 4,
) -> dict[str, set[str]]:
    cmscan = _legacy._require_executable("cmscan")
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl = out_dir / "rfam.tbl"
    command = [
        cmscan,
        "--cpu",
        str(cpu),
        "--cut_ga",
        "--rfam",
        "--nohmmonly",
        "--clanin",
        str(clanin),
        "--oskip",
        "--fmt",
        "2",
        "--tblout",
        str(tbl),
        str(rfam_cm),
        str(fasta),
    ]
    _legacy._run(command, out_dir / "cmscan.log")
    return parse_rfam_tbl_fmt2(tbl.read_text(encoding="utf-8").splitlines())


# The legacy annotate function resolves this global when it executes.
_legacy.run_rfam_cmscan = run_rfam_cmscan
annotate_all_candidates = _legacy.annotate_all_candidates
run_mmseqs_cluster = _legacy.run_mmseqs_cluster
prepare_rfam_database = _legacy.prepare_rfam_database

__all__ = [
    "parse_rfam_tbl_fmt2",
    "run_rfam_cmscan",
    "run_mmseqs_cluster",
    "prepare_rfam_database",
    "annotate_all_candidates",
]
