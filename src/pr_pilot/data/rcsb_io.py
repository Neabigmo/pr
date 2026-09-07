"""RCSB PDB discovery and download helpers for the pilot.

The pilot uses official RCSB Search/File Download services rather than a manually
curated hidden list. Discovery is deliberately broad; all scientific QC happens
locally in ``screening.py`` and is recorded row-by-row.

Data order:
  discover candidate IDs -> download mmCIF -> local structural screening ->
  clustering/Rfam annotation -> freeze strict split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
from typing import Literal

import pandas as pd
import requests


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.cif"
RCSB_ASSEMBLY1_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}-assembly1.cif"
RFAM_CM_URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz"
RFAM_CLANIN_URL = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.clanin"


@dataclass(frozen=True)
class DownloadRecord:
    pdb_id: str
    url: str
    path: str
    sha256: str
    bytes: int
    assembly: str


def _term(attribute: str, operator: str, value) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def rcsb_query(kind: Literal["protein", "rna", "complex"]) -> dict:
    """Return an entry-level query for broad experimental candidates.

    DNA and NA-hybrid entries are excluded because this project is RNA-only.
    Resolution, lengths, missing atoms, ribosome/spliceosome status and actual
    Protein-RNA contact are checked from downloaded coordinates, not trusted from
    search metadata.
    """
    nodes = [
        _term("rcsb_entry_info.structure_determination_methodology", "exact_match", "experimental"),
        _term("rcsb_entry_info.polymer_entity_count_DNA", "equals", 0),
        _term("rcsb_entry_info.polymer_entity_count_nucleic_acid_hybrid", "equals", 0),
    ]
    if kind == "protein":
        nodes += [
            _term("rcsb_entry_info.polymer_entity_count_protein", "greater", 0),
            _term("rcsb_entry_info.polymer_entity_count_RNA", "equals", 0),
        ]
    elif kind == "rna":
        nodes += [
            _term("rcsb_entry_info.polymer_entity_count_RNA", "greater", 0),
            _term("rcsb_entry_info.polymer_entity_count_protein", "equals", 0),
        ]
    elif kind == "complex":
        nodes += [
            _term("rcsb_entry_info.polymer_entity_count_protein", "greater", 0),
            _term("rcsb_entry_info.polymer_entity_count_RNA", "greater", 0),
        ]
    else:
        raise ValueError(kind)
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,
            "results_content_type": ["experimental"],
        },
    }


def discover_rcsb(kind: Literal["protein", "rna", "complex"], out_tsv: Path, timeout: int = 120) -> pd.DataFrame:
    """Query all matching RCSB entry IDs and persist the exact query and IDs."""
    query = rcsb_query(kind)
    response = requests.post(RCSB_SEARCH_URL, json=query, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    ids = sorted({str(x["identifier"]).upper() for x in payload.get("result_set", [])})
    if not ids:
        raise RuntimeError(f"RCSB query returned no {kind} candidates")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"pdb_id": ids, "source": "RCSB_PDB", "kind": kind})
    frame.to_csv(out_tsv, sep="\t", index=False)
    out_tsv.with_suffix(".query.json").write_text(json.dumps(query, indent=2), encoding="utf-8")
    return frame


def stable_candidate_order(ids: list[str], seed: int) -> list[str]:
    """Deterministic pseudo-random order independent of API result order."""
    return sorted(ids, key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, path: Path, *, timeout: int = 120, retries: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with tmp.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
                tmp.replace(path)
                return
        except Exception as exc:  # network errors are retried, never silently ignored
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


def download_rcsb_candidates(
    candidate_tsv: Path,
    out_dir: Path,
    *,
    seed: int,
    max_candidates: int,
    biological_assembly: bool,
    timeout: int = 120,
) -> pd.DataFrame:
    """Download a deterministic prefix of discovered entries.

    Complexes should use biological assembly 1. Protein/RNA single-chain pools
    may use deposited entry coordinates because the screener extracts one clean
    polymer sample and the baseline conversion later records the chosen chain.
    """
    candidates = pd.read_csv(candidate_tsv, sep="\t")
    ids = stable_candidate_order(candidates["pdb_id"].astype(str).tolist(), seed)[:max_candidates]
    records: list[dict] = []
    failures: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdb_id in ids:
        pdb_id = pdb_id.upper()
        if biological_assembly:
            url = RCSB_ASSEMBLY1_DOWNLOAD.format(pdb_id=pdb_id)
            assembly = "1"
            name = f"{pdb_id}-assembly1.cif"
        else:
            url = RCSB_DOWNLOAD.format(pdb_id=pdb_id)
            assembly = "deposited"
            name = f"{pdb_id}.cif"
        path = out_dir / name
        try:
            if not path.exists() or path.stat().st_size == 0:
                _download(url, path, timeout=timeout)
            records.append(DownloadRecord(
                pdb_id=pdb_id,
                url=url,
                path=str(path.resolve()),
                sha256=_sha256(path),
                bytes=path.stat().st_size,
                assembly=assembly,
            ).__dict__)
        except Exception as exc:
            failures.append({"pdb_id": pdb_id, "url": url, "error": repr(exc)})
    manifest = pd.DataFrame(records)
    manifest.to_csv(out_dir / "download_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "download_failures.tsv", sep="\t", index=False)
    if manifest.empty:
        raise RuntimeError("No structures downloaded successfully")
    return manifest


def download_rfam_resources(out_dir: Path, timeout: int = 300) -> dict[str, str]:
    """Download official current Rfam CM and clan files for strict RNA annotation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, url in [("Rfam.cm.gz", RFAM_CM_URL), ("Rfam.clanin", RFAM_CLANIN_URL)]:
        path = out_dir / name
        if not path.exists() or path.stat().st_size == 0:
            _download(url, path, timeout=timeout)
        outputs[name] = str(path.resolve())
    (out_dir / "checksums.json").write_text(
        json.dumps({name: _sha256(Path(path)) for name, path in outputs.items()}, indent=2),
        encoding="utf-8",
    )
    return outputs
