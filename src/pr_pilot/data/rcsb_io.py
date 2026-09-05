"""RCSB PDB discovery and reproducible download helpers for the pilot.

Discovery is deliberately broad; scientific QC is performed locally in
``screening.py`` and recorded row-by-row. Downloads are concurrent but the final
manifest is sorted by the deterministic candidate rank, so network completion
order can never change the dataset. Partial files are atomic and every successful
file is checksummed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
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
    deterministic_rank: int


def _term(attribute: str, operator: str, value) -> dict:
    return {
        "type": "terminal",
        "service": "text",
        "parameters": {"attribute": attribute, "operator": operator, "value": value},
    }


def rcsb_query(kind: Literal["protein", "rna", "complex"]) -> dict:
    nodes = [
        _term(
            "rcsb_entry_info.structure_determination_methodology",
            "exact_match",
            "experimental",
        ),
        _term("rcsb_entry_info.polymer_entity_count_DNA", "exact_match", 0),
        _term(
            "rcsb_entry_info.polymer_entity_count_nucleic_acid_hybrid",
            "exact_match",
            0,
        ),
    ]
    if kind == "protein":
        nodes += [
            _term("rcsb_entry_info.polymer_entity_count_protein", "greater", 0),
            _term("rcsb_entry_info.polymer_entity_count_RNA", "exact_match", 0),
        ]
    elif kind == "rna":
        nodes += [
            _term("rcsb_entry_info.polymer_entity_count_RNA", "greater", 0),
            _term("rcsb_entry_info.polymer_entity_count_protein", "exact_match", 0),
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


def discover_rcsb(
    kind: Literal["protein", "rna", "complex"],
    out_tsv: Path,
    timeout: int = 120,
) -> pd.DataFrame:
    query = rcsb_query(kind)
    response = requests.post(RCSB_SEARCH_URL, json=query, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    ids = sorted(
        {str(x["identifier"]).upper() for x in payload.get("result_set", [])}
    )
    if not ids:
        raise RuntimeError(f"RCSB query returned no {kind} candidates")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"pdb_id": ids, "source": "RCSB_PDB", "kind": kind})
    frame.to_csv(out_tsv, sep="\t", index=False)
    out_tsv.with_suffix(".query.json").write_text(
        json.dumps(query, indent=2), encoding="utf-8"
    )
    return frame


def stable_candidate_order(ids: list[str], seed: int) -> list[str]:
    return sorted(
        ids, key=lambda x: hashlib.sha256(f"{seed}|{x}".encode()).hexdigest()
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(
    url: str,
    path: Path,
    *,
    timeout: int = 120,
    retries: int = 4,
) -> str:
    """Atomic retrying download; return final response URL for provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(retries):
        try:
            if tmp.exists():
                tmp.unlink()
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
                if not tmp.exists() or tmp.stat().st_size == 0:
                    raise RuntimeError(f"Empty download from {url}")
                tmp.replace(path)
                return str(response.url)
        except Exception as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


def _download_one_rcsb(
    pdb_id: str,
    rank: int,
    out_dir: Path,
    biological_assembly: bool,
    timeout: int,
) -> tuple[dict | None, dict | None]:
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
        final_url = url
        if not path.exists() or path.stat().st_size == 0:
            final_url = _download(url, path, timeout=timeout)
        record = DownloadRecord(
            pdb_id=pdb_id,
            url=final_url,
            path=str(path.resolve()),
            sha256=_sha256(path),
            bytes=path.stat().st_size,
            assembly=assembly,
            deterministic_rank=rank,
        ).__dict__
        return record, None
    except Exception as exc:
        return None, {
            "pdb_id": pdb_id,
            "url": url,
            "deterministic_rank": rank,
            "error": repr(exc),
        }


def download_rcsb_candidates(
    candidate_tsv: Path,
    out_dir: Path,
    *,
    seed: int,
    max_candidates: int,
    biological_assembly: bool,
    timeout: int = 120,
    workers: int = 12,
) -> pd.DataFrame:
    """Download a deterministic candidate prefix using bounded concurrency."""
    if workers < 1:
        raise ValueError("workers must be >=1")
    candidates = pd.read_csv(candidate_tsv, sep="\t")
    ids = stable_candidate_order(
        candidates["pdb_id"].astype(str).tolist(), seed
    )[:max_candidates]
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_rank = {
            pool.submit(
                _download_one_rcsb,
                pdb_id,
                rank,
                out_dir,
                biological_assembly,
                timeout,
            ): rank
            for rank, pdb_id in enumerate(ids)
        }
        for future in as_completed(future_to_rank):
            record, failure = future.result()
            if record is not None:
                records.append(record)
            if failure is not None:
                failures.append(failure)

    # Network completion order never becomes data order.
    records.sort(key=lambda x: int(x["deterministic_rank"]))
    failures.sort(key=lambda x: int(x["deterministic_rank"]))
    manifest = pd.DataFrame(records)
    manifest.to_csv(out_dir / "download_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(failures).to_csv(
        out_dir / "download_failures.tsv", sep="\t", index=False
    )
    metadata = {
        "seed": int(seed),
        "requested_candidates": len(ids),
        "downloaded": len(records),
        "failed": len(failures),
        "workers": int(workers),
        "biological_assembly": bool(biological_assembly),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if manifest.empty:
        raise RuntimeError("No structures downloaded successfully")
    return manifest


def download_rfam_resources(out_dir: Path, timeout: int = 300) -> dict[str, str]:
    """Download official Rfam CURRENT resources with provenance and checksums."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    provenance: dict[str, dict] = {}
    for name, url in [
        ("Rfam.cm.gz", RFAM_CM_URL),
        ("Rfam.clanin", RFAM_CLANIN_URL),
    ]:
        path = out_dir / name
        final_url = url
        if not path.exists() or path.stat().st_size == 0:
            final_url = _download(url, path, timeout=timeout)
        outputs[name] = str(path.resolve())
        provenance[name] = {
            "requested_url": url,
            "resolved_url": final_url,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    metadata = {
        "requested_channel": "CURRENT",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "resources": provenance,
        "note": (
            "If the server does not expose a versioned redirect for CURRENT, the "
            "content SHA256 values are the immutable reproduction identifiers."
        ),
    }
    (out_dir / "rfam_resource_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return outputs
