#!/usr/bin/env python3
"""Download a date-bounded, Protein/RNA RCSB candidate pool for blind screening.

This tool only performs candidate retrieval. It never selects a blind test or
reads model outputs. Existing local CIFs and all IDs present in the supplied
manifests are excluded before downloading.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
FILE_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def search_payload(release_from: str) -> dict:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                        "operator": "greater_or_equal",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater_or_equal",
                        "value": release_from,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match",
                        "value": "experimental",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "results_verbosity": "compact",
        },
    }


def fetch_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def manifest_ids(roots: list[Path]) -> set[str]:
    ids: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.tsv"):
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle, delimiter="\t"):
                        value = str(row.get("pdb_id", "")).strip().upper()
                        if value:
                            ids.add(value)
                        sample = str(row.get("sample_id", ""))
                        if sample:
                            ids.add(sample.split("-", 1)[0].split(":", 1)[0].upper())
            except (OSError, UnicodeError):
                continue
    return ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(pdb_id: str, out_dir: Path, retries: int) -> dict:
    target = out_dir / f"{pdb_id.lower()}.cif"
    if target.exists() and target.stat().st_size > 0:
        return {"pdb_id": pdb_id, "status": "existing", "bytes": target.stat().st_size, "sha256": sha256_file(target)}
    part = target.with_suffix(".cif.part")
    url = FILE_URL.format(pdb_id=pdb_id)
    last_error = ""
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "PR-pilot-blind-candidate-downloader/1.0"})
            with urlopen(request, timeout=120) as response, part.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if part.stat().st_size == 0:
                raise ValueError("empty CIF response")
            part.replace(target)
            return {"pdb_id": pdb_id, "status": "downloaded", "bytes": target.stat().st_size, "sha256": sha256_file(target)}
        except (HTTPError, URLError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            if part.exists():
                part.unlink()
            time.sleep(min(30.0, 2.0 ** attempt))
    return {"pdb_id": pdb_id, "status": "error", "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--manifest-root", type=Path, action="append", required=True)
    parser.add_argument("--release-from", default="2025-01-01")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    out = args.out
    raw_out = out / "raw_cif"
    raw_out.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    existing = set()
    for root in args.raw_root:
        if root.exists():
            existing.update(path.stem.split("-")[0].upper() for path in root.rglob("*.cif"))
    existing.update(manifest_ids(args.manifest_root))

    payload = search_payload(args.release_from)
    response = fetch_json(SEARCH_URL, payload)
    all_ids = sorted({str(x).upper() for x in response.get("result_set", [])})
    candidates = [pdb_id for pdb_id in all_ids if pdb_id not in existing]
    (out / "search_query.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "search_response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    (out / "candidate_ids.tsv").write_text(
        "pdb_id\n" + "\n".join(candidates) + "\n", encoding="utf-8"
    )

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(download_one, pdb_id, raw_out, int(args.retries)) for pdb_id in candidates]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: row["pdb_id"])
    with (out / "download_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in records for key in row}), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "release_from": args.release_from,
        "search_total": int(response.get("total_count", len(all_ids))),
        "search_ids": len(all_ids),
        "existing_excluded": len(existing),
        "download_candidates": len(candidates),
        "downloaded_or_existing": sum(row["status"] in {"downloaded", "existing"} for row in records),
        "errors": sum(row["status"] == "error" for row in records),
        "workers": int(args.workers),
        "raw_cif": str(raw_out),
    }
    (out / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
