#!/usr/bin/env python3
"""Count strict blind-screen outcomes over the complete local raw CIF pool.

Read-only diagnostic: it does not write manifests, annotations, or metrics.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import collections
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pr_pilot.data.screening import ScreenConfig, screen_file  # noqa: E402

RAW = Path(r"H:\2026try\9.7PRI-General100k\data\raw_downloads\sources\rcsb_pdb\experimental_structures\raw_cif")
OUT = Path(r"I:\PR_PILOT_SCIENTIFIC\20260919\resplit_e0\screen")
CFG = ScreenConfig(
    protein_min_length=40,
    protein_max_length=2000,
    rna_min_length=10,
    rna_max_length=500,
    max_total_tokens=1000,
    max_resolution_angstrom=4.0,
    allow_nmr_without_resolution=True,
    apply_resolution_method_filter=False,
    interface_contact_angstrom=6.0,
    min_interfacial_residue_pairs=3,
    max_interface_missing_fraction=0.10,
    exclude_large_rnp_keywords=True,
)


def one(path: Path) -> tuple[str, dict | None, str]:
    record, reason = screen_file(path, "complex", CFG)
    return str(path), record, reason


def main() -> None:
    files = sorted(RAW.rglob("*.cif"))
    reasons: collections.Counter[str] = collections.Counter()
    eligible: list[dict] = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(one, path) for path in files]
        for future in as_completed(futures):
            path, record, reason = future.result()
            if record is None:
                reasons[reason] += 1
            else:
                eligible.append(record)
    OUT.mkdir(parents=True, exist_ok=True)
    pd = __import__("pandas")
    frame = pd.DataFrame(sorted(eligible, key=lambda row: str(row["sample_id"])))
    frame.to_csv(OUT / "complex_eligible.tsv", sep="\t", index=False)
    (OUT / "summary.json").write_text(json.dumps({"raw_files": len(files), "eligible": len(eligible), "rejected": sum(reasons.values()), "reasons": dict(reasons)}, indent=2), encoding="utf-8")
    print(json.dumps({"raw_files": len(files), "eligible": len(eligible), "rejected": sum(reasons.values()), "reasons": dict(reasons), "eligible_path": str(OUT / "complex_eligible.tsv")}, indent=2))


if __name__ == "__main__":
    main()
