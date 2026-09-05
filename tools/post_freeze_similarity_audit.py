#!/usr/bin/env python3
"""Read-only local-similarity audit between development and final100.

The strict split is frozen before this audit.  MMseqs2 is used only to *report*
nearest local sequence similarity that P30/R80/Rfam clustering may not summarize.
If the audit reveals an unacceptable near duplicate before training, create a new
versioned manifest (pilot_v2); never edit a split after final metrics are known.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import pandas as pd


def _write_fasta(frame: pd.DataFrame, id_col: str, sequence_col: str, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            sample_id = str(getattr(row, id_col))
            seq = str(getattr(row, sequence_col)).replace("|", "").replace(" ", "").upper()
            if not seq:
                raise ValueError(f"Empty {sequence_col} for {sample_id}")
            handle.write(f">{sample_id}\n{seq}\n")


def _easy_search(query: Path, target: Path, output: Path, tmp: Path, threads: int) -> None:
    fields = "query,target,pident,alnlen,qcov,tcov,evalue,bits"
    command = [
        "mmseqs",
        "easy-search",
        str(query),
        str(target),
        str(output),
        str(tmp),
        "--format-output",
        fields,
        "--threads",
        str(threads),
        "--max-seqs",
        "50",
        "-e",
        "1e-3",
    ]
    subprocess.run(command, check=True)


def _summarize(path: Path) -> tuple[pd.DataFrame, dict]:
    cols = ["query", "target", "pident", "alnlen", "qcov", "tcov", "evalue", "bits"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols), {"queries_with_hits": 0}
    frame = pd.read_csv(path, sep="\t", names=cols)
    frame = frame.sort_values(["query", "bits"], ascending=[True, False])
    best = frame.groupby("query", as_index=False).first()
    summary = {
        "queries_with_hits": int(best.query.nunique()),
        "max_identity_percent": float(best.pident.max()),
        "max_query_coverage_percent": float(best.qcov.max()),
        "max_target_coverage_percent": float(best.tcov.max()),
        "near_duplicate_like_hits_identity_ge_80_qcov_ge_80": int(
            ((best.pident >= 80) & (best.qcov >= 80)).sum()
        ),
    }
    return best, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if shutil.which("mmseqs") is None:
        raise SystemExit("mmseqs executable not found; install MMseqs2 before this read-only audit")

    dev = pd.read_csv(args.manifest_root / "complex_dev.tsv", sep="\t")
    test = pd.read_csv(args.manifest_root / "complex_test.tsv", sep="\t")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "policy": "read-only pre-training audit; never mutate split after final metrics",
        "development_complexes": int(len(dev)),
        "final_complexes": int(len(test)),
        "results": {},
    }

    with tempfile.TemporaryDirectory(prefix="pr_similarity_") as temp_dir:
        temp = Path(temp_dir)
        for polymer, sequence_col in [
            ("protein", "protein_sequence"),
            ("rna", "rna_sequence"),
        ]:
            query = temp / f"test_{polymer}.fa"
            target = temp / f"dev_{polymer}.fa"
            raw = args.out / f"{polymer}_test_vs_dev_mmseqs.tsv"
            _write_fasta(test, "sample_id", sequence_col, query)
            _write_fasta(dev, "sample_id", sequence_col, target)
            _easy_search(query, target, raw, temp / f"tmp_{polymer}", args.threads)
            best, summary = _summarize(raw)
            best.to_csv(args.out / f"{polymer}_best_local_hit.tsv", sep="\t", index=False)
            report["results"][polymer] = summary

    (args.out / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
