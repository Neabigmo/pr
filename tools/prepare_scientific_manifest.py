#!/usr/bin/env python3
"""Create the length-constrained scientific manifest without touching inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pr_pilot.adapter_pilot.scientific import ScientificProtocol, validate_length_bounds


def _filter_complex(frame: pd.DataFrame, protocol: ScientificProtocol) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.to_numeric(frame["protein_length"], errors="coerce")
    r = pd.to_numeric(frame["rna_length"], errors="coerce")
    keep = p.between(protocol.protein_min_length, protocol.protein_max_length, inclusive="both") & r.between(protocol.rna_min_length, protocol.rna_max_length, inclusive="both")
    return frame.loc[keep].copy(), frame.loc[~keep].copy()


def _filter_single(frame: pd.DataFrame, molecule: str, protocol: ScientificProtocol) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "length" not in frame:
        raise ValueError(f"{molecule} manifest has no length column")
    lower, upper = (protocol.protein_min_length, protocol.protein_max_length) if molecule == "protein" else (protocol.rna_min_length, protocol.rna_max_length)
    length = pd.to_numeric(frame["length"], errors="coerce")
    keep = length.between(lower, upper, inclusive="both")
    return frame.loc[keep].copy(), frame.loc[~keep].copy()


def prepare(source: Path, target: Path) -> dict:
    protocol = ScientificProtocol()
    target.mkdir(parents=True, exist_ok=False)
    audit: dict = {"source": str(source), "target": str(target), "protocol": protocol.as_dict(), "excluded": {}}
    for split in ("train", "val", "dev", "pool", "test"):
        path = source / f"complex_{split}.tsv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep="\t")
        filtered, excluded = _filter_complex(frame, protocol)
        if split == "test" and len(excluded):
            raise ValueError("the frozen holdout violates the new length protocol; refusing to rewrite it")
        filtered.to_csv(target / path.name, sep="\t", index=False)
        audit["excluded"][split] = excluded[["sample_id", "protein_length", "rna_length"]].to_dict(orient="records")
    for molecule in ("protein", "rna"):
        for split in ("train", "val", "pool"):
            path = source / f"{molecule}_{split}.tsv"
            if not path.exists():
                continue
            frame = pd.read_csv(path, sep="\t")
            filtered, excluded = _filter_single(frame, molecule, protocol)
            filtered.to_csv(target / path.name, sep="\t", index=False)
            audit["excluded"][f"{molecule}_{split}"] = excluded[["sample_id", "length"]].to_dict(orient="records")
    audit["counts"] = {}
    for path in sorted(target.glob("*.tsv")):
        audit["counts"][path.name] = int(len(pd.read_csv(path, sep="\t")))
    (target / "length_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.target), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
