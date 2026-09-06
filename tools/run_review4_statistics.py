#!/usr/bin/env python3
"""Run strict five-hypothesis inference on already-exported per-complex effects.

Does not generate or invent effects, train models, or certify biological results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from pr_pilot.evaluation.confirmatory import analyze_confirmatory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=20260905)
    parser.add_argument("--data-kind", choices=["synthetic", "experimental"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; preserve prior results and choose a new path")
    effects = pd.read_csv(args.effects, dtype={"sample_id": str, "group_id": str})
    roster = pd.read_csv(args.roster, dtype={"sample_id": str, "group_id": str})
    try:
        report = analyze_confirmatory(effects, roster, args.seeds,
                                     resamples=args.resamples, random_seed=args.random_seed)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    report["declared_data_kind"] = args.data_kind
    report["not_a_biological_result"] = args.data_kind == "synthetic"
    report["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (args.effects, args.roster)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
