#!/usr/bin/env python3
"""Run H1-H4 confirmatory statistics after all frozen final evaluations exist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pr_pilot.evaluation.confirmatory import run_confirmatory_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-runs", type=Path, required=True)
    parser.add_argument("--control-runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    result = run_confirmatory_statistics(
        args.component_runs,
        args.control_runs,
        args.out,
        args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
