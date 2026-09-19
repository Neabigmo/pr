#!/usr/bin/env python3
"""Run the locked single-seed E0 CV on the retrospective resplit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import run_rna_entropy_balance as runner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=18)
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f"unknown arguments: {unknown}")
    runner.SEED = int(args.seed)
    runner.MANIFESTS = Path(args.manifests)
    runner.DEV_CACHE = Path(args.cache_root)
    runner.OUT = Path(args.out)
    runner.EXPERIMENTS = {"E0_resplit": {"rna_gate_mode": "baseline", "balanced": False}}
    sys.argv = [sys.argv[0], "--device", args.device, "--fold-workers", str(args.fold_workers), "--epochs", str(args.epochs), "--patience", str(args.patience)]
    if args.resume:
        sys.argv.append("--resume")
    if args.force:
        sys.argv.append("--force")
    runner.main()


if __name__ == "__main__":
    main()
