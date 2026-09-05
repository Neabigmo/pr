#!/usr/bin/env python3
"""Safe final100 wrapper around evaluate_official_baselines.evaluate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.evaluate_official_baselines import evaluate


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--prepared-holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--third-party-root", type=Path, default=Path("third_party/checkouts"))
    parser.add_argument("--na-probability-samples", type=int, default=64)
    args = parser.parse_args()

    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if _sha256(args.test_manifest) != lock["test_manifest_sha256"]:
        raise RuntimeError("test manifest differs from frozen evaluation protocol")
    result = evaluate(
        args.repo_root.resolve(),
        args.baseline_summary.resolve(),
        args.prepared_holdout.resolve(),
        args.out.resolve(),
        args.third_party_root.resolve(),
        args.na_probability_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
