#!/usr/bin/env python3
"""Freeze final-evaluation budget after development-only runtime profiling.

This tool does not read final-test coordinates or metrics.  It records immutable
hashes of the pilot config, final-test manifest and runtime profile plus the
predeclared Tier-A/Tier-B budgets.  The resulting lock is required by
``run_primary_final_evaluation.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import yaml


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/EVALUATION_PROTOCOL_LOCK.json"))
    args = parser.parse_args()

    profile = json.loads(args.runtime_profile.read_text(encoding="utf-8"))
    if not profile.get("development_only", False):
        raise ValueError("Runtime profile must explicitly state development_only=true")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = cfg["evaluation"]
    payload = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "test_manifest_path": str(args.test_manifest.resolve()),
        "test_manifest_sha256": _sha256(args.test_manifest),
        "runtime_profile_path": str(args.runtime_profile.resolve()),
        "runtime_profile_sha256": _sha256(args.runtime_profile),
        "runtime_profile_summary": profile,
        "primary_training_seeds": [int(x) for x in cfg["experiment"]["primary_training_seeds"]],
        "analysis_seed": int(evaluation["analysis_seed"]),
        "tier_a": {
            "complexes": 100,
            "all_primary_seeds": True,
            "joint_teacher_forced_orders": int(evaluation["joint_teacher_forced_orders"]),
            "scramble_repeats": int(evaluation["scramble_repeats"]),
        },
        "tier_b": {
            "analysis_seed_only": True,
            "ablation_candidate_budget": int(evaluation["ablation_candidate_budget"]),
            "repeated_spir_cycles": int(evaluation["repeated_spir_cycles"]),
        },
        "confirmatory_hypotheses": list(evaluation["primary_hypotheses"]),
        "statistical_unit": "biological complex",
        "multiple_testing": "Holm across H1-H4 only",
        "immutable_after_final100_access": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
