#!/usr/bin/env python3
"""Freeze final-evaluation protocol after development-only profiling.

The lock covers not only config/test/budget, but also the exact primary stage
checkpoints and H2 internal-control checkpoints. Final100 therefore cannot be used
to swap a more favorable C/alpha/joint/control checkpoint after the fact.
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


def _primary_checkpoint_inventory(training_ready: dict) -> list[dict]:
    inventory = []
    for run in training_ready["runs"]:
        seed = int(run["seed"])
        refit = Path(run["refit_dir"])
        for stage in ["rna_prior", "global_c", "delta_c", "alpha", "joint"]:
            path = refit / stage / "refit.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            inventory.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
    return inventory


def _control_checkpoint_inventory(control_ready: dict) -> list[dict]:
    inventory = []
    for run in control_ready["runs"]:
        path = Path(run["checkpoint"])
        if not path.exists():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if digest != str(run.get("checkpoint_sha256", "")):
            raise RuntimeError(f"control checkpoint digest differs from training-ready record: {path}")
        inventory.append(
            {
                "model": str(run["model"]),
                "seed": int(run["seed"]),
                "path": str(path.resolve()),
                "sha256": digest,
            }
        )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--training-ready", type=Path, required=True)
    parser.add_argument("--control-training-ready", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/EVALUATION_PROTOCOL_LOCK.json"))
    args = parser.parse_args()

    profile = json.loads(args.runtime_profile.read_text(encoding="utf-8"))
    if not profile.get("development_only", False):
        raise ValueError("Runtime profile must explicitly state development_only=true")
    training_ready = json.loads(args.training_ready.read_text(encoding="utf-8"))
    controls_ready = json.loads(args.control_training_ready.read_text(encoding="utf-8"))
    if training_ready.get("status") != "PRIMARY_TRAINING_COMPLETE_FINAL_TEST_STILL_LOCKED":
        raise ValueError("invalid primary training-ready status")
    if controls_ready.get("status") != "CONTROL_TRAINING_COMPLETE_FINAL_TEST_STILL_LOCKED":
        raise ValueError("invalid control training-ready status")

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
        "training_ready_path": str(args.training_ready.resolve()),
        "training_ready_sha256": _sha256(args.training_ready),
        "control_training_ready_path": str(args.control_training_ready.resolve()),
        "control_training_ready_sha256": _sha256(args.control_training_ready),
        "primary_checkpoint_inventory": _primary_checkpoint_inventory(training_ready),
        "control_checkpoint_inventory": _control_checkpoint_inventory(controls_ready),
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
