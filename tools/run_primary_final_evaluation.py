#!/usr/bin/env python3
"""Open the immutable final100 only after an evaluation protocol is frozen.

The script verifies config/test-manifest hashes from EVALUATION_PROTOCOL_LOCK.json,
then runs Tier A on every primary refit seed and Tier B only on the predeclared
analysis seed. It never trains or selects a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from pr_pilot.evaluation.full_suite import run_full_suite
from pr_pilot.evaluation.tier_a import evaluate_tier_a


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--training-ready", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/pilot_experiments/evaluation"))
    parser.add_argument("--device")
    args = parser.parse_args()

    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if _sha256(args.config) != lock["config_sha256"]:
        raise RuntimeError("pilot config changed after evaluation protocol lock")
    if _sha256(args.test_manifest) != lock["test_manifest_sha256"]:
        raise RuntimeError("final-test manifest changed after evaluation protocol lock")
    ready = json.loads(args.training_ready.read_text(encoding="utf-8"))
    if ready.get("status") != "PRIMARY_TRAINING_COMPLETE_FINAL_TEST_STILL_LOCKED":
        raise ValueError("training-ready file is not a valid pre-final status record")

    expected_seeds = [int(x) for x in lock["primary_training_seeds"]]
    runs = {int(row["seed"]): row for row in ready["runs"]}
    if sorted(runs) != sorted(expected_seeds):
        raise RuntimeError("training-ready seeds do not match frozen protocol seeds")

    args.out.mkdir(parents=True, exist_ok=True)
    run_manifest = []
    for seed in expected_seeds:
        row = runs[seed]
        seed_cfg = yaml.safe_load(Path(row["config"]).read_text(encoding="utf-8"))
        if int(seed_cfg["experiment"]["pilot_seed"]) != seed:
            raise RuntimeError(f"seed config mismatch for {seed}")
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        seed_out = args.out / "primary_refit_full1000" / f"seed{seed}"
        evaluate_tier_a(
            seed_cfg,
            checkpoint,
            args.test_manifest,
            seed_out,
            args.device,
            "F_joint_full1000",
        )
        run_manifest.append(
            {"model": "F_joint_full1000", "seed": seed, "run_dir": str(seed_out.resolve())}
        )

        if seed == int(lock["analysis_seed"]):
            tier_b_cfg = dict(seed_cfg)
            tier_b_cfg["inference"] = dict(seed_cfg["inference"])
            tier_b_cfg["inference"]["candidates_per_complex"] = int(
                lock["tier_b"]["ablation_candidate_budget"]
            )
            run_full_suite(
                tier_b_cfg,
                checkpoint,
                args.test_manifest,
                seed_out / "tier_b_full_battery",
                dev_manifest=args.dev_manifest,
                device=args.device,
            )

    pd.DataFrame(run_manifest).to_csv(
        args.out / "primary_runs.tsv", sep="\t", index=False
    )
    summary = {
        "status": "FINAL100_PRIMARY_EVALUATION_COMPLETE",
        "protocol_lock": str(args.protocol_lock.resolve()),
        "protocol_lock_sha256": _sha256(args.protocol_lock),
        "test_manifest_sha256": _sha256(args.test_manifest),
        "runs": run_manifest,
        "training_or_selection_performed": False,
    }
    (args.out / "FINAL_EVALUATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
