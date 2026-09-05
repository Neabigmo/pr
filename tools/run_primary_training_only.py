#!/usr/bin/env python3
"""Train/refit primary DM-ICF seeds without ever opening complex_test.tsv.

This is the authoritative formal-training entrypoint.  It runs development
selection, schedule-prefix full1000 refit and development-only DeltaC drift audit.
Final100 evaluation is intentionally impossible from this script.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import yaml

from pr_pilot.evaluation.delta_drift import audit_delta_c_drift
from pr_pilot.training.refit import refit_full_pipeline


def _write_config(base: dict, seed: int, path: Path) -> dict:
    cfg = copy.deepcopy(base)
    cfg["experiment"]["pilot_seed"] = int(seed)
    cfg["experiment"]["name"] = f"{base['experiment']['name']}__primary__seed{seed}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests/pilot_v1"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/pilot_experiments"))
    parser.add_argument("--device")
    args = parser.parse_args()

    # Deliberately verify only development/prior manifests. This script must not
    # read final100 content even for a convenience check.
    required = [
        "protein_train.tsv", "protein_val.tsv", "protein_pool.tsv",
        "rna_train.tsv", "rna_val.tsv", "rna_pool.tsv",
        "complex_train.tsv", "complex_val.tsv", "complex_dev.tsv",
    ]
    missing = [name for name in required if not (args.manifest_root / name).exists()]
    if missing:
        raise SystemExit(f"Missing training manifests: {missing}")

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
        config_path = args.out / "configs" / f"primary_seed{seed}.yaml"
        cfg = _write_config(base, seed, config_path)
        development = args.out / "training" / "primary_development" / f"seed{seed}"
        command = [
            sys.executable, "-m", "pr_pilot.cli", "train-all",
            "--config", str(config_path),
            "--manifest-root", str(args.manifest_root),
            "--out", str(development),
        ]
        if args.device:
            command += ["--device", args.device]
        subprocess.run(command, check=True)

        refit = args.out / "training" / "primary_refit_full1000" / f"seed{seed}"
        final_checkpoint = refit_full_pipeline(
            cfg, args.manifest_root, development, refit, device=args.device
        )
        drift_dir = args.out / "development_audits" / "delta_c_drift" / f"seed{seed}"
        audit_delta_c_drift(
            cfg,
            final_checkpoint,
            args.manifest_root / "complex_dev.tsv",
            drift_dir,
            args.device,
        )
        rows.append(
            {
                "seed": seed,
                "config": str(config_path.resolve()),
                "development_dir": str(development.resolve()),
                "refit_dir": str(refit.resolve()),
                "checkpoint": str(final_checkpoint.resolve()),
                "delta_drift_audit": str(drift_dir.resolve()),
                "final_test_read": False,
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PRIMARY_TRAINING_COMPLETE_FINAL_TEST_STILL_LOCKED",
        "runs": rows,
        "next": "profile evaluation budget on complex_val, then create evaluation protocol lock",
    }
    (args.out / "PRIMARY_TRAINING_READY.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
