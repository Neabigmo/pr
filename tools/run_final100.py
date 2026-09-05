#!/usr/bin/env python3
"""Run the frozen final-100 evaluation under a pre-registered compute budget.

Tier A (all primary seeds, all 100 complexes)
----------------------------------------------
Runs checkpoint-level conditional/joint metrics and the confirmatory model
interventions that do not require expensive candidate generation.

Tier B (one predeclared analysis seed)
--------------------------------------
Runs the full mechanistic/robustness/inference suite. Candidate-based order/SPIR
ablations use the smaller predeclared ablation budget, while the primary mixed
joint candidate set keeps the full 64-candidate budget.

This script never changes architecture or tuning based on final-test results.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _run(cmd: list[str], execute: bool) -> None:
    print("$", " ".join(map(str, cmd)), flush=True)
    if execute:
        subprocess.run(cmd, check=True)


def _checkpoint(root: Path, seed: int) -> Path:
    path = root / f"seed{seed}" / "joint" / "refit.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--refit-root", type=Path, required=True, help="Directory containing seedX/joint/refit.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = [int(x) for x in cfg["experiment"]["primary_training_seeds"]]
    analysis_seed = int(cfg["evaluation"].get("full_diagnostic_seed", cfg["experiment"]["analysis_seed"]))
    if analysis_seed not in seeds:
        raise ValueError("full_diagnostic_seed must be one of primary_training_seeds")
    test = args.manifests / "complex_test.tsv"
    dev = args.manifests / "complex_dev.tsv"
    if not test.exists() or not dev.exists():
        raise FileNotFoundError("Frozen complex_test.tsv and complex_dev.tsv are required")
    args.out.mkdir(parents=True, exist_ok=True)

    run_manifest = []
    for seed in seeds:
        checkpoint = _checkpoint(args.refit_root, seed)
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["experiment"]["pilot_seed"] = seed
        cfg_path = args.out / "configs" / f"seed{seed}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(seed_cfg, sort_keys=False), encoding="utf-8")
        core_out = args.out / "tierA_all_seeds" / f"seed{seed}"
        cmd = [
            sys.executable, "-m", "pr_pilot.cli", "evaluate",
            "--config", str(cfg_path),
            "--checkpoint", str(checkpoint),
            "--manifest", str(test),
            "--out", str(core_out),
            "--model-name", f"DMICF_full1000_seed{seed}",
        ]
        if args.device:
            cmd += ["--device", args.device]
        _run(cmd, args.execute)
        run_manifest.append({"tier": "A", "seed": seed, "checkpoint": str(checkpoint), "output": str(core_out)})

    # Tier B only on a predeclared seed. Use 16 candidates for order/SPIR/robustness
    # ablations so the diagnostic battery does not explode combinatorially.
    checkpoint = _checkpoint(args.refit_root, analysis_seed)
    diagnostic_cfg = copy.deepcopy(cfg)
    diagnostic_cfg["experiment"]["pilot_seed"] = analysis_seed
    diagnostic_cfg["inference"]["candidates_per_complex"] = int(cfg["inference"].get("ablation_candidates_per_complex", 16))
    diagnostic_path = args.out / "configs" / f"diagnostic_seed{analysis_seed}.yaml"
    diagnostic_path.write_text(yaml.safe_dump(diagnostic_cfg, sort_keys=False), encoding="utf-8")
    diagnostic_out = args.out / "tierB_diagnostic" / f"seed{analysis_seed}"
    cmd = [
        sys.executable, "-m", "pr_pilot.evaluation.full_suite",
        "--config", str(diagnostic_path),
        "--checkpoint", str(checkpoint),
        "--test", str(test),
        "--dev", str(dev),
        "--out", str(diagnostic_out),
    ]
    if args.device:
        cmd += ["--device", args.device]
    _run(cmd, args.execute)
    run_manifest.append({"tier": "B", "seed": analysis_seed, "checkpoint": str(checkpoint), "output": str(diagnostic_out), "candidate_budget": diagnostic_cfg["inference"]["candidates_per_complex"]})

    # Primary generation remains separate and keeps the full candidate budget.
    primary_cfg = copy.deepcopy(cfg)
    primary_cfg["experiment"]["pilot_seed"] = analysis_seed
    primary_cfg_path = args.out / "configs" / f"primary_generation_seed{analysis_seed}.yaml"
    primary_cfg_path.write_text(yaml.safe_dump(primary_cfg, sort_keys=False), encoding="utf-8")
    generation_out = args.out / "primary_joint_generation" / f"seed{analysis_seed}"
    cmd = [
        sys.executable, "-m", "pr_pilot.cli", "sample-joint",
        "--config", str(primary_cfg_path),
        "--checkpoint", str(checkpoint),
        "--manifest", str(test),
        "--out", str(generation_out),
    ]
    if args.device:
        cmd += ["--device", args.device]
    _run(cmd, args.execute)
    run_manifest.append({"tier": "primary_generation", "seed": analysis_seed, "checkpoint": str(checkpoint), "output": str(generation_out), "candidate_budget": int(cfg["inference"]["candidates_per_complex"])})

    (args.out / "FINAL100_EVALUATION_PLAN.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print("Execution mode:", "EXECUTE" if args.execute else "DRY-RUN")


if __name__ == "__main__":
    main()
