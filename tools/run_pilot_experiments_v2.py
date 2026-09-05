#!/usr/bin/env python3
"""Compute-tiered orchestrator for the frozen Protein-RNA mini-pilot.

Primary path:
  development 900/100 -> select epoch count -> schedule-prefix refit on full 1000
  -> development-only DeltaC drift audit -> Tier-A final100 for every seed
  -> Tier-B diagnostic battery for the predeclared analysis seed only.

Secondary ablations default to the analysis seed.  ``--secondary-all-seeds`` may
be used when the pilot budget permits, but this choice must be made before
inspecting final-test outcomes.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys

import pandas as pd
import torch
import yaml

from pr_pilot.data.manifest import deterministic_sample
from pr_pilot.evaluation.delta_drift import audit_delta_c_drift
from pr_pilot.training.refit import (
    refit_full_pipeline,
    refit_stage,
    selected_epoch_count,
)
from pr_pilot.training.stages import Stage


PRIMARY_STAGES = [
    Stage.PROTEIN_PRIOR,
    Stage.RNA_PRIOR,
    Stage.GLOBAL_C,
    Stage.DELTA_C,
    Stage.ALPHA,
    Stage.JOINT,
]


def _run(command: list[str], execute: bool) -> None:
    print("$", shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, check=True)


def _module(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", module, *[str(x) for x in args]]


def _cli(*args: object) -> list[str]:
    return _module("pr_pilot.cli", *args)


def _write_yaml(config: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _config_for_seed(base: dict, seed: int, suffix: str = "primary") -> dict:
    config = copy.deepcopy(base)
    config["experiment"]["pilot_seed"] = int(seed)
    config["experiment"]["name"] = (
        f"{base['experiment']['name']}__{suffix}__seed{seed}"
    )
    return config


def _targeted_variant(base: dict, name: str) -> dict:
    config = copy.deepcopy(base)
    if name == "distance_only":
        config["geometry"]["rich_pr_geometry"] = False
    elif name == "no_coordinate_noise":
        config["geometry"]["coordinate_noise_angstrom"] = 0.0
    elif name == "no_graph_stochastic_regularization":
        config["geometry"]["edge_dropout"] = 0.0
        config["geometry"]["pr_edge_dropout"] = 0.0
        config["geometry"]["drop_path"] = 0.0
    else:
        raise ValueError(name)
    return config


def _train_all_development(
    config_path: Path,
    manifests: Path,
    out_dir: Path,
    device: str | None,
    execute: bool,
) -> None:
    command = _cli(
        "train-all",
        "--config",
        config_path,
        "--manifest-root",
        manifests,
        "--out",
        out_dir,
    )
    if device:
        command += ["--device", device]
    _run(command, execute)


def _train_stage(
    stage: Stage,
    config_path: Path,
    train: Path,
    val: Path,
    out_dir: Path,
    init: Path | None,
    device: str | None,
    execute: bool,
) -> Path:
    command = _cli(
        "train",
        "--stage",
        stage.value,
        "--config",
        config_path,
        "--manifest",
        train,
        "--validation",
        val,
        "--out",
        out_dir,
    )
    if init is not None:
        command += ["--init-checkpoint", init]
    if device:
        command += ["--device", device]
    _run(command, execute)
    return out_dir / "best.pt"


def _tier_a(
    config_path: Path,
    checkpoint: Path,
    test: Path,
    out_dir: Path,
    model_name: str,
    device: str | None,
    execute: bool,
) -> None:
    command = _module(
        "pr_pilot.evaluation.tier_a",
        "--config",
        config_path,
        "--checkpoint",
        checkpoint,
        "--manifest",
        test,
        "--out",
        out_dir,
        "--model-name",
        model_name,
    )
    if device:
        command += ["--device", device]
    _run(command, execute)


def _tier_b(
    config_path: Path,
    checkpoint: Path,
    test: Path,
    dev_manifest: Path,
    out_dir: Path,
    device: str | None,
    execute: bool,
) -> None:
    command = _module(
        "pr_pilot.evaluation.full_suite",
        "--config",
        config_path,
        "--checkpoint",
        checkpoint,
        "--test",
        test,
        "--dev",
        dev_manifest,
        "--out",
        out_dir,
    )
    if device:
        command += ["--device", device]
    _run(command, execute)


def _zero_partner_field_checkpoint(
    source: Path, destination: Path, execute: bool
) -> None:
    print(f"[transform] zero interaction field: {source} -> {destination}")
    if not execute:
        return
    payload = torch.load(source, map_location="cpu")
    state = payload["model"]
    for key in [
        "dmicf.global_c.raw",
        "dmicf.delta.out.weight",
        "dmicf.delta.out.bias",
    ]:
        state[key] = torch.zeros_like(state[key])
    payload["stage"] = "dual_structural_prior_partner_blind"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def _fixed_fraction_manifest(
    full_train: Path,
    fraction: float,
    seed: int,
    destination: Path,
    execute: bool,
) -> int:
    frame = pd.read_csv(full_train, sep="\t")
    n = max(1, int(round(len(frame) * fraction)))
    print(f"[data-efficiency] {fraction:.2f} -> {n}/{len(frame)}: {destination}")
    if execute:
        subset = deterministic_sample(frame, n, seed=seed, key="sample_id")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(destination, sep="\t", index=False)
    return n


def _write_runs(rows: list[dict], path: Path, execute: bool) -> None:
    print(f"[run manifest] {path}")
    if execute:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _secondary_seeds(base: dict, all_seeds: bool) -> list[int]:
    if all_seeds:
        return [int(x) for x in base["experiment"]["primary_training_seeds"]]
    return [int(base["evaluation"]["analysis_seed"])]


def run_primary(
    base: dict,
    manifests: Path,
    root: Path,
    device: str | None,
    execute: bool,
) -> list[dict]:
    rows = []
    analysis_seed = int(base["evaluation"]["analysis_seed"])
    for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
        config = _config_for_seed(base, seed)
        config_path = _write_yaml(
            config, root / "configs" / f"primary_seed{seed}.yaml"
        )
        development = root / "training" / "primary_development" / f"seed{seed}"
        _train_all_development(config_path, manifests, development, device, execute)

        refit_dir = root / "training" / "primary_refit_full1000" / f"seed{seed}"
        if execute:
            final_checkpoint = refit_full_pipeline(
                config, manifests, development, refit_dir, device=device
            )
            # Must happen before final100 is touched.
            audit_delta_c_drift(
                config,
                final_checkpoint,
                manifests / "complex_dev.tsv",
                root / "development_audits" / "delta_c_drift" / f"seed{seed}",
                device,
            )
        else:
            final_checkpoint = refit_dir / "joint" / "refit.pt"
            print(
                f"[development audit] DeltaC mean drift -> "
                f"{root / 'development_audits' / 'delta_c_drift' / f'seed{seed}'}"
            )

        evaluation = root / "evaluation" / "primary_refit_full1000" / f"seed{seed}"
        _tier_a(
            config_path,
            final_checkpoint,
            manifests / "complex_test.tsv",
            evaluation,
            "F_joint_full1000",
            device,
            execute,
        )
        rows.append(
            {
                "model": "F_joint_full1000",
                "seed": seed,
                "run_dir": str(evaluation.resolve()),
            }
        )

        if seed == analysis_seed:
            diagnostic = copy.deepcopy(config)
            diagnostic["inference"]["candidates_per_complex"] = int(
                base["evaluation"]["ablation_candidate_budget"]
            )
            diagnostic_path = _write_yaml(
                diagnostic,
                root / "configs" / f"tierB_analysis_seed{seed}.yaml",
            )
            _tier_b(
                diagnostic_path,
                final_checkpoint,
                manifests / "complex_test.tsv",
                manifests / "complex_dev.tsv",
                evaluation / "tier_b_full_battery",
                device,
                execute,
            )
    _write_runs(rows, root / "statistics" / "primary_runs.tsv", execute)
    return rows


def run_component_ladder(
    base: dict,
    manifests: Path,
    root: Path,
    device: str | None,
    execute: bool,
) -> list[dict]:
    rows = []
    for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
        config = _config_for_seed(base, seed)
        config_path = _write_yaml(
            config, root / "configs" / f"primary_seed{seed}.yaml"
        )
        refit_root = root / "training" / "primary_refit_full1000" / f"seed{seed}"
        development = root / "training" / "primary_development" / f"seed{seed}"
        if execute and not (refit_root / "joint" / "refit.pt").exists():
            raise FileNotFoundError(
                "Primary refit must be completed before component ladder: "
                f"{refit_root}"
            )

        scratch_dev = (
            root
            / "training"
            / "component_ladder"
            / f"seed{seed}"
            / "A_scratch_development"
        )
        scratch_dev_ckpt = _train_stage(
            Stage.JOINT,
            config_path,
            manifests / "complex_train.tsv",
            manifests / "complex_val.tsv",
            scratch_dev,
            None,
            device,
            execute,
        )
        scratch_refit = (
            root
            / "training"
            / "component_ladder"
            / f"seed{seed}"
            / "A_scratch_refit_full1000"
        )
        if execute:
            scratch_ckpt = refit_stage(
                config,
                Stage.JOINT,
                manifests / "complex_dev.tsv",
                selected_epoch_count(scratch_dev_ckpt),
                scratch_refit,
                init_checkpoint=None,
                device=device,
            )
        else:
            scratch_ckpt = scratch_refit / "refit.pt"

        prior_zero = (
            root
            / "training"
            / "component_ladder"
            / f"seed{seed}"
            / "B_dual_prior_zero_field.pt"
        )
        _zero_partner_field_checkpoint(
            refit_root / "rna_prior" / "refit.pt", prior_zero, execute
        )
        ladder = {
            "A_scratch_joint_full1000": scratch_ckpt,
            "B_dual_prior_full1000": prior_zero,
            "C_global_C_full1000": refit_root / "global_c" / "refit.pt",
            "D_DeltaC_full1000": refit_root / "delta_c" / "refit.pt",
            "E_alpha_full1000": refit_root / "alpha" / "refit.pt",
            "F_joint_full1000": refit_root / "joint" / "refit.pt",
        }
        for model_name, checkpoint in ladder.items():
            if model_name == "F_joint_full1000":
                evaluation = (
                    root
                    / "evaluation"
                    / "primary_refit_full1000"
                    / f"seed{seed}"
                )
            else:
                evaluation = (
                    root
                    / "evaluation"
                    / "component_ladder"
                    / model_name
                    / f"seed{seed}"
                )
                _tier_a(
                    config_path,
                    checkpoint,
                    manifests / "complex_test.tsv",
                    evaluation,
                    model_name,
                    device,
                    execute,
                )
            rows.append(
                {"model": model_name, "seed": seed, "run_dir": str(evaluation.resolve())}
            )
    _write_runs(rows, root / "statistics" / "component_ladder_runs.tsv", execute)
    if execute:
        _run(
            _module(
                "pr_pilot.evaluation.compare_runs",
                "--runs",
                root / "statistics" / "component_ladder_runs.tsv",
                "--reference",
                "F_joint_full1000",
                "--out",
                root / "statistics" / "component_ladder_exploratory",
                "--bootstrap",
                int(base["evaluation"]["bootstrap_resamples"]),
            ),
            True,
        )
    return rows


def run_data_efficiency(
    base: dict,
    manifests: Path,
    root: Path,
    device: str | None,
    execute: bool,
    all_secondary_seeds: bool,
) -> list[dict]:
    rows = []
    fractions = [float(x) for x in base["evaluation"]["data_efficiency_fractions"]]
    for seed in _secondary_seeds(base, all_secondary_seeds):
        config = _config_for_seed(base, seed, "data_efficiency")
        config_path = _write_yaml(
            config, root / "configs" / f"data_efficiency_seed{seed}.yaml"
        )
        prior = (
            root / "training" / "primary_development" / f"seed{seed}" / "rna_prior" / "best.pt"
        )
        for fraction in fractions:
            percent = int(round(100 * fraction))
            model_name = f"DMICF_complex_train_{percent:03d}pct"
            subset = (
                root
                / "manifests"
                / "data_efficiency"
                / f"seed{seed}_{percent:03d}pct.tsv"
            )
            _fixed_fraction_manifest(
                manifests / "complex_train.tsv",
                fraction,
                seed + percent * 7919,
                subset,
                execute,
            )
            previous = prior
            train_root = (
                root / "training" / "data_efficiency" / model_name / f"seed{seed}"
            )
            for stage in [Stage.GLOBAL_C, Stage.DELTA_C, Stage.ALPHA, Stage.JOINT]:
                previous = _train_stage(
                    stage,
                    config_path,
                    subset,
                    manifests / "complex_val.tsv",
                    train_root / stage.value,
                    previous,
                    device,
                    execute,
                )
            evaluation = (
                root / "evaluation" / "data_efficiency" / model_name / f"seed{seed}"
            )
            _tier_a(
                config_path,
                previous,
                manifests / "complex_test.tsv",
                evaluation,
                model_name,
                device,
                execute,
            )
            rows.append(
                {"model": model_name, "seed": seed, "run_dir": str(evaluation.resolve())}
            )
    _write_runs(rows, root / "statistics" / "data_efficiency_runs.tsv", execute)
    return rows


def run_targeted_ablation(
    base: dict,
    manifests: Path,
    root: Path,
    device: str | None,
    execute: bool,
    all_secondary_seeds: bool,
) -> list[dict]:
    rows = []
    variants = [
        "distance_only",
        "no_coordinate_noise",
        "no_graph_stochastic_regularization",
    ]
    for variant in variants:
        variant_base = _targeted_variant(base, variant)
        for seed in _secondary_seeds(base, all_secondary_seeds):
            config = _config_for_seed(variant_base, seed, variant)
            config_path = _write_yaml(
                config, root / "configs" / f"{variant}_seed{seed}.yaml"
            )
            development = (
                root
                / "training"
                / "targeted_ablation_development"
                / variant
                / f"seed{seed}"
            )
            _train_all_development(
                config_path, manifests, development, device, execute
            )
            refit_root = (
                root
                / "training"
                / "targeted_ablation_refit_full1000"
                / variant
                / f"seed{seed}"
            )
            if execute:
                final_checkpoint = refit_full_pipeline(
                    config, manifests, development, refit_root, device=device
                )
            else:
                final_checkpoint = refit_root / "joint" / "refit.pt"
            evaluation = (
                root / "evaluation" / "targeted_ablation" / variant / f"seed{seed}"
            )
            _tier_a(
                config_path,
                final_checkpoint,
                manifests / "complex_test.tsv",
                evaluation,
                variant,
                device,
                execute,
            )
            rows.append(
                {"model": variant, "seed": seed, "run_dir": str(evaluation.resolve())}
            )
    _write_runs(rows, root / "statistics" / "targeted_ablation_runs.tsv", execute)
    return rows


def _check_manifests(root: Path) -> None:
    required = [
        "protein_pool.tsv",
        "protein_train.tsv",
        "protein_val.tsv",
        "rna_pool.tsv",
        "rna_train.tsv",
        "rna_val.tsv",
        "complex_dev.tsv",
        "complex_train.tsv",
        "complex_val.tsv",
        "complex_test.tsv",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise SystemExit(f"Frozen manifests missing: {missing}")
    for name in ["complex_dev.tsv", "complex_train.tsv", "complex_val.tsv", "complex_test.tsv"]:
        frame = pd.read_csv(root / name, sep="\t", nrows=2)
        for column in [
            "protein_interface_residue_ids",
            "rna_interface_residue_ids",
            "canonical_interface_cutoff_angstrom",
            "canonical_interface_definition",
        ]:
            if column not in frame:
                raise SystemExit(
                    f"{name} lacks {column}; rebuild manifests with canonical heavy-atom interface schema"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("manifests/pilot_v1")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/pilot_experiments")
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=[
            "primary",
            "component_ladder",
            "data_efficiency",
            "targeted_ablation",
            "all",
        ],
        default=["all"],
    )
    parser.add_argument(
        "--secondary-all-seeds",
        action="store_true",
        help="Run secondary data-efficiency/targeted ablations on all primary seeds instead of the frozen analysis seed only.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    execute = bool(args.execute)
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if execute:
        _check_manifests(args.manifest_root)
    requested = set(args.families)
    if "all" in requested:
        requested = {
            "primary",
            "component_ladder",
            "data_efficiency",
            "targeted_ablation",
        }

    args.out.mkdir(parents=True, exist_ok=True)
    plan = {
        "config": str(args.config.resolve()),
        "manifest_root": str(args.manifest_root.resolve()),
        "families": sorted(requested),
        "execute": execute,
        "primary_training_seeds": base["experiment"]["primary_training_seeds"],
        "analysis_seed": int(base["evaluation"]["analysis_seed"]),
        "secondary_all_seeds": bool(args.secondary_all_seeds),
        "primary_refit_policy": (
            "900/100 selects epoch count; full1000 refit replays the same development schedule prefix"
        ),
        "evaluation_policy": (
            "Tier A all primary seeds; Tier B expensive battery only on predeclared analysis seed"
        ),
        "final_test_policy": (
            "immutable 100 complexes; no optimization, threshold tuning or architecture changes"
        ),
    }
    (args.out / "experiment_plan_v2.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )

    if "primary" in requested:
        run_primary(base, args.manifest_root, args.out, args.device, execute)
    if "component_ladder" in requested:
        run_component_ladder(base, args.manifest_root, args.out, args.device, execute)
    if "data_efficiency" in requested:
        run_data_efficiency(
            base,
            args.manifest_root,
            args.out,
            args.device,
            execute,
            args.secondary_all_seeds,
        )
    if "targeted_ablation" in requested:
        run_targeted_ablation(
            base,
            args.manifest_root,
            args.out,
            args.device,
            execute,
            args.secondary_all_seeds,
        )

    if not execute:
        print(
            "\nDRY RUN ONLY. No GPU training started. Execute only after data audit, "
            "baseline preflight and GO/NO-GO checks pass."
        )


if __name__ == "__main__":
    main()
