#!/usr/bin/env python3
"""Orchestrate the frozen PR mini-pilot without final-test feedback.

Primary models use 900/100 only for model-selection, then replay the selected
schedule prefix from random initialization on the full 1,000 development samples.
The heavyweight mechanistic/candidate-generation battery is restricted to the
predeclared analysis seed; all three primary seeds still receive the inexpensive
core final-100 evaluation used for seed aggregation.
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from pr_pilot.data.manifest import deterministic_sample
from pr_pilot.training.refit import (
    refit_full_pipeline,
    refit_stage,
    selected_epoch_count,
    selected_schedule_horizon,
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


def _print_command(command: list[str]) -> None:
    print("$", shlex.join(command), flush=True)


def _run(command: list[str], execute: bool) -> None:
    _print_command(command)
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
    config["experiment"]["name"] = f"{base['experiment']['name']}__{suffix}__seed{seed}"
    return config


def _scratch_config(primary: dict, seed: int) -> dict:
    config = _config_for_seed(primary, seed, suffix="scratch_joint")
    config["training_stages"]["joint"]["unfreezing_mode"] = "all_trainable_from_start"
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
        raise ValueError(f"Unknown targeted variant: {name}")
    return config


def _train_all_development(
    config_path: Path,
    manifest_root: Path,
    out_dir: Path,
    device: str | None,
    execute: bool,
) -> None:
    command = _cli(
        "train-all", "--config", config_path, "--manifest-root", manifest_root, "--out", out_dir
    )
    if device:
        command += ["--device", device]
    _run(command, execute)


def _train_one_development_stage(
    stage: Stage,
    config_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    out_dir: Path,
    init_checkpoint: Path | None,
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
        train_manifest,
        "--validation",
        val_manifest,
        "--out",
        out_dir,
    )
    if init_checkpoint is not None:
        command += ["--init-checkpoint", init_checkpoint]
    if device:
        command += ["--device", device]
    _run(command, execute)
    return out_dir / "best.pt"


def _core_eval(
    config_path: Path,
    checkpoint: Path,
    test_manifest: Path,
    output: Path,
    model_name: str,
    device: str | None,
    execute: bool,
) -> None:
    command = _cli(
        "evaluate",
        "--config",
        config_path,
        "--checkpoint",
        checkpoint,
        "--manifest",
        test_manifest,
        "--out",
        output / "core",
        "--model-name",
        model_name,
    )
    if device:
        command += ["--device", device]
    _run(command, execute)


def _full_suite(
    config_path: Path,
    checkpoint: Path,
    test_manifest: Path,
    dev_manifest: Path,
    output: Path,
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
        test_manifest,
        "--dev",
        dev_manifest,
        "--out",
        output,
    )
    if device:
        command += ["--device", device]
    _run(command, execute)


def _delta_drift(
    config_path: Path,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    device: str | None,
    execute: bool,
) -> None:
    command = _module(
        "pr_pilot.evaluation.field_audit",
        "--config",
        config_path,
        "--checkpoint",
        checkpoint,
        "--manifest",
        manifest,
        "--out",
        output,
    )
    if device:
        command += ["--device", device]
    _run(command, execute)


def _zero_partner_field_checkpoint(source: Path, destination: Path, execute: bool) -> None:
    print(f"[transform] zero interaction field: {source} -> {destination}")
    if not execute:
        return
    payload = torch.load(source, map_location="cpu")
    state = payload["model"]
    for key in ["dmicf.global_c.raw", "dmicf.delta.out.weight", "dmicf.delta.out.bias"]:
        state[key] = torch.zeros_like(state[key])
    payload["stage"] = "dual_structural_prior_partner_blind"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def _fixed_fraction_manifest(
    full_train: Path,
    fraction: float,
    ranking_seed: int,
    destination: Path,
    execute: bool,
) -> int:
    """Create nested data-efficiency subsets by using one ranking seed at all fractions."""
    frame = pd.read_csv(full_train, sep="\t")
    n = max(1, int(round(len(frame) * fraction)))
    print(f"[data-efficiency] {fraction:.2f} -> {n}/{len(frame)}: {destination}")
    if execute:
        subset = deterministic_sample(frame, n, seed=ranking_seed, key="sample_id")
        destination.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(destination, sep="\t", index=False)
    return n


def _compare_runs(
    run_rows: list[dict],
    path: Path,
    reference: str,
    output: Path,
    bootstrap: int,
    execute: bool,
) -> None:
    frame = pd.DataFrame(run_rows)
    print(f"[statistics manifest] {path}")
    if not execute:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False)
    _run(
        _module(
            "pr_pilot.evaluation.compare_runs",
            "--runs",
            path,
            "--reference",
            reference,
            "--out",
            output,
            "--bootstrap",
            bootstrap,
        ),
        execute=True,
    )


def run_primary(
    base: dict,
    manifests: Path,
    root: Path,
    device: str | None,
    execute: bool,
) -> list[dict]:
    rows = []
    analysis_seed = int(base["evaluation"].get("analysis_seed", base["experiment"]["pilot_seed"]))
    for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
        config = _config_for_seed(base, seed)
        config_path = _write_yaml(config, root / "configs" / f"primary_seed{seed}.yaml")
        development = root / "training" / "primary_development" / f"seed{seed}"
        _train_all_development(config_path, manifests, development, device, execute)

        refit_dir = root / "training" / "primary_refit_full1000" / f"seed{seed}"
        print(
            f"[final refit] seed={seed}: 1000 Protein + 1000 RNA + 1000 complex development samples"
        )
        final_checkpoint = (
            refit_full_pipeline(config, manifests, development, refit_dir, device=device)
            if execute
            else refit_dir / "joint" / "refit.pt"
        )

        evaluation = root / "evaluation" / "primary_refit_full1000" / f"seed{seed}"
        if seed == analysis_seed:
            _full_suite(
                config_path,
                final_checkpoint,
                manifests / "complex_test.tsv",
                manifests / "complex_dev.tsv",
                evaluation,
                device,
                execute,
            )
            _delta_drift(
                config_path,
                final_checkpoint,
                manifests / "complex_dev.tsv",
                evaluation / "delta_c_drift_dev",
                device,
                execute,
            )
        else:
            _core_eval(
                config_path,
                final_checkpoint,
                manifests / "complex_test.tsv",
                evaluation,
                "DMICF_full1000",
                device,
                execute,
            )
        rows.append(
            {"model": "DMICF_full1000", "seed": seed, "run_dir": str(evaluation.resolve())}
        )
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
        primary_config_path = _write_yaml(
            config, root / "configs" / f"primary_seed{seed}.yaml"
        )
        refit_root = root / "training" / "primary_refit_full1000" / f"seed{seed}"

        # Scratch control: all random-initialized parameters are trainable from step 0.
        scratch_cfg = _scratch_config(base, seed)
        scratch_config_path = _write_yaml(
            scratch_cfg, root / "configs" / f"scratch_joint_seed{seed}.yaml"
        )
        scratch_dev = (
            root / "training" / "component_ladder" / f"seed{seed}" / "A_scratch_development"
        )
        scratch_dev_checkpoint = _train_one_development_stage(
            Stage.JOINT,
            scratch_config_path,
            manifests / "complex_train.tsv",
            manifests / "complex_val.tsv",
            scratch_dev,
            None,
            device,
            execute,
        )
        scratch_refit_dir = (
            root / "training" / "component_ladder" / f"seed{seed}" / "A_scratch_refit_full1000"
        )
        if execute:
            scratch_epochs = selected_epoch_count(scratch_dev_checkpoint)
            horizon = selected_schedule_horizon(scratch_dev_checkpoint, scratch_cfg, Stage.JOINT)
            scratch_checkpoint = refit_stage(
                scratch_cfg,
                Stage.JOINT,
                manifests / "complex_dev.tsv",
                scratch_epochs,
                scratch_refit_dir,
                init_checkpoint=None,
                device=device,
                schedule_horizon_epochs=horizon,
            )
        else:
            scratch_checkpoint = scratch_refit_dir / "refit.pt"

        prior_source = refit_root / "rna_prior" / "refit.pt"
        prior_zero = (
            root / "training" / "component_ladder" / f"seed{seed}" / "B_dual_prior_zero_field.pt"
        )
        _zero_partner_field_checkpoint(prior_source, prior_zero, execute)

        ladder = {
            "A_scratch_joint_full1000": (scratch_checkpoint, scratch_config_path),
            "B_dual_prior_full1000": (prior_zero, primary_config_path),
            "C_global_C_full1000": (refit_root / "global_c" / "refit.pt", primary_config_path),
            "D_DeltaC_full1000": (refit_root / "delta_c" / "refit.pt", primary_config_path),
            "E_alpha_full1000": (refit_root / "alpha" / "refit.pt", primary_config_path),
            "F_joint_full1000": (refit_root / "joint" / "refit.pt", primary_config_path),
        }
        for model_name, (checkpoint, config_path) in ladder.items():
            evaluation = root / "evaluation" / "component_ladder" / model_name / f"seed{seed}"
            _core_eval(
                config_path,
                checkpoint,
                manifests / "complex_test.tsv",
                evaluation,
                model_name,
                device,
                execute,
            )
            rows.append({"model": model_name, "seed": seed, "run_dir": str(evaluation.resolve())})
    return rows


def run_data_efficiency(
    base: dict,
    manifests: Path,
    root: Path,
    device: str | None,
    execute: bool,
) -> list[dict]:
    """Nested 10/25/50/100% subsets of the same 900-complex development train split."""
    rows = []
    fractions = [float(x) for x in base["evaluation"]["data_efficiency_fractions"]]
    for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
        config = _config_for_seed(base, seed)
        config_path = _write_yaml(config, root / "configs" / f"primary_seed{seed}.yaml")
        prior = root / "training" / "primary_development" / f"seed{seed}" / "rna_prior" / "best.pt"
        ranking_seed = seed + 9901
        for fraction in sorted(fractions):
            percent = int(round(100 * fraction))
            model_name = f"DMICF_complex_train_{percent:03d}pct"
            subset = root / "manifests" / "data_efficiency" / f"seed{seed}_{percent:03d}pct.tsv"
            _fixed_fraction_manifest(
                manifests / "complex_train.tsv", fraction, ranking_seed, subset, execute
            )
            previous = prior
            train_root = root / "training" / "data_efficiency" / model_name / f"seed{seed}"
            for stage in [Stage.GLOBAL_C, Stage.DELTA_C, Stage.ALPHA, Stage.JOINT]:
                previous = _train_one_development_stage(
                    stage,
                    config_path,
                    subset,
                    manifests / "complex_val.tsv",
                    train_root / stage.value,
                    previous,
                    device,
                    execute,
                )
            evaluation = root / "evaluation" / "data_efficiency" / model_name / f"seed{seed}"
            _core_eval(
                config_path,
                previous,
                manifests / "complex_test.tsv",
                evaluation,
                model_name,
                device,
                execute,
            )
            rows.append({"model": model_name, "seed": seed, "run_dir": str(evaluation.resolve())})
    return rows


def run_targeted_ablation(
    base: dict,
    manifests: Path,
    root: Path,
    device: str | None,
    execute: bool,
) -> list[dict]:
    rows = []
    variants = ["distance_only", "no_coordinate_noise", "no_graph_stochastic_regularization"]
    for variant in variants:
        variant_base = _targeted_variant(base, variant)
        for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
            config = _config_for_seed(variant_base, seed, suffix=variant)
            config_path = _write_yaml(config, root / "configs" / f"{variant}_seed{seed}.yaml")
            development = (
                root / "training" / "targeted_ablation_development" / variant / f"seed{seed}"
            )
            _train_all_development(config_path, manifests, development, device, execute)
            refit_root = (
                root / "training" / "targeted_ablation_refit_full1000" / variant / f"seed{seed}"
            )
            final_checkpoint = (
                refit_full_pipeline(config, manifests, development, refit_root, device=device)
                if execute
                else refit_root / "joint" / "refit.pt"
            )
            evaluation = root / "evaluation" / "targeted_ablation" / variant / f"seed{seed}"
            _core_eval(
                config_path,
                final_checkpoint,
                manifests / "complex_test.tsv",
                evaluation,
                variant,
                device,
                execute,
            )
            rows.append({"model": variant, "seed": seed, "run_dir": str(evaluation.resolve())})
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
        raise SystemExit(f"Frozen manifests missing: {missing}. Run data pipeline/audit first.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests/pilot_v1"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/pilot_experiments"))
    parser.add_argument("--device")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["primary", "component_ladder", "data_efficiency", "targeted_ablation", "all"],
        default=["all"],
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    execute = bool(args.execute)
    if execute:
        _check_manifests(args.manifest_root)
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    requested = set(args.families)
    if "all" in requested:
        requested = {"primary", "component_ladder", "data_efficiency", "targeted_ablation"}

    args.out.mkdir(parents=True, exist_ok=True)
    plan = {
        "config": str(args.config.resolve()),
        "manifest_root": str(args.manifest_root.resolve()),
        "families": sorted(requested),
        "execute": execute,
        "primary_training_seeds": base["experiment"]["primary_training_seeds"],
        "analysis_seed": base["evaluation"].get("analysis_seed"),
        "primary_final_model_policy": (
            "900/100 selects schedule prefix; refit from scratch on all 1000 without validation"
        ),
        "heavy_final_battery_policy": "predeclared analysis seed only; other seeds get core final evaluation",
        "final_test_policy": "100 complexes never used for optimization or model selection",
    }
    (args.out / "experiment_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )

    all_statistics: list[dict] = []
    if "primary" in requested:
        all_statistics.extend(
            run_primary(base, args.manifest_root, args.out, args.device, execute)
        )
    if "component_ladder" in requested:
        ladder_rows = run_component_ladder(
            base, args.manifest_root, args.out, args.device, execute
        )
        _compare_runs(
            ladder_rows,
            args.out / "statistics" / "component_ladder_runs.tsv",
            "F_joint_full1000",
            args.out / "statistics" / "component_ladder",
            int(base["evaluation"]["bootstrap_resamples"]),
            execute,
        )
    if "data_efficiency" in requested:
        efficiency_rows = run_data_efficiency(
            base, args.manifest_root, args.out, args.device, execute
        )
        _compare_runs(
            efficiency_rows,
            args.out / "statistics" / "data_efficiency_runs.tsv",
            "DMICF_complex_train_100pct",
            args.out / "statistics" / "data_efficiency",
            int(base["evaluation"]["bootstrap_resamples"]),
            execute,
        )
    if "targeted_ablation" in requested:
        all_statistics.extend(
            run_targeted_ablation(base, args.manifest_root, args.out, args.device, execute)
        )

    if execute and all_statistics:
        path = args.out / "statistics" / "primary_and_full_ablation_runs.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_statistics).to_csv(path, sep="\t", index=False)
    if not execute:
        print(
            "\nDRY RUN: no GPU training started. Use --execute only after data audit, "
            "baseline preflight and GO/NO-GO checks pass."
        )


if __name__ == "__main__":
    main()
