#!/usr/bin/env python3
"""Run the complete frozen mini-pilot without leaking final-test information.

The orchestrator deliberately separates two phases:

DEVELOPMENT PHASE
  - 900/100 Protein prior split;
  - 900/100 RNA prior split;
  - 900/100 complex train/validation split;
  - validation chooses epoch counts and all tunable settings.

FINAL REFIT PHASE
  - retrain Protein prior on all 1,000 Protein structures;
  - retrain RNA prior on all 1,000 RNA structures;
  - retrain C -> DeltaC -> alpha -> joint on all 1,000 complex-development samples;
  - every stage uses the epoch count selected during development;
  - validation and the final 100 complexes are not consulted during refit.

Only these full-1,000 refit checkpoints are allowed to support the primary final
claim. Development checkpoints remain available for diagnostics and data-efficiency
curves.

Default is --dry-run. Expensive training starts only with --execute.
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
from pr_pilot.training.refit import refit_full_pipeline, refit_stage, selected_epoch_count
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
        "train-all",
        "--config",
        config_path,
        "--manifest-root",
        manifest_root,
        "--out",
        out_dir,
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


def _zero_partner_field_checkpoint(source: Path, destination: Path, execute: bool) -> None:
    """Create an exact dual-structural-prior checkpoint for component ladder B."""
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
    print(f"[data-efficiency] {fraction:.2f} -> {n}/{len(frame)} training complexes: {destination}")
    if execute:
        subset = deterministic_sample(frame, n, seed=seed, key="sample_id")
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
    """Development-select then final-refit all primary seeds."""
    rows = []
    for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
        config = _config_for_seed(base, seed)
        config_path = _write_yaml(config, root / "configs" / f"primary_seed{seed}.yaml")
        development = root / "training" / "primary_development" / f"seed{seed}"
        _train_all_development(config_path, manifests, development, device, execute)

        refit_dir = root / "training" / "primary_refit_full1000" / f"seed{seed}"
        print(f"[final refit] seed={seed}: all 1000 Protein + 1000 RNA + 1000 complex development samples")
        if execute:
            final_checkpoint = refit_full_pipeline(
                config,
                manifests,
                development,
                refit_dir,
                device=device,
            )
        else:
            final_checkpoint = refit_dir / "joint" / "refit.pt"

        evaluation = root / "evaluation" / "primary_refit_full1000" / f"seed{seed}"
        _full_suite(
            config_path,
            final_checkpoint,
            manifests / "complex_test.tsv",
            manifests / "complex_dev.tsv",
            evaluation,
            device,
            execute,
        )
        rows.append({"model": "DMICF_full1000", "seed": seed, "run_dir": str(evaluation.resolve())})
    return rows


def run_component_ladder(
    base: dict,
    manifests: Path,
    root: Path,
    device: str | None,
    execute: bool,
) -> list[dict]:
    """A--F component comparison using full-1000 refit checkpoints where applicable."""
    rows = []
    for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
        config = _config_for_seed(base, seed)
        config_path = _write_yaml(config, root / "configs" / f"primary_seed{seed}.yaml")
        development = root / "training" / "primary_development" / f"seed{seed}"
        refit_root = root / "training" / "primary_refit_full1000" / f"seed{seed}"

        # A: choose scratch-joint epoch count on 900/100, then refit scratch joint
        # on all 1000 complex-development structures for that fixed epoch count.
        scratch_dev = root / "training" / "component_ladder" / f"seed{seed}" / "A_scratch_development"
        scratch_dev_checkpoint = _train_one_development_stage(
            Stage.JOINT,
            config_path,
            manifests / "complex_train.tsv",
            manifests / "complex_val.tsv",
            scratch_dev,
            None,
            device,
            execute,
        )
        scratch_refit_dir = root / "training" / "component_ladder" / f"seed{seed}" / "A_scratch_refit_full1000"
        if execute:
            scratch_epochs = selected_epoch_count(scratch_dev_checkpoint)
            scratch_checkpoint = refit_stage(
                config,
                Stage.JOINT,
                manifests / "complex_dev.tsv",
                scratch_epochs,
                scratch_refit_dir,
                init_checkpoint=None,
                device=device,
            )
        else:
            scratch_checkpoint = scratch_refit_dir / "refit.pt"

        # B: the full-1000 dual-prior refit after RNA-prior stage, but with exact
        # zero interaction field so no random C leaks into the prior-only control.
        prior_source = refit_root / "rna_prior" / "refit.pt"
        prior_zero = root / "training" / "component_ladder" / f"seed{seed}" / "B_dual_prior_zero_field.pt"
        _zero_partner_field_checkpoint(prior_source, prior_zero, execute)

        ladder = {
            "A_scratch_joint_full1000": scratch_checkpoint,
            "B_dual_prior_full1000": prior_zero,
            "C_global_C_full1000": refit_root / "global_c" / "refit.pt",
            "D_DeltaC_full1000": refit_root / "delta_c" / "refit.pt",
            "E_alpha_full1000": refit_root / "alpha" / "refit.pt",
            "F_joint_full1000": refit_root / "joint" / "refit.pt",
        }
        for model_name, checkpoint in ladder.items():
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
    """10/25/50/100% of the 900 complex-training set with unchanged 100 validation."""
    rows = []
    fractions = [float(x) for x in base["evaluation"]["data_efficiency_fractions"]]
    for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
        config = _config_for_seed(base, seed)
        config_path = _write_yaml(config, root / "configs" / f"primary_seed{seed}.yaml")
        # Development dual prior is intentionally reused: the question here is
        # complex-data efficiency, not the effect of changing prior data volume.
        prior = root / "training" / "primary_development" / f"seed{seed}" / "rna_prior" / "best.pt"
        for fraction in fractions:
            percent = int(round(100 * fraction))
            model_name = f"DMICF_complex_train_{percent:03d}pct"
            subset = root / "manifests" / "data_efficiency" / f"seed{seed}_{percent:03d}pct.tsv"
            _fixed_fraction_manifest(
                manifests / "complex_train.tsv",
                fraction,
                seed + percent * 7919,
                subset,
                execute,
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
    """Full-pipeline, full-1000 refit for three predeclared implementation questions."""
    rows = []
    variants = ["distance_only", "no_coordinate_noise", "no_graph_stochastic_regularization"]
    for variant in variants:
        variant_base = _targeted_variant(base, variant)
        for seed in [int(x) for x in base["experiment"]["primary_training_seeds"]]:
            config = _config_for_seed(variant_base, seed, suffix=variant)
            config_path = _write_yaml(config, root / "configs" / f"{variant}_seed{seed}.yaml")
            development = root / "training" / "targeted_ablation_development" / variant / f"seed{seed}"
            _train_all_development(config_path, manifests, development, device, execute)
            refit_root = root / "training" / "targeted_ablation_refit_full1000" / variant / f"seed{seed}"
            if execute:
                final_checkpoint = refit_full_pipeline(
                    config,
                    manifests,
                    development,
                    refit_root,
                    device=device,
                )
            else:
                final_checkpoint = refit_root / "joint" / "refit.pt"
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
        "primary_final_model_policy": "validation-select epochs on 900/100, then refit from scratch on all 1000 without validation",
        "final_test_policy": "100 complexes never used for optimization or selection; only predeclared final analyses",
    }
    (args.out / "experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    all_statistics = []
    if "primary" in requested:
        primary_rows = run_primary(base, args.manifest_root, args.out, args.device, execute)
        all_statistics.extend(primary_rows)
    if "component_ladder" in requested:
        ladder_rows = run_component_ladder(base, args.manifest_root, args.out, args.device, execute)
        _compare_runs(
            ladder_rows,
            args.out / "statistics" / "component_ladder_runs.tsv",
            "F_joint_full1000",
            args.out / "statistics" / "component_ladder",
            int(base["evaluation"]["bootstrap_resamples"]),
            execute,
        )
    if "data_efficiency" in requested:
        efficiency_rows = run_data_efficiency(base, args.manifest_root, args.out, args.device, execute)
        _compare_runs(
            efficiency_rows,
            args.out / "statistics" / "data_efficiency_runs.tsv",
            "DMICF_complex_train_100pct",
            args.out / "statistics" / "data_efficiency",
            int(base["evaluation"]["bootstrap_resamples"]),
            execute,
        )
    if "targeted_ablation" in requested:
        ablation_rows = run_targeted_ablation(base, args.manifest_root, args.out, args.device, execute)
        all_statistics.extend(ablation_rows)

    if execute and all_statistics:
        statistics_manifest = args.out / "statistics" / "primary_and_full_ablation_runs.tsv"
        statistics_manifest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_statistics).to_csv(statistics_manifest, sep="\t", index=False)
    if not execute:
        print("\nDRY RUN: no GPU training started. Re-run with --execute only after data audit and baseline preparation pass.")


if __name__ == "__main__":
    main()
