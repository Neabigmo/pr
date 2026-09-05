#!/usr/bin/env python3
"""Orchestrate the complete frozen mini-pilot experiment matrix.

This script never chooses hyperparameters from the final 100 complexes. It only
consumes already-frozen manifests and a development-selected base config.

Experiment families
-------------------
1. Primary DM-ICF: full six-stage training for every predeclared training seed.
2. Component ladder on the same 100 final complexes:
   A scratch-joint; B dual structural prior; C + global C; D + DeltaC;
   E + learned alpha; F + joint coordination.
3. Data efficiency: 10/25/50/100% of the 900 complex-training samples, reusing
   the corresponding seed's frozen dual-prior checkpoint and unchanged 100-complex
   validation set.
4. Targeted full-pipeline ablations: distance-only PR geometry, no coordinate
   noise, and no graph stochastic regularization. These are deliberately fewer
   than the component ladder: the pilot should answer scientific questions, not
   turn into an unbounded hyperparameter sweep.
5. Full multidimensional suite on every primary final model; cross-seed/model
   paired inference is run at the biological-complex level.

Default mode is --dry-run. Pass --execute explicitly to launch expensive runs.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

from pr_pilot.data.manifest import deterministic_sample


STAGES = ["protein_prior", "rna_prior", "global_c", "delta_c", "alpha", "joint"]


def _run(command: list[str], *, execute: bool, env: dict[str, str] | None = None) -> None:
    print("$", shlex.join(command), flush=True)
    if not execute:
        return
    subprocess.run(command, check=True, env=env)


def _write_config(config: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _seed_config(base: dict, seed: int) -> dict:
    config = copy.deepcopy(base)
    config["experiment"]["pilot_seed"] = int(seed)
    config["experiment"]["name"] = f"{base['experiment']['name']}__seed{seed}"
    return config


def _variant_config(base: dict, name: str) -> dict:
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
    config["experiment"]["name"] = f"{base['experiment']['name']}__{name}"
    return config


def _python_module(module: str, *args: object) -> list[str]:
    return [sys.executable, "-m", module, *[str(x) for x in args]]


def _cli(*args: object) -> list[str]:
    return _python_module("pr_pilot.cli", *args)


def _zero_interaction_prior_checkpoint(source: Path, destination: Path, *, execute: bool) -> None:
    """Make dual-prior checkpoint exactly partner-blind for ladder B evaluation."""
    print(f"[checkpoint transform] {source} -> {destination}: zero dmicf.global_c.raw", flush=True)
    if not execute:
        return
    payload = torch.load(source, map_location="cpu")
    state = payload["model"]
    state["dmicf.global_c.raw"] = torch.zeros_like(state["dmicf.global_c.raw"])
    # DeltaC output is guaranteed zero by the prior-stage initialization contract;
    # enforce it again so the ladder cannot depend on initializer drift.
    for key in ["dmicf.delta.out.weight", "dmicf.delta.out.bias"]:
        state[key] = torch.zeros_like(state[key])
    payload["stage"] = "dual_prior_partner_blind_eval"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def _fraction_manifest(
    full_train: Path,
    fraction: float,
    seed: int,
    out_path: Path,
    *,
    execute: bool,
) -> int:
    frame = pd.read_csv(full_train, sep="\t")
    n = max(1, int(round(len(frame) * fraction)))
    print(f"[manifest subset] fraction={fraction:.2f}, n={n}: {out_path}")
    if execute:
        subset = deterministic_sample(frame, n, seed=seed, key="sample_id")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(out_path, sep="\t", index=False)
    return n


def _train_all(
    config_path: Path,
    manifest_root: Path,
    output: Path,
    device: str | None,
    *,
    execute: bool,
) -> None:
    cmd = _cli(
        "train-all",
        "--config",
        config_path,
        "--manifest-root",
        manifest_root,
        "--out",
        output,
    )
    if device:
        cmd += ["--device", device]
    _run(cmd, execute=execute)


def _train_stage(
    stage: str,
    config_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    init_checkpoint: Path | None,
    output: Path,
    device: str | None,
    *,
    execute: bool,
) -> Path:
    cmd = _cli(
        "train",
        "--stage",
        stage,
        "--config",
        config_path,
        "--manifest",
        train_manifest,
        "--validation",
        val_manifest,
        "--out",
        output,
    )
    if init_checkpoint is not None:
        cmd += ["--init-checkpoint", init_checkpoint]
    if device:
        cmd += ["--device", device]
    _run(cmd, execute=execute)
    return output / "best.pt"


def _core_eval(
    config_path: Path,
    checkpoint: Path,
    test_manifest: Path,
    output: Path,
    model_name: str,
    device: str | None,
    *,
    execute: bool,
) -> None:
    cmd = _cli(
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
        cmd += ["--device", device]
    _run(cmd, execute=execute)


def _full_suite(
    config_path: Path,
    checkpoint: Path,
    test_manifest: Path,
    dev_manifest: Path,
    output: Path,
    device: str | None,
    *,
    execute: bool,
) -> None:
    cmd = _python_module(
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
        cmd += ["--device", device]
    _run(cmd, execute=execute)


def run_primary(
    base: dict,
    manifest_root: Path,
    root: Path,
    device: str | None,
    *,
    execute: bool,
) -> pd.DataFrame:
    seeds = [int(x) for x in base["experiment"]["primary_training_seeds"]]
    run_rows = []
    for seed in seeds:
        config = _seed_config(base, seed)
        config_path = _write_config(config, root / "configs" / f"primary_seed{seed}.yaml")
        train_out = root / "training" / "primary" / f"seed{seed}"
        _train_all(config_path, manifest_root, train_out, device, execute=execute)
        final_checkpoint = train_out / "joint" / "best.pt"
        evaluation_out = root / "evaluation" / "primary" / f"seed{seed}"
        _full_suite(
            config_path,
            final_checkpoint,
            manifest_root / "complex_test.tsv",
            manifest_root / "complex_dev.tsv",
            evaluation_out,
            device,
            execute=execute,
        )
        run_rows.append({"model": "DMICF", "seed": seed, "run_dir": str(evaluation_out.resolve())})
    frame = pd.DataFrame(run_rows)
    if execute:
        path = root / "statistics" / "primary_runs.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
        _run(
            _python_module(
                "pr_pilot.evaluation.compare_runs",
                "--runs",
                path,
                "--reference",
                "DMICF",
                "--out",
                root / "statistics" / "primary_seed_summary",
                "--bootstrap",
                int(base["evaluation"]["bootstrap_resamples"]),
            ),
            execute=True,
        )
    return frame


def run_component_ladder(
    base: dict,
    manifest_root: Path,
    root: Path,
    device: str | None,
    *,
    execute: bool,
) -> pd.DataFrame:
    """Run A scratch control and evaluate B--F checkpoints from each primary seed."""
    seeds = [int(x) for x in base["experiment"]["primary_training_seeds"]]
    rows = []
    for seed in seeds:
        config_path = root / "configs" / f"primary_seed{seed}.yaml"
        if not config_path.exists() and execute:
            _write_config(_seed_config(base, seed), config_path)
        primary = root / "training" / "primary" / f"seed{seed}"

        scratch_out = root / "training" / "component_ladder" / f"seed{seed}" / "A_scratch_joint"
        scratch_checkpoint = _train_stage(
            "joint",
            config_path,
            manifest_root / "complex_train.tsv",
            manifest_root / "complex_val.tsv",
            None,
            scratch_out,
            device,
            execute=execute,
        )

        dual_prior_source = primary / "rna_prior" / "best.pt"
        dual_prior_checkpoint = root / "training" / "component_ladder" / f"seed{seed}" / "B_dual_prior.pt"
        _zero_interaction_prior_checkpoint(dual_prior_source, dual_prior_checkpoint, execute=execute)

        ladder = {
            "A_scratch_joint": scratch_checkpoint,
            "B_dual_prior": dual_prior_checkpoint,
            "C_global_C": primary / "global_c" / "best.pt",
            "D_context_DeltaC": primary / "delta_c" / "best.pt",
            "E_learned_alpha": primary / "alpha" / "best.pt",
            "F_joint_coordination": primary / "joint" / "best.pt",
        }
        for label, checkpoint in ladder.items():
            evaluation_out = root / "evaluation" / "component_ladder" / label / f"seed{seed}"
            _core_eval(
                config_path,
                checkpoint,
                manifest_root / "complex_test.tsv",
                evaluation_out,
                label,
                device,
                execute=execute,
            )
            rows.append({"model": label, "seed": seed, "run_dir": str(evaluation_out.resolve())})

    frame = pd.DataFrame(rows)
    if execute:
        path = root / "statistics" / "component_ladder_runs.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
        _run(
            _python_module(
                "pr_pilot.evaluation.compare_runs",
                "--runs",
                path,
                "--reference",
                "F_joint_coordination",
                "--out",
                root / "statistics" / "component_ladder",
                "--bootstrap",
                int(base["evaluation"]["bootstrap_resamples"]),
            ),
            execute=True,
        )
    return frame


def run_data_efficiency(
    base: dict,
    manifest_root: Path,
    root: Path,
    device: str | None,
    *,
    execute: bool,
) -> pd.DataFrame:
    fractions = [float(x) for x in base["evaluation"]["data_efficiency_fractions"]]
    seeds = [int(x) for x in base["experiment"]["primary_training_seeds"]]
    rows = []
    for seed in seeds:
        config_path = root / "configs" / f"primary_seed{seed}.yaml"
        if not config_path.exists() and execute:
            _write_config(_seed_config(base, seed), config_path)
        dual_prior = root / "training" / "primary" / f"seed{seed}" / "rna_prior" / "best.pt"
        for fraction in fractions:
            label = f"DMICF_{int(round(fraction * 100)):03d}pct"
            subset = root / "manifests" / "data_efficiency" / f"seed{seed}_{int(round(fraction*100)):03d}pct.tsv"
            _fraction_manifest(
                manifest_root / "complex_train.tsv",
                fraction,
                seed + int(round(fraction * 10000)),
                subset,
                execute=execute,
            )
            previous = dual_prior
            base_out = root / "training" / "data_efficiency" / label / f"seed{seed}"
            for stage in ["global_c", "delta_c", "alpha", "joint"]:
                previous = _train_stage(
                    stage,
                    config_path,
                    subset,
                    manifest_root / "complex_val.tsv",
                    previous,
                    base_out / stage,
                    device,
                    execute=execute,
                )
            evaluation_out = root / "evaluation" / "data_efficiency" / label / f"seed{seed}"
            _core_eval(
                config_path,
                previous,
                manifest_root / "complex_test.tsv",
                evaluation_out,
                label,
                device,
                execute=execute,
            )
            rows.append({"model": label, "seed": seed, "run_dir": str(evaluation_out.resolve())})
    frame = pd.DataFrame(rows)
    if execute:
        path = root / "statistics" / "data_efficiency_runs.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
        _run(
            _python_module(
                "pr_pilot.evaluation.compare_runs",
                "--runs",
                path,
                "--reference",
                "DMICF_100pct",
                "--out",
                root / "statistics" / "data_efficiency",
                "--bootstrap",
                int(base["evaluation"]["bootstrap_resamples"]),
            ),
            execute=True,
        )
    return frame


def run_targeted_full_ablation(
    base: dict,
    manifest_root: Path,
    root: Path,
    device: str | None,
    *,
    execute: bool,
) -> pd.DataFrame:
    variants = ["distance_only", "no_coordinate_noise", "no_graph_stochastic_regularization"]
    seeds = [int(x) for x in base["experiment"]["primary_training_seeds"]]
    rows = []
    for variant in variants:
        for seed in seeds:
            config = _seed_config(_variant_config(base, variant), seed)
            config_path = _write_config(config, root / "configs" / f"{variant}_seed{seed}.yaml")
            train_out = root / "training" / "targeted_ablation" / variant / f"seed{seed}"
            _train_all(config_path, manifest_root, train_out, device, execute=execute)
            evaluation_out = root / "evaluation" / "targeted_ablation" / variant / f"seed{seed}"
            _core_eval(
                config_path,
                train_out / "joint" / "best.pt",
                manifest_root / "complex_test.tsv",
                evaluation_out,
                variant,
                device,
                execute=execute,
            )
            rows.append({"model": variant, "seed": seed, "run_dir": str(evaluation_out.resolve())})
    frame = pd.DataFrame(rows)
    if execute:
        path = root / "statistics" / "targeted_ablation_runs.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, sep="\t", index=False)
    return frame


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
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    requested = set(args.families)
    if "all" in requested:
        requested = {"primary", "component_ladder", "data_efficiency", "targeted_ablation"}

    args.out.mkdir(parents=True, exist_ok=True)
    plan = {
        "config": str(args.config.resolve()),
        "manifest_root": str(args.manifest_root.resolve()),
        "execute": execute,
        "families": sorted(requested),
        "primary_training_seeds": base["experiment"]["primary_training_seeds"],
    }
    (args.out / "experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    if "primary" in requested:
        run_primary(base, args.manifest_root, args.out, args.device, execute=execute)
    if "component_ladder" in requested:
        run_component_ladder(base, args.manifest_root, args.out, args.device, execute=execute)
    if "data_efficiency" in requested:
        run_data_efficiency(base, args.manifest_root, args.out, args.device, execute=execute)
    if "targeted_ablation" in requested:
        run_targeted_full_ablation(base, args.manifest_root, args.out, args.device, execute=execute)

    if not execute:
        print("\nDRY RUN ONLY. Re-run with --execute after manifests/audit and baseline preparation are frozen.")


if __name__ == "__main__":
    main()
