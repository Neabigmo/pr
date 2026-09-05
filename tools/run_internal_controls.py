#!/usr/bin/env python3
"""Train, full-1000-refit and evaluate the two same-data internal controls.

This script is separate from the primary orchestrator so the control forward
semantics cannot accidentally be lost when a checkpoint is loaded by a normal
DM-ICF evaluator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from pr_pilot.evaluation.runner import partner_scramble, score_conditional, score_joint_teacher_forced
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row
from pr_pilot.training.control_modes import install_control_mode, run_control_pipeline
from pr_pilot.training.engine import build_model_from_config


MODES = ("partner_blind", "geometry_only")


def _adapter(cfg: dict) -> GemmiStructureAdapter:
    g = cfg["geometry"]
    return GemmiStructureAdapter(
        int(g["rbf_bins"]),
        int(g["intra_max_neighbors"]),
        float(g["pr_cutoff_angstrom"]),
        int(g["pr_max_neighbors"]),
        0.0,
        int(cfg["experiment"]["pilot_seed"]),
        bool(g["rich_pr_geometry"]),
    )


def _move(sample, device):
    for graph in [sample.protein, sample.rna]:
        for name in ["node_x", "edge_index", "edge_x", "sequence", "interface", "valid", "fixed", "reference_xyz", "chain_index"]:
            setattr(graph, name, getattr(graph, name).to(device))
    sample.pr.protein_index = sample.pr.protein_index.to(device)
    sample.pr.rna_index = sample.pr.rna_index.to(device)
    sample.pr.edge_features = sample.pr.edge_features.to(device)
    sample.pr.effective_distance = sample.pr.effective_distance.to(device)
    if sample.pr.edge_batch is not None:
        sample.pr.edge_batch = sample.pr.edge_batch.to(device)
    return sample


def evaluate_control(
    cfg: dict,
    mode: str,
    checkpoint: Path,
    test_manifest: Path,
    out_dir: Path,
    device: str | None,
) -> None:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(dev)
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("control_mode") != mode:
        raise ValueError(f"Checkpoint control mode {payload.get('control_mode')} != requested {mode}")
    model.load_state_dict(payload["model"])
    install_control_mode(model, mode)
    model.eval()
    adapter = _adapter(cfg)
    table = ManifestTable(test_manifest)
    seed = int(cfg["experiment"]["pilot_seed"])
    p_frames, r_frames, j_frames, s_frames = [], [], [], []
    for row in table.rows():
        sample = _move(load_complex_row(adapter, row), dev)
        p_frames.append(score_conditional(model, sample, "protein", False, seed=seed, model_name=mode))
        r_frames.append(score_conditional(model, sample, "rna", False, seed=seed, model_name=mode))
        j_frames.append(
            score_joint_teacher_forced(
                model,
                sample,
                orders=int(cfg["evaluation"].get("joint_teacher_forced_orders", 5)),
                seed=seed,
                model_name=mode,
            )
        )
        s_frames.append(
            partner_scramble(
                model,
                sample,
                repeats=int(cfg["evaluation"].get("scramble_repeats", 20)),
                seed=seed,
            )
        )
    core = out_dir / "core"
    core.mkdir(parents=True, exist_ok=True)
    pd.concat(p_frames, ignore_index=True).to_csv(core / "conditional_protein.tsv", sep="\t", index=False)
    pd.concat(r_frames, ignore_index=True).to_csv(core / "conditional_rna.tsv", sep="\t", index=False)
    pd.concat(j_frames, ignore_index=True).to_csv(core / "joint_teacher_forced.tsv", sep="\t", index=False)
    pd.concat(s_frames, ignore_index=True).to_csv(core / "partner_scramble.tsv", sep="\t", index=False)
    (out_dir / "control_semantics.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "checkpoint": str(checkpoint),
                "partner_identity_available": False,
                "cross_partner_geometry_available": mode == "geometry_only",
                "same_architecture_as_primary": True,
                "same_frozen_manifests": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests/pilot_v1"))
    parser.add_argument("--primary-root", type=Path, default=Path("artifacts/pilot_experiments/training"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/internal_controls"))
    parser.add_argument("--device")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--modes", choices=list(MODES), nargs="+", default=list(MODES))
    args = parser.parse_args()

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(x) for x in base["experiment"]["primary_training_seeds"]]
    run_rows = []
    for seed in seeds:
        cfg = json.loads(json.dumps(base))
        cfg["experiment"]["pilot_seed"] = int(seed)
        dev_prior = args.primary_root / "primary_development" / f"seed{seed}" / "rna_prior" / "best.pt"
        refit_prior = args.primary_root / "primary_refit_full1000" / f"seed{seed}" / "rna_prior" / "refit.pt"
        for mode in args.modes:
            train_out = args.out / "training" / mode / f"seed{seed}"
            final_checkpoint = run_control_pipeline(
                cfg,
                mode,
                args.manifest_root,
                dev_prior,
                refit_prior,
                train_out,
                args.device,
            )
            eval_out = args.out / "evaluation" / mode / f"seed{seed}"
            evaluate_control(cfg, mode, final_checkpoint, args.manifest_root / "complex_test.tsv", eval_out, args.device)
            run_rows.append({"model": mode, "seed": seed, "run_dir": str(eval_out.resolve())})

    runs = pd.DataFrame(run_rows)
    runs_path = args.out / "control_runs.tsv"
    runs.to_csv(runs_path, sep="\t", index=False)
    print(f"Control runs ready for paired comparison: {runs_path}")


if __name__ == "__main__":
    main()
