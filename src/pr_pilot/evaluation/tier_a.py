"""Lightweight all-seed final evaluation supporting H1-H4.

Tier A is deliberately limited to quantities required across every primary seed:
conditional Protein/RNA predictions, deterministic teacher-forced joint pseudo-NLL
and partner scrambling.  Expensive counterfactual, field, robustness and sampling
analyses belong to Tier B and are run only on the predeclared analysis seed.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import pandas as pd
import torch
import yaml

from pr_pilot.evaluation.battery import mandatory_test_registry, token_metrics
from pr_pilot.evaluation.runner import (
    _move,
    load_model,
    partner_scramble,
    score_conditional,
    score_joint_teacher_forced,
)
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row


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


@torch.no_grad()
def evaluate_tier_a(
    cfg: dict,
    checkpoint: Path,
    manifest: Path,
    out_dir: Path,
    device: str | None = None,
    model_name: str = "DMICF",
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(checkpoint, cfg, dev)
    adapter = _adapter(cfg)
    seed = int(cfg["experiment"]["pilot_seed"])
    p_frames, r_frames, joint_frames, scramble_frames = [], [], [], []
    for row in ManifestTable(manifest).rows():
        sample = _move(load_complex_row(adapter, row), dev)
        p_frames.append(
            score_conditional(
                model, sample, "protein", False, seed=seed, model_name=model_name
            )
        )
        r_frames.append(
            score_conditional(
                model, sample, "rna", False, seed=seed, model_name=model_name
            )
        )
        joint_frames.append(
            score_joint_teacher_forced(
                model,
                sample,
                orders=int(cfg["evaluation"].get("joint_teacher_forced_orders", 5)),
                seed=seed,
                model_name=model_name,
            )
        )
        scramble_frames.append(
            partner_scramble(
                model,
                sample,
                repeats=int(cfg["evaluation"].get("scramble_repeats", 20)),
                seed=seed,
            )
        )

    core = out_dir / "core"
    core.mkdir(parents=True, exist_ok=True)
    outputs = {
        "conditional_protein": pd.concat(p_frames, ignore_index=True),
        "conditional_rna": pd.concat(r_frames, ignore_index=True),
        "joint_teacher_forced": pd.concat(joint_frames, ignore_index=True),
        "partner_scramble": pd.concat(scramble_frames, ignore_index=True),
    }
    for name, frame in outputs.items():
        frame.to_csv(core / f"{name}.tsv", sep="\t", index=False)
    mandatory_test_registry().to_csv(
        out_dir / "test_registry.tsv", sep="\t", index=False
    )
    summary = {
        "model": model_name,
        "seed": seed,
        "complexes": int(outputs["conditional_protein"].sample_id.nunique()),
        "protein": token_metrics(outputs["conditional_protein"], 20),
        "rna": token_metrics(outputs["conditional_rna"], 4),
        "joint_protein": token_metrics(
            outputs["joint_teacher_forced"].query("polymer=='protein'"), 20
        ),
        "joint_rna": token_metrics(
            outputs["joint_teacher_forced"].query("polymer=='rna'"), 4
        ),
        "tier": "A",
        "contains_expensive_secondary_battery": False,
    }
    (out_dir / "tier_a_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", default="DMICF")
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = evaluate_tier_a(
        cfg,
        args.checkpoint,
        args.manifest,
        args.out,
        args.device,
        args.model_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
