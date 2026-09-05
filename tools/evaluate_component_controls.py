#!/usr/bin/env python3
"""Evaluate frozen B/C/E/F component checkpoints after final-test unlock.

B = dual structural priors with cross-molecular correction forced to zero.
C = Stage-C global compatibility anchor.
E = post-alpha contextual field before Joint coordination.
F = final Joint model (normally reuses primary Tier-A run directory).

No checkpoint is retrained or selected here; every source checkpoint SHA comes
from EVALUATION_PROTOCOL_LOCK.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from pr_pilot.evaluation.runner import _move, partner_scramble, score_conditional, score_joint_teacher_forced
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row
from pr_pilot.training.control_modes import install_control_mode
from pr_pilot.training.engine import build_model_from_config


COMPONENTS = {
    "B_dual_prior_full1000": "rna_prior",
    "C_global_C_full1000": "global_c",
    "E_alpha_full1000": "alpha",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _adapter(cfg: dict) -> GemmiStructureAdapter:
    g = cfg["geometry"]
    return GemmiStructureAdapter(
        int(g["rbf_bins"]), int(g["intra_max_neighbors"]),
        float(g["pr_cutoff_angstrom"]), int(g["pr_max_neighbors"]),
        0.0, int(cfg["experiment"]["pilot_seed"]), bool(g["rich_pr_geometry"]),
    )


@torch.no_grad()
def _evaluate_component(
    cfg: dict,
    checkpoint: Path,
    test_manifest: Path,
    out_dir: Path,
    model_name: str,
    device: str | None,
    partner_blind: bool,
) -> None:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(dev)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    if partner_blind:
        install_control_mode(model, "partner_blind")
    model.eval()
    adapter = _adapter(cfg)
    seed = int(cfg["experiment"]["pilot_seed"])
    p, r, j, s = [], [], [], []
    for row in ManifestTable(test_manifest).rows():
        sample = _move(load_complex_row(adapter, row), dev)
        p.append(score_conditional(model, sample, "protein", False, seed=seed, model_name=model_name))
        r.append(score_conditional(model, sample, "rna", False, seed=seed, model_name=model_name))
        j.append(score_joint_teacher_forced(model, sample, int(cfg["evaluation"]["joint_teacher_forced_orders"]), seed, model_name))
        s.append(partner_scramble(model, sample, int(cfg["evaluation"]["scramble_repeats"]), seed))
    core = out_dir / "core"
    core.mkdir(parents=True, exist_ok=True)
    pd.concat(p, ignore_index=True).to_csv(core / "conditional_protein.tsv", sep="\t", index=False)
    pd.concat(r, ignore_index=True).to_csv(core / "conditional_rna.tsv", sep="\t", index=False)
    pd.concat(j, ignore_index=True).to_csv(core / "joint_teacher_forced.tsv", sep="\t", index=False)
    pd.concat(s, ignore_index=True).to_csv(core / "partner_scramble.tsv", sep="\t", index=False)
    (out_dir / "component_semantics.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "source_stage": payload.get("stage"),
                "partner_blind": partner_blind,
                "training_or_selection_during_evaluation": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--primary-evaluation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/component_controls"))
    parser.add_argument("--device")
    args = parser.parse_args()

    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if _sha256(args.config) != lock["config_sha256"]:
        raise RuntimeError("config differs from evaluation protocol lock")
    if _sha256(args.test_manifest) != lock["test_manifest_sha256"]:
        raise RuntimeError("test manifest differs from evaluation protocol lock")
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inventory = {
        (int(item["seed"]), str(item["stage"])): item
        for item in lock["primary_checkpoint_inventory"]
    }
    rows = []
    for seed in [int(x) for x in lock["primary_training_seeds"]]:
        cfg = copy.deepcopy(base)
        cfg["experiment"]["pilot_seed"] = seed
        for model_name, stage in COMPONENTS.items():
            frozen = inventory[(seed, stage)]
            checkpoint = Path(frozen["path"])
            if _sha256(checkpoint) != str(frozen["sha256"]):
                raise RuntimeError(f"frozen {stage} checkpoint changed for seed {seed}")
            run_dir = args.out / model_name / f"seed{seed}"
            _evaluate_component(
                cfg,
                checkpoint,
                args.test_manifest,
                run_dir,
                model_name,
                args.device,
                partner_blind=(stage == "rna_prior"),
            )
            rows.append({"model": model_name, "seed": seed, "run_dir": str(run_dir.resolve())})

        # F was already evaluated by the gated primary final-evaluation script.
        f_dir = args.primary_evaluation_root / "primary_refit_full1000" / f"seed{seed}"
        for required in ["conditional_protein.tsv", "conditional_rna.tsv", "joint_teacher_forced.tsv", "partner_scramble.tsv"]:
            if not (f_dir / "core" / required).exists():
                raise FileNotFoundError(f_dir / "core" / required)
        rows.append({"model": "F_joint_full1000", "seed": seed, "run_dir": str(f_dir.resolve())})

    args.out.mkdir(parents=True, exist_ok=True)
    runs = args.out / "component_ladder_runs.tsv"
    pd.DataFrame(rows).to_csv(runs, sep="\t", index=False)
    print(json.dumps({"status": "FROZEN_COMPONENT_EVALUATION_COMPLETE", "runs": len(rows), "manifest": str(runs.resolve())}, indent=2))


if __name__ == "__main__":
    main()
