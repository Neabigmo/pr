"""Post-training audit for contextual-residual mean drift.

A frozen Stage-C anchor does not guarantee that DeltaC is mean-zero. This module
quantifies whether the contextual field has learned a nearly global offset and
exports both raw C and C_eff = C + mean(DeltaC) without changing training.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import torch
import yaml

from pr_pilot.evaluation.runner import _move, load_model
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
        float(cfg["structure_filters"]["interface_contact_angstrom"]),
    )


@torch.no_grad()
def audit_delta_c(
    cfg: dict,
    checkpoint: Path,
    manifest: Path,
    out_dir: Path,
    device: str | None = None,
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(checkpoint, cfg, dev)
    adapter = _adapter(cfg)
    deltas: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    c_value = model.dmicf.global_c().detach().cpu().numpy()

    for row in ManifestTable(manifest).rows():
        sample = _move(load_complex_row(adapter, row), dev)
        hp, hr = model.encode_backbones(
            sample.protein.node_x,
            sample.protein.edge_index,
            sample.protein.edge_x,
            sample.rna.node_x,
            sample.rna.edge_index,
            sample.rna.edge_x,
        )
        field = model.dmicf.field(hp, hr, sample.pr, use_delta=True, learned_alpha=True)
        dc = field["DeltaC"].detach().float().cpu().numpy()
        # Symmetric descriptive weight; normalized globally only for this audit.
        w = 0.5 * (
            field["alpha_p"].detach().float().cpu().numpy()
            + field["alpha_r"].detach().float().cpu().numpy()
        )
        deltas.append(dc)
        weights.append(w)

    if not deltas:
        raise ValueError("Empty manifest for DeltaC drift audit")
    delta = np.concatenate(deltas, axis=0)
    weight = np.concatenate(weights, axis=0)
    mean = delta.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum(delta**2, axis=(1, 2)))))
    mean_norm = float(np.linalg.norm(mean))
    ratio = mean_norm / max(rms, 1e-12)

    weight = np.clip(weight, 0.0, None)
    if float(weight.sum()) <= 0:
        weighted_mean = mean.copy()
    else:
        weighted_mean = np.tensordot(weight / weight.sum(), delta, axes=(0, 0))
    weighted_norm = float(np.linalg.norm(weighted_mean))
    weighted_ratio = weighted_norm / max(rms, 1e-12)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "C_stage1_anchor.npy", c_value)
    np.save(out_dir / "DeltaC_mean.npy", mean)
    np.save(out_dir / "DeltaC_alpha_weighted_mean.npy", weighted_mean)
    np.save(out_dir / "C_eff_unweighted.npy", c_value + mean)
    np.save(out_dir / "C_eff_alpha_weighted.npy", c_value + weighted_mean)
    summary = {
        "edges": int(delta.shape[0]),
        "delta_rms_frobenius": rms,
        "mean_delta_frobenius": mean_norm,
        "mean_shift_ratio": ratio,
        "alpha_weighted_mean_delta_frobenius": weighted_norm,
        "alpha_weighted_mean_shift_ratio": weighted_ratio,
        "interpretation": (
            "Large ratios mean DeltaC carries a population-wide offset; in that case call raw C "
            "the Stage-C anchor and report C_eff rather than claiming raw C is the final average field."
        ),
    }
    (out_dir / "delta_c_drift.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            audit_delta_c(cfg, args.checkpoint, args.manifest, args.out, args.device),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
