"""Development-only audit of whether contextual DeltaC silently becomes a new global C.

No regularizer is introduced here.  The audit quantifies the mean contextual
residual before final-test inspection so the interpretation of C can be stated
honestly.  If the residual has a large population mean, raw C is described as the
stage-1 global anchor and C_eff = C + E[DeltaC] is reported post hoc.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch

from pr_pilot.evaluation.runner import _move, load_model
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row


@torch.no_grad()
def audit_delta_c_drift(
    cfg: dict,
    checkpoint: Path,
    manifest: Path,
    out_dir: Path,
    device: str | None = None,
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(checkpoint, cfg, dev)
    g = cfg["geometry"]
    adapter = GemmiStructureAdapter(
        int(g["rbf_bins"]),
        int(g["intra_max_neighbors"]),
        float(g["pr_cutoff_angstrom"]),
        int(g["pr_max_neighbors"]),
        0.0,
        int(cfg["experiment"]["pilot_seed"]),
        bool(g["rich_pr_geometry"]),
    )

    sum_delta = np.zeros((20, 4), dtype=np.float64)
    sum_sq_fro = 0.0
    edge_count = 0
    weighted_sum = np.zeros((20, 4), dtype=np.float64)
    weight_sum = 0.0
    complex_count = 0

    for row in ManifestTable(manifest).rows():
        sample = _move(load_complex_row(adapter, row), dev)
        out = model(
            sample.protein.node_x,
            sample.protein.edge_index,
            sample.protein.edge_x,
            sample.rna.node_x,
            sample.rna.edge_index,
            sample.rna.edge_x,
            sample.pr,
            sample.protein.sequence,
            sample.rna.sequence,
            sample.protein.valid,
            sample.rna.valid,
            use_delta=True,
            learned_alpha=True,
        )
        delta = out["DeltaC"].float().cpu().numpy().astype(np.float64)
        if len(delta) == 0:
            continue
        sum_delta += delta.sum(axis=0)
        sum_sq_fro += float(np.square(delta).sum())
        edge_count += int(delta.shape[0])

        # Combine the two directional neighborhood relevance weights only for an
        # interpretable weighted population summary; they are normalized in
        # different target neighborhoods during actual prediction.
        weights = (
            0.5 * (out["alpha_p"].float() + out["alpha_r"].float())
        ).cpu().numpy().astype(np.float64)
        weighted_sum += (delta * weights[:, None, None]).sum(axis=0)
        weight_sum += float(weights.sum())
        complex_count += 1

    if edge_count == 0:
        raise ValueError("No PR edges available for DeltaC drift audit")

    mean_delta = sum_delta / edge_count
    rms_delta = float(np.sqrt(sum_sq_fro / edge_count))
    mean_fro = float(np.linalg.norm(mean_delta))
    mean_shift_ratio = mean_fro / max(rms_delta, 1e-12)
    weighted_mean = weighted_sum / max(weight_sum, 1e-12)

    c = model.dmicf.global_c().detach().float().cpu().numpy().astype(np.float64)
    c_eff = c + mean_delta
    summary = {
        "manifest": str(Path(manifest).resolve()),
        "complexes": complex_count,
        "edges": edge_count,
        "mean_delta_frobenius": mean_fro,
        "rms_edge_delta_frobenius": rms_delta,
        "mean_shift_ratio": float(mean_shift_ratio),
        "alpha_weighted_mean_delta_frobenius": float(np.linalg.norm(weighted_mean)),
        "interpretation": (
            "If mean_shift_ratio is substantial, call raw C the stage-1 global "
            "compatibility anchor and report C_eff=C+mean(DeltaC); do not retrofit "
            "a DeltaC penalty after inspecting final100."
        ),
        "final_test_used": False,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mean_DeltaC.npy", mean_delta.astype(np.float32))
    np.save(out_dir / "alpha_weighted_mean_DeltaC.npy", weighted_mean.astype(np.float32))
    np.save(out_dir / "C_stage1.npy", c.astype(np.float32))
    np.save(out_dir / "C_eff.npy", c_eff.astype(np.float32))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
