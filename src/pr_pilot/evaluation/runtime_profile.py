"""Development-only profiler for freezing final evaluation compute budget."""
from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from pr_pilot.evaluation.runner import _move, load_model
from pr_pilot.inference.sampler import sample_joint
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


def _representative_rows(table: ManifestTable, n: int) -> list:
    rows = list(table.rows())
    if len(rows) <= n:
        return rows
    lengths = []
    for row in rows:
        p = str(row.raw.get("protein_sequence", "")).replace("|", "")
        r = str(row.raw.get("rna_sequence", "")).replace("|", "")
        lengths.append(len(p) + len(r))
    order = np.argsort(lengths)
    quantiles = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [rows[int(order[q])] for q in quantiles]


@torch.no_grad()
def profile_inference_budget(
    cfg: dict,
    checkpoint: Path,
    development_manifest: Path,
    out_dir: Path,
    device: str | None = None,
    profile_candidates: int = 2,
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(checkpoint, cfg, dev)
    adapter = _adapter(cfg)
    n_targets = int(cfg["evaluation"].get("runtime_profile_complexes", 10))
    rows = _representative_rows(ManifestTable(development_manifest), n_targets)
    seed = int(cfg["experiment"]["pilot_seed"])
    icfg = cfg["inference"]
    spir = icfg["spir"]
    timings = []

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for k, row in enumerate(rows):
        sample = _move(load_complex_row(adapter, row), dev)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        start = time.perf_counter()
        sample_joint(
            model,
            sample,
            profile_candidates,
            float(icfg["initial_temperature"]),
            seed + k,
            spir_enabled=True,
            spir_reopen_fraction=float(spir["reopen_fraction"]),
            spir_temperature=float(spir["temperature"]),
            spir_cycles=1,
            reverse_direction_fraction=float(spir["reverse_direction_fraction"]),
            order_mode="mixed",
        )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elapsed = time.perf_counter() - start
        tokens = int(sample.protein.valid.sum() + sample.rna.valid.sum())
        timings.append(
            {
                "sample_id": sample.sample_id,
                "tokens": tokens,
                "seconds": elapsed,
                "candidates": profile_candidates,
                "seconds_per_candidate": elapsed / profile_candidates,
            }
        )

    frame = pd.DataFrame(timings)
    sec_per_candidate = float(frame.seconds_per_candidate.median())
    ablation_budget = int(cfg["evaluation"].get("ablation_candidate_budget", 16))
    # full_suite currently evaluates 3 SPIR-cycle conditions + 3 order modes x
    # with/without SPIR = 9 candidate-generating conditions on 100 complexes.
    estimated_candidates = 100 * 9 * ablation_budget
    estimated_hours = sec_per_candidate * estimated_candidates / 3600.0
    peak_mb = (
        float(torch.cuda.max_memory_allocated(dev) / 1024**2)
        if dev.type == "cuda"
        else None
    )
    summary = {
        "development_only": True,
        "profile_complexes": len(frame),
        "profile_candidates_per_complex": int(profile_candidates),
        "median_seconds_per_candidate": sec_per_candidate,
        "p90_seconds_per_candidate": float(frame.seconds_per_candidate.quantile(0.9)),
        "peak_allocated_mb": peak_mb,
        "tier_b_ablation_candidate_budget": ablation_budget,
        "estimated_tier_b_candidate_generations": estimated_candidates,
        "estimated_tier_b_gpu_hours_from_median": estimated_hours,
        "warning": (
            "Estimate covers joint candidate-generation calls only; teacher-forced, "
            "counterfactual and external-structure checks add separate cost. Freeze "
            "the final budget before final100 inspection."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "per_complex_profile.tsv", sep="\t", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
