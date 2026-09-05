"""One-command primary evaluation suite for the frozen 100-complex OOD holdout."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import numpy as np
import pandas as pd
import torch

from pr_pilot.evaluation.battery import empirical_pmi, matrix_correlations, mandatory_test_registry
from pr_pilot.evaluation.empirical_contacts import analyze_empirical_pmi
from pr_pilot.evaluation.runner import _move, evaluate_holdout, load_model, score_conditional
from pr_pilot.evaluation.robustness import (
    alpha_edge_removal,
    calibration_table,
    evaluate_geometry_permutation,
    evaluate_partner_hiding,
    evaluate_pr_edge_removal,
)
from pr_pilot.inference.sampler import sample_joint
from pr_pilot.model.dmicf import double_center
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row


def _adapter(cfg: dict, noise: float = 0.0, seed_offset: int = 0) -> GemmiStructureAdapter:
    g = cfg["geometry"]
    return GemmiStructureAdapter(
        int(g["rbf_bins"]),
        int(g["intra_max_neighbors"]),
        float(g["pr_cutoff_angstrom"]),
        int(g["pr_max_neighbors"]),
        noise,
        int(cfg["experiment"]["pilot_seed"]) + seed_offset,
        bool(g["rich_pr_geometry"]),
    )


def _candidate_metrics(candidates, sample, condition: str) -> pd.DataFrame:
    rows = []
    native_p = sample.protein.sequence.cpu().numpy()
    native_r = sample.rna.sequence.cpu().numpy()
    pairs = []
    for candidate in candidates:
        p = candidate.protein_tokens.cpu().numpy()
        r = candidate.rna_tokens.cpu().numpy()
        pre_p = candidate.pre_spir_protein.cpu().numpy()
        pre_r = candidate.pre_spir_rna.cpu().numpy()
        pairs.append((p.copy(), r.copy()))
        p_interface = sample.protein.interface.cpu().numpy()
        r_interface = sample.rna.interface.cpu().numpy()
        rows.append(
            {
                "sample_id": sample.sample_id,
                "candidate_id": candidate.candidate_id,
                "condition": condition,
                "order_mode": candidate.order_mode,
                "spir_cycles": candidate.spir_cycles,
                "spir_direction": candidate.spir_direction,
                "protein_recovery": float((p == native_p).mean()),
                "rna_recovery": float((r == native_r).mean()),
                "protein_interface_recovery": float((p[p_interface] == native_p[p_interface]).mean()) if p_interface.any() else np.nan,
                "rna_interface_recovery": float((r[r_interface] == native_r[r_interface]).mean()) if r_interface.any() else np.nan,
                "pre_to_post_protein_change": float((p != pre_p).mean()),
                "pre_to_post_rna_change": float((r != pre_r).mean()),
                "mean_generation_logprob": float(np.mean(candidate.token_logprobs)) if candidate.token_logprobs else np.nan,
            }
        )
    if len(pairs) > 1:
        protein_distances = []
        rna_distances = []
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                protein_distances.append(float((pairs[i][0] != pairs[j][0]).mean()))
                rna_distances.append(float((pairs[i][1] != pairs[j][1]).mean()))
        protein_diversity = float(np.mean(protein_distances))
        rna_diversity = float(np.mean(rna_distances))
    else:
        protein_diversity = rna_diversity = 0.0
    unique_fraction = len({(tuple(p), tuple(r)) for p, r in pairs}) / max(1, len(pairs))
    for row in rows:
        row.update(
            {
                "protein_pairwise_diversity": protein_diversity,
                "rna_pairwise_diversity": rna_diversity,
                "unique_pair_fraction": unique_fraction,
            }
        )
    return pd.DataFrame(rows)


def _permutation_null_interaction_c(
    c_interaction: np.ndarray,
    contacts: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict:
    """Permutation null preserving global AA and base marginals for heavy-atom contacts."""
    observed_counts = np.zeros((20, 4), dtype=int)
    for aa, base in zip(contacts.aa.astype(int), contacts.base.astype(int)):
        observed_counts[aa, base] += 1
    observed_pmi = empirical_pmi(observed_counts)
    observed_pmi_centered = double_center(torch.from_numpy(observed_pmi.astype(np.float32))).numpy()
    observed = matrix_correlations(c_interaction, observed_pmi_centered)["spearman_rho"]
    rng = np.random.default_rng(seed)
    values = []
    aa = contacts.aa.to_numpy(int)
    base = contacts.base.to_numpy(int)
    for _ in range(repeats):
        shuffled_base = base.copy()
        rng.shuffle(shuffled_base)
        counts = np.zeros((20, 4), dtype=int)
        for a, b in zip(aa, shuffled_base):
            counts[a, b] += 1
        pmi = empirical_pmi(counts)
        centered = double_center(torch.from_numpy(pmi.astype(np.float32))).numpy()
        values.append(matrix_correlations(c_interaction, centered)["spearman_rho"])
    values = np.asarray(values, dtype=float)
    p = (1 + np.sum(np.abs(values) >= abs(observed))) / (len(values) + 1)
    return {
        "observed_spearman": float(observed),
        "null_mean": float(values.mean()),
        "null_sd": float(values.std(ddof=1)),
        "empirical_two_sided_p": float(p),
        "repeats": int(repeats),
    }


def _numeric_shift(dev: pd.DataFrame, test: pd.DataFrame, column: str) -> dict | None:
    if column not in dev or column not in test:
        return None
    a = pd.to_numeric(dev[column], errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(test[column], errors="coerce").dropna().to_numpy(float)
    if len(a) < 2 or len(b) < 2:
        return None
    variance_sum = a.var(ddof=1) + b.var(ddof=1)
    pooled = math.sqrt(variance_sum / 2.0) if variance_sum > 0 else 0.0
    return {
        "dev_mean": float(a.mean()),
        "test_mean": float(b.mean()),
        "standardized_mean_difference": float((b.mean() - a.mean()) / pooled) if pooled else 0.0,
        "dev_n": len(a),
        "test_n": len(b),
    }


def dataset_shift_audit(dev_path: Path, test_path: Path) -> dict:
    dev = pd.read_csv(dev_path, sep="\t")
    test = pd.read_csv(test_path, sep="\t")
    out = {}
    for column in [
        "protein_length",
        "rna_length",
        "total_tokens",
        "resolution",
        "interface_residue_pairs",
        "interface_missing_fraction",
    ]:
        result = _numeric_shift(dev, test, column)
        if result is not None:
            out[column] = result
    for column in ["method", "rna_type", "origin", "source"]:
        if column in dev and column in test:
            categories = sorted(set(dev[column].astype(str)) | set(test[column].astype(str)))
            p = np.array([(dev[column].astype(str) == category).mean() for category in categories]) + 1e-8
            q = np.array([(test[column].astype(str) == category).mean() for category in categories]) + 1e-8
            p /= p.sum()
            q /= q.sum()
            m = (p + q) / 2
            js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
            out[column] = {"jensen_shannon_divergence_nats": float(js), "categories": categories}
    return out


@torch.no_grad()
def run_full_suite(
    cfg: dict,
    checkpoint: Path,
    test_manifest: Path,
    out_dir: Path,
    dev_manifest: Path | None = None,
    device: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mandatory_test_registry().to_csv(out_dir / "test_registry.tsv", sep="\t", index=False)

    core_dir = out_dir / "core"
    summary = evaluate_holdout(cfg, checkpoint, test_manifest, core_dir, device, "DMICF")
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(checkpoint, cfg, device_obj)
    table = ManifestTable(test_manifest)
    base_adapter = _adapter(cfg)
    seed = int(cfg["experiment"]["pilot_seed"])
    ecfg = cfg["evaluation"]

    edge_removal = []
    partner_hiding = []
    geometry_permutation = []
    alpha_ablation = []
    noise_rows = []
    spir_frames = []
    order_frames = []

    for row_index, row in enumerate(table.rows()):
        sample = _move(load_complex_row(base_adapter, row), device_obj)
        edge_removal.append(
            evaluate_pr_edge_removal(model, sample, [float(x) for x in ecfg["pr_edge_drop_levels"]], seed + row_index)
        )
        partner_hiding.append(
            evaluate_partner_hiding(model, sample, [float(x) for x in ecfg["partner_hide_levels"]], seed + row_index)
        )
        geometry_permutation.append(
            evaluate_geometry_permutation(model, sample, int(ecfg.get("geometry_permutation_repeats", 10)), seed + row_index)
        )
        alpha_ablation.append(alpha_edge_removal(model, sample, int(ecfg.get("alpha_ablation_max_targets", 32))))

        for sigma in [float(x) for x in ecfg["noise_levels_angstrom"]]:
            noisy_sample = _move(load_complex_row(_adapter(cfg, sigma, 10000 + row_index), row), device_obj)
            for direction in ["protein", "rna"]:
                frame = score_conditional(model, noisy_sample, direction, interface_only=True, seed=seed + row_index)
                noise_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "direction": direction,
                        "noise_angstrom": sigma,
                        "nll": float(-frame.native_log_probability.mean()),
                        "recovery": float((frame.native_token == frame.predicted_token).mean()),
                    }
                )

        icfg = cfg["inference"]
        spir_cfg = icfg["spir"]
        candidate_count = int(icfg["candidates_per_complex"])
        # SPIR ablation on identical mixed-order candidate budgets.
        for cycles in sorted({0, 1, int(ecfg.get("repeated_spir_cycles", 3))}):
            candidates = sample_joint(
                model,
                sample,
                candidate_count,
                float(icfg["initial_temperature"]),
                seed + row_index,
                spir_enabled=cycles > 0,
                spir_reopen_fraction=float(spir_cfg["reopen_fraction"]),
                spir_temperature=float(spir_cfg["temperature"]),
                spir_cycles=cycles,
                reverse_direction_fraction=float(spir_cfg["reverse_direction_fraction"]),
                order_mode="mixed",
            )
            spir_frames.append(_candidate_metrics(candidates, sample, f"spir_cycles_{cycles}"))

        # Initial-order sensitivity both before and after the primary one-pass SPIR.
        for order_mode in ["mixed", "protein_first", "rna_first"]:
            for use_spir in [False, True]:
                candidates = sample_joint(
                    model,
                    sample,
                    candidate_count,
                    float(icfg["initial_temperature"]),
                    seed + row_index,
                    spir_enabled=use_spir,
                    spir_reopen_fraction=float(spir_cfg["reopen_fraction"]),
                    spir_temperature=float(spir_cfg["temperature"]),
                    spir_cycles=1 if use_spir else 0,
                    reverse_direction_fraction=float(spir_cfg["reverse_direction_fraction"]),
                    order_mode=order_mode,
                )
                order_frames.append(
                    _candidate_metrics(
                        candidates,
                        sample,
                        f"{order_mode}__{'with_spir' if use_spir else 'no_spir'}",
                    )
                )

    tables = {
        "edge_removal": pd.concat(edge_removal, ignore_index=True),
        "partner_hiding": pd.concat(partner_hiding, ignore_index=True),
        "geometry_permutation": pd.concat(geometry_permutation, ignore_index=True),
        "alpha_edge_removal": pd.concat(alpha_ablation, ignore_index=True) if any(len(x) for x in alpha_ablation) else pd.DataFrame(),
        "coordinate_noise": pd.DataFrame(noise_rows),
        "spir_candidates": pd.concat(spir_frames, ignore_index=True),
        "order_sensitivity_candidates": pd.concat(order_frames, ignore_index=True),
    }
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)

    conditional_p = pd.read_csv(core_dir / "conditional_protein.tsv", sep="\t")
    conditional_r = pd.read_csv(core_dir / "conditional_rna.tsv", sep="\t")
    calibration = {
        "protein": calibration_table(conditional_p),
        "rna": calibration_table(conditional_r),
    }
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    empirical_dir = out_dir / "empirical_heavy_atom_pmi"
    empirical_summary = analyze_empirical_pmi(
        test_manifest,
        core_dir / "C_global_centered.npy",
        empirical_dir,
        cutoff=float(ecfg.get("empirical_contact_cutoff_angstrom", 5.0)),
    )
    contacts = pd.read_csv(empirical_dir / "heavy_atom_contacts.tsv", sep="\t")
    c_interaction = np.load(core_dir / "C_interaction_only.npy")
    permutation = _permutation_null_interaction_c(
        c_interaction,
        contacts,
        int(ecfg.get("pmi_permutation_repeats", 1000)),
        seed,
    )
    (out_dir / "C_PMI_permutation_null.json").write_text(json.dumps(permutation, indent=2), encoding="utf-8")

    shift = None
    if dev_manifest is not None:
        shift = dataset_shift_audit(dev_manifest, test_manifest)
        (out_dir / "strict_ood_shift.json").write_text(json.dumps(shift, indent=2), encoding="utf-8")

    final = {
        "core": summary,
        "calibration": calibration,
        "empirical_heavy_atom_PMI": empirical_summary,
        "C_PMI_permutation": permutation,
        "strict_ood_shift": shift,
        "status": "single-checkpoint primary battery complete; use compare_runs.py for cross-model/seed paired inference",
    }
    (out_dir / "suite_summary.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    import yaml

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run_full_suite(cfg, args.checkpoint, args.test, args.out, args.dev, args.device), indent=2))


if __name__ == "__main__":
    main()
