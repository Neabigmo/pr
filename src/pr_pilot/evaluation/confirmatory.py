"""Pre-registered confirmatory inference for the mini-pilot.

Only four scientific hypotheses enter the primary Holm family:
H1 full DM-ICF improves canonical-interface normalized NLL over dual priors;
H2 full DM-ICF improves over *both* partner-blind and geometry-only controls;
H3 the contextual field (DeltaC+alpha) improves over global-C-only;
H4 partner scrambling increases interface NLL in the full model.

All observations are first reduced to the biological-complex level, then averaged
across training seeds. Residues are never treated as independent samples.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

from pr_pilot.evaluation.battery import holm_adjust, paired_bootstrap


FULL = "F_joint_full1000"
DUAL = "B_dual_prior_full1000"
GLOBAL_C = "C_global_C_full1000"
CONTEXT = "E_alpha_full1000"
PARTNER_BLIND = "partner_blind"
GEOMETRY_ONLY = "geometry_only"


def _read_runs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    required = {"model", "seed", "run_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    if df[["model", "seed"]].astype(str).duplicated().any():
        raise ValueError(f"Duplicate model/seed rows in {path}")
    return df


def _interface_composite(run_dir: Path) -> pd.Series:
    """Equal P/R composite of canonical-interface entropy-normalized NLL."""
    polymer_values: dict[str, pd.Series] = {}
    for polymer, alphabet, filename in [
        ("protein", 20, "conditional_protein.tsv"),
        ("rna", 4, "conditional_rna.tsv"),
    ]:
        frame = pd.read_csv(run_dir / "core" / filename, sep="\t")
        if "is_interface" not in frame:
            raise ValueError(f"{run_dir}/{filename} lacks canonical interface labels")
        frame = frame[frame["is_interface"].astype(bool)].copy()
        if frame.empty:
            raise ValueError(f"No canonical interface rows in {run_dir}/{filename}")
        frame["normalized_nll"] = -frame["native_log_probability"].astype(float) / np.log(alphabet)
        polymer_values[polymer] = frame.groupby("sample_id")["normalized_nll"].mean()
    aligned = pd.concat(polymer_values, axis=1).dropna()
    if aligned.empty:
        raise ValueError(f"No P/R paired interface complexes under {run_dir}")
    return aligned.mean(axis=1).rename("interface_normalized_nll")


def _seed_averaged_metric(runs: pd.DataFrame, model: str) -> pd.Series:
    rows = []
    subset = runs[runs.model.astype(str) == model]
    if subset.empty:
        raise ValueError(f"Missing model {model!r} in run manifest")
    for record in subset.itertuples(index=False):
        series = _interface_composite(Path(record.run_dir))
        for sample_id, value in series.items():
            rows.append(
                {
                    "seed": int(record.seed),
                    "sample_id": str(sample_id),
                    "value": float(value),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.groupby("sample_id")["value"].mean().rename(model)


def _one_sided_wilcoxon_positive(diff: pd.Series) -> float:
    values = diff.dropna().to_numpy(float)
    if len(values) < 2:
        raise ValueError("Need >=2 paired complexes")
    if np.allclose(values, 0):
        return 1.0
    try:
        return float(
            stats.wilcoxon(
                values,
                np.zeros_like(values),
                alternative="greater",
                zero_method="wilcox",
            ).pvalue
        )
    except ValueError:
        return 1.0


def _benefit(reference: pd.Series, competitor: pd.Series, seed: int) -> dict:
    """Positive means the reference model has lower NLL and is therefore better."""
    aligned = pd.concat(
        [reference.rename("reference"), competitor.rename("competitor")], axis=1
    ).dropna()
    diff = aligned["competitor"] - aligned["reference"]
    boot = paired_bootstrap(
        diff,
        pd.Series(np.zeros(len(diff)), index=diff.index),
        resamples=10000,
        seed=seed,
    )
    return {
        "n_complexes": int(len(diff)),
        "effect_positive_favors_reference": float(diff.mean()),
        "ci_low": float(boot["ci_low"]),
        "ci_high": float(boot["ci_high"]),
        "primary_p_one_sided_wilcoxon": _one_sided_wilcoxon_positive(diff),
    }


def _scramble_seed_average(runs: pd.DataFrame, model: str) -> pd.Series:
    rows = []
    subset = runs[runs.model.astype(str) == model]
    if subset.empty:
        raise ValueError(f"Missing model {model!r} for partner-scramble H4")
    for record in subset.itertuples(index=False):
        path = Path(record.run_dir) / "core" / "partner_scramble.tsv"
        frame = pd.read_csv(path, sep="\t")
        series = frame.groupby("sample_id")["delta_nll"].mean()
        for sample_id, value in series.items():
            rows.append(
                {
                    "seed": int(record.seed),
                    "sample_id": str(sample_id),
                    "value": float(value),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.groupby("sample_id")["value"].mean().rename("scramble_delta_nll")


def run_confirmatory_statistics(
    component_runs_path: Path,
    control_runs_path: Path,
    out_dir: Path,
    seed: int = 20260905,
) -> dict:
    component = _read_runs(component_runs_path)
    controls = _read_runs(control_runs_path)

    full = _seed_averaged_metric(component, FULL)
    dual = _seed_averaged_metric(component, DUAL)
    global_c = _seed_averaged_metric(component, GLOBAL_C)
    context = _seed_averaged_metric(component, CONTEXT)
    blind = _seed_averaged_metric(controls, PARTNER_BLIND)
    geometry = _seed_averaged_metric(controls, GEOMETRY_ONLY)

    h1 = _benefit(full, dual, seed + 1)
    h3 = _benefit(context, global_c, seed + 3)
    h2_blind = _benefit(full, blind, seed + 20)
    h2_geometry = _benefit(full, geometry, seed + 21)
    # H2 is an intersection-union claim: full must beat BOTH controls.  Using the
    # larger component p-value is conservative and yields one H2 p-value for Holm.
    h2 = {
        "n_complexes": min(h2_blind["n_complexes"], h2_geometry["n_complexes"]),
        "effect_minimum_across_controls": min(
            h2_blind["effect_positive_favors_reference"],
            h2_geometry["effect_positive_favors_reference"],
        ),
        "primary_p_one_sided_wilcoxon": max(
            h2_blind["primary_p_one_sided_wilcoxon"],
            h2_geometry["primary_p_one_sided_wilcoxon"],
        ),
        "vs_partner_blind": h2_blind,
        "vs_geometry_only": h2_geometry,
        "combination": "intersection-union; max component p-value",
    }

    scramble = _scramble_seed_average(component, FULL)
    h4_boot = paired_bootstrap(
        scramble,
        pd.Series(np.zeros(len(scramble)), index=scramble.index),
        resamples=10000,
        seed=seed + 4,
    )
    h4 = {
        "n_complexes": int(len(scramble)),
        "mean_interface_delta_nll": float(scramble.mean()),
        "ci_low": float(h4_boot["ci_low"]),
        "ci_high": float(h4_boot["ci_high"]),
        "primary_p_one_sided_wilcoxon": _one_sided_wilcoxon_positive(scramble),
        "interpretation": "model-interventional partner dependence; not biological causality",
    }

    hypotheses = {
        "H1_full_vs_dual_prior": h1,
        "H2_full_vs_both_partner_controls": h2,
        "H3_contextual_field_vs_global_C": h3,
        "H4_partner_scramble_interface_delta_nll": h4,
    }
    raw = {
        name: float(values["primary_p_one_sided_wilcoxon"])
        for name, values in hypotheses.items()
    }
    adjusted = holm_adjust(raw)
    for name in hypotheses:
        hypotheses[name]["holm_adjusted_p"] = float(adjusted[name])

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, values in hypotheses.items():
        rows.append(
            {
                "hypothesis": name,
                "raw_primary_p": raw[name],
                "holm_adjusted_p": adjusted[name],
                "n_complexes": values["n_complexes"],
            }
        )
    pd.DataFrame(rows).to_csv(
        out_dir / "confirmatory_hypotheses.tsv", sep="\t", index=False
    )
    payload = {
        "statistical_unit": "biological complex",
        "seed_aggregation": "per-complex mean across training seeds before inference",
        "primary_family_size": 4,
        "multiple_testing": "Holm across H1-H4 only",
        "hypotheses": hypotheses,
    }
    (out_dir / "confirmatory_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload
