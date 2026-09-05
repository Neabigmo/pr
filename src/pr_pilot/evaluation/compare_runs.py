"""Broad cross-model/cross-seed descriptive statistics.

This module is intentionally *not* the confirmatory hypothesis engine.  H1-H4 and
Holm correction are implemented in ``confirmatory.py``.  Here every generic metric
is secondary/exploratory so a large metric table cannot silently expand the
primary multiple-testing family.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from pr_pilot.evaluation.battery import bh_adjust, paired_bootstrap


@dataclass(frozen=True)
class MetricSpec:
    name: str
    table: str
    subset: str | None
    value: str
    higher_is_better: bool
    primary: bool = False


METRICS = [
    MetricSpec("protein_conditional_nll", "core/conditional_protein.tsv", None, "nll", False),
    MetricSpec("protein_conditional_recovery", "core/conditional_protein.tsv", None, "recovery", True),
    MetricSpec("rna_conditional_nll", "core/conditional_rna.tsv", None, "nll", False),
    MetricSpec("rna_conditional_recovery", "core/conditional_rna.tsv", None, "recovery", True),
    MetricSpec("joint_protein_nll", "core/joint_teacher_forced.tsv", "protein", "nll", False),
    MetricSpec("joint_protein_recovery", "core/joint_teacher_forced.tsv", "protein", "recovery", True),
    MetricSpec("joint_rna_nll", "core/joint_teacher_forced.tsv", "rna", "nll", False),
    MetricSpec("joint_rna_recovery", "core/joint_teacher_forced.tsv", "rna", "recovery", True),
    MetricSpec("protein_partner_scramble_delta_nll", "core/partner_scramble.tsv", "protein", "delta_nll", True),
    MetricSpec("rna_partner_scramble_delta_nll", "core/partner_scramble.tsv", "rna", "delta_nll", True),
]


def _token_metric(df: pd.DataFrame, value: str) -> pd.Series:
    if value == "nll":
        return -df.groupby("sample_id")["native_log_probability"].mean()
    if value == "recovery":
        correct = (
            df["native_token"].astype(int) == df["predicted_token"].astype(int)
        ).astype(float)
        return correct.groupby(df["sample_id"]).mean()
    raise ValueError(value)


def _per_complex(run_dir: Path, spec: MetricSpec) -> pd.Series:
    path = run_dir / spec.table
    if not path.exists():
        raise FileNotFoundError(f"Missing required evaluation table: {path}")
    df = pd.read_csv(path, sep="\t")
    if spec.subset is not None:
        if "polymer" in df.columns:
            df = df[df["polymer"].astype(str) == spec.subset]
        elif "direction" in df.columns:
            df = df[df["direction"].astype(str) == spec.subset]
        else:
            raise ValueError(f"Cannot apply subset {spec.subset!r} to {path}")
    if spec.value in {"nll", "recovery"}:
        return _token_metric(df, spec.value).rename(spec.name)
    if spec.value not in df.columns:
        raise ValueError(f"{path} lacks {spec.value}")
    return df.groupby("sample_id")[spec.value].mean().rename(spec.name)


def load_run_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    required = {"model", "seed", "run_dir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Run manifest missing columns: {sorted(missing)}")
    if df[["model", "seed"]].astype(str).duplicated().any():
        raise ValueError("Each model/seed pair must appear exactly once")
    df["run_dir"] = df["run_dir"].map(lambda x: str(Path(str(x)).resolve()))
    return df


def collect_metric(
    manifest: pd.DataFrame, spec: MetricSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for record in manifest.itertuples(index=False):
        series = _per_complex(Path(record.run_dir), spec)
        for sample_id, value in series.items():
            rows.append(
                {
                    "model": str(record.model),
                    "seed": int(record.seed),
                    "sample_id": str(sample_id),
                    "metric": spec.name,
                    "value": float(value),
                }
            )
    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        raise ValueError(f"No values collected for {spec.name}")
    per_model = (
        per_seed.groupby(["model", "sample_id", "metric"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "seed_mean"})
    )
    return per_seed, per_model


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if np.allclose(diff, 0):
        return 1.0
    try:
        return float(
            stats.wilcoxon(
                a, b, zero_method="wilcox", alternative="two-sided"
            ).pvalue
        )
    except ValueError:
        return 1.0


def compare_models(
    manifest_path: Path,
    reference_model: str,
    out_dir: Path,
    *,
    bootstrap_resamples: int = 10000,
    seed: int = 20260905,
) -> dict:
    manifest = load_run_manifest(manifest_path)
    models = sorted(manifest["model"].astype(str).unique())
    if reference_model not in models:
        raise ValueError(f"Reference model {reference_model!r} not in {models}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_seed_tables = []
    all_model_tables = []
    comparisons = []
    raw_exploratory_p: dict[str, float] = {}

    for spec in METRICS:
        per_seed, per_model = collect_metric(manifest, spec)
        all_seed_tables.append(per_seed)
        all_model_tables.append(per_model)
        ref = per_model[per_model.model == reference_model].set_index("sample_id")[
            "seed_mean"
        ]
        for competitor in models:
            if competitor == reference_model:
                continue
            other = per_model[per_model.model == competitor].set_index("sample_id")[
                "seed_mean"
            ]
            aligned = pd.concat(
                [ref.rename("reference"), other.rename("other")],
                axis=1,
                join="inner",
            ).dropna()
            if len(aligned) < 2:
                continue
            raw_diff = aligned["reference"] - aligned["other"]
            benefit_diff = raw_diff if spec.higher_is_better else -raw_diff
            boot = paired_bootstrap(
                benefit_diff,
                pd.Series(np.zeros(len(benefit_diff)), index=benefit_diff.index),
                resamples=bootstrap_resamples,
                seed=seed + len(comparisons),
            )
            wilcoxon_p = _paired_wilcoxon(
                benefit_diff.to_numpy(float), np.zeros(len(benefit_diff), dtype=float)
            )
            key = f"{spec.name}:{reference_model}_vs_{competitor}"
            record = {
                "comparison": key,
                "metric": spec.name,
                "reference_model": reference_model,
                "other_model": competitor,
                "higher_is_better": spec.higher_is_better,
                "n_complexes": int(len(aligned)),
                "reference_mean": float(aligned.reference.mean()),
                "other_mean": float(aligned.other.mean()),
                "raw_reference_minus_other": float(raw_diff.mean()),
                "benefit_effect_positive_favors_reference": float(benefit_diff.mean()),
                "benefit_ci_low": float(boot["ci_low"]),
                "benefit_ci_high": float(boot["ci_high"]),
                "bootstrap_p": float(boot["bootstrap_p_two_sided"]),
                "wilcoxon_p": wilcoxon_p,
                "primary": False,
            }
            comparisons.append(record)
            raw_exploratory_p[key] = record["bootstrap_p"]

    comparison_df = pd.DataFrame(comparisons)
    if len(comparison_df):
        bh = bh_adjust(raw_exploratory_p)
        comparison_df["bh_adjusted_exploratory"] = comparison_df["comparison"].map(bh)
    comparison_df.to_csv(
        out_dir / "paired_model_comparisons.tsv", sep="\t", index=False
    )

    per_seed_all = pd.concat(all_seed_tables, ignore_index=True)
    per_model_all = pd.concat(all_model_tables, ignore_index=True)
    per_seed_all.to_csv(
        out_dir / "per_seed_per_complex_metrics.tsv", sep="\t", index=False
    )
    per_model_all.to_csv(
        out_dir / "seed_averaged_per_complex_metrics.tsv", sep="\t", index=False
    )
    seed_summary = (
        per_seed_all.groupby(["model", "seed", "metric"])["value"]
        .mean()
        .reset_index()
        .groupby(["model", "metric"])["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    seed_summary.to_csv(
        out_dir / "training_seed_stability.tsv", sep="\t", index=False
    )

    summary = {
        "statistical_unit": "complex",
        "seed_aggregation": "per-complex mean across training seeds before paired inference",
        "bootstrap_resamples": int(bootstrap_resamples),
        "reference_model": reference_model,
        "models": models,
        "n_comparisons": int(len(comparison_df)),
        "multiple_testing": "Benjamini-Hochberg exploratory only",
        "confirmatory_statistics": "Use pr_pilot.evaluation.confirmatory / tools/run_confirmatory_statistics.py for H1-H4 Holm inference.",
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_models(
                args.runs,
                args.reference,
                args.out,
                bootstrap_resamples=args.bootstrap,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
