"""Held-out 100-complex evaluation battery.

This module implements metric/statistics primitives and defines the mandatory test
registry. Model-specific prediction calls live in the evaluation runner so this
file can remain independently unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class MandatoryTest:
    name: str
    family: str
    primary: bool
    description: str


MANDATORY_TESTS = [
    MandatoryTest("core_sequence_metrics", "prediction", True, "Protein/RNA NLL, normalized NLL, recovery; interface/non-interface; macro/micro."),
    MandatoryTest("conditional_protein", "prediction", True, "RNA-known protein design against prior and ProteinMPNN where fair."),
    MandatoryTest("conditional_rna", "prediction", True, "Protein-known RNA design against prior and MPNN-fixbb/NA-MPNN where fair."),
    MandatoryTest("partner_scramble", "mechanism", True, "Composition-aware partner scrambling; interface vs non-interface DeltaNLL."),
    MandatoryTest("counterfactual_partner_mutation", "mechanism", True, "Single partner-token perturbation; local KL vs distance."),
    MandatoryTest("global_c_vs_pmi", "interpretability", True, "Post-hoc C vs empirical PMI with permutation null."),
    MandatoryTest("delta_c_context", "interpretability", True, "Residual magnitude, geometry stratification, contextual reversal cases."),
    MandatoryTest("alpha_relevance", "interpretability", False, "Entropy/effective neighbours/distance relation/non-nearest relevance."),
    MandatoryTest("coordinate_noise_robustness", "robustness", False, "0/0.05/0.10/0.20 A coordinate noise."),
    MandatoryTest("pr_edge_dropout_robustness", "robustness", False, "0/5/10/20% cross-edge removal."),
    MandatoryTest("partner_hide_robustness", "robustness", False, "0/10/20/40% known partner token hiding."),
    MandatoryTest("decoding_order_sensitivity", "inference", False, "Mixed random order variance and directional bias."),
    MandatoryTest("spir_ablation", "inference", True, "No refinement vs one SPIR vs repeated refinement."),
    MandatoryTest("calibration", "reliability", False, "ECE/Brier/reliability interface and non-interface."),
    MandatoryTest("sequence_collapse_audit", "reliability", False, "Composition, charge proxy, entropy and near-duplicate candidate rate."),
    MandatoryTest("ablation_ladder", "ablation", True, "Scratch -> priors -> C -> DeltaC -> alpha -> joint -> SPIR."),
    MandatoryTest("data_efficiency", "ablation", False, "10/25/50/100% complex data: scratch vs dual prior."),
    MandatoryTest("seed_stability", "reproducibility", False, "Across-seed stability for metrics, C, DeltaC summaries and alpha."),
]


def token_metrics(df: pd.DataFrame, alphabet_size: int) -> dict[str, float]:
    """Compute raw and normalized NLL plus recovery from standardized predictions."""
    required = {"native_log_probability", "native_token", "predicted_token"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction table missing {sorted(missing)}")
    nll = float(-df["native_log_probability"].mean())
    return {
        "n": int(len(df)),
        "nll": nll,
        "normalized_nll": nll / math.log(alphabet_size),
        "perplexity": float(math.exp(min(nll, 50.0))),
        "recovery": float((df["native_token"] == df["predicted_token"]).mean()),
    }


def macro_by_complex(df: pd.DataFrame, alphabet_size: int) -> pd.DataFrame:
    rows = []
    for sample_id, part in df.groupby("sample_id", sort=True):
        metrics = token_metrics(part, alphabet_size)
        metrics["sample_id"] = sample_id
        rows.append(metrics)
    return pd.DataFrame(rows)


def expected_calibration_error(correct: np.ndarray, confidence: np.ndarray, bins: int = 15) -> float:
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if correct.shape != confidence.shape:
        raise ValueError("correct/confidence shape mismatch")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_multiclass(native_probability: np.ndarray, max_probability: np.ndarray, correct: np.ndarray) -> float:
    """Lightweight confidence Brier proxy for standardized baseline outputs.

    Full model evaluations should additionally compute true multiclass Brier from
    complete logits. This proxy exists so external baselines can still be audited
    when only native/max probabilities are exportable.
    """
    correct = np.asarray(correct, dtype=float)
    max_probability = np.asarray(max_probability, dtype=float)
    return float(np.mean((max_probability - correct) ** 2))


def paired_bootstrap(
    a: pd.Series,
    b: pd.Series,
    resamples: int = 10000,
    seed: int = 20260905,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, float]:
    """Paired target-level bootstrap for difference a-b."""
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = (aligned["a"] - aligned["b"]).to_numpy(float)
    if len(diff) < 2:
        raise ValueError("Need >=2 paired targets")
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, dtype=float)
    n = len(diff)
    for i in range(resamples):
        boot[i] = statistic(diff[rng.integers(0, n, size=n)])
    return {
        "n": n,
        "effect": float(statistic(diff)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "bootstrap_p_two_sided": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
    }


def paired_wilcoxon(a: pd.Series, b: pd.Series) -> dict[str, float]:
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    result = stats.wilcoxon(aligned["a"], aligned["b"], alternative="two-sided", zero_method="wilcox")
    return {"n": int(len(aligned)), "statistic": float(result.statistic), "p": float(result.pvalue)}


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def bh_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1], reverse=True)
    m = len(items)
    adjusted = {}
    running = 1.0
    for reverse_rank, (name, p) in enumerate(items, start=1):
        rank = m - reverse_rank + 1
        value = min(1.0, p * m / rank)
        running = min(running, value)
        adjusted[name] = running
    return adjusted


def partner_scramble_delta(native_nll: pd.Series, scrambled_nll: pd.Series) -> pd.Series:
    aligned = pd.concat([native_nll.rename("native"), scrambled_nll.rename("scrambled")], axis=1).dropna()
    return aligned["scrambled"] - aligned["native"]


def empirical_pmi(counts_20x4: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    """Compute contact PMI from experimental counts; never use for C initialization."""
    counts = np.asarray(counts_20x4, dtype=float)
    if counts.shape != (20, 4):
        raise ValueError("Expected 20x4 counts")
    counts = counts + pseudocount
    p = counts / counts.sum()
    pa = p.sum(axis=1, keepdims=True)
    pb = p.sum(axis=0, keepdims=True)
    return np.log(p / (pa * pb))


def matrix_correlations(C: np.ndarray, pmi: np.ndarray) -> dict[str, float]:
    c = np.asarray(C, dtype=float).ravel()
    p = np.asarray(pmi, dtype=float).ravel()
    pearson = stats.pearsonr(c, p)
    spearman = stats.spearmanr(c, p)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def effective_neighbor_number(alpha: np.ndarray) -> float:
    a = np.asarray(alpha, dtype=float)
    a = a / a.sum()
    entropy = -(a * np.log(np.clip(a, 1e-12, 1.0))).sum()
    return float(np.exp(entropy))


def mandatory_test_registry() -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in MANDATORY_TESTS])
