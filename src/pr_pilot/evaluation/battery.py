"""Held-out evaluation primitives and pre-registered test hierarchy.

Only five hypothesis families are confirmatory and enter the primary Holm
correction.  The broader battery remains mandatory for characterization but is
secondary/exploratory; a rich test suite must not be turned into dozens of
nominally primary hypotheses after seeing the final 100 complexes.
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
    evidence_level: str = "secondary"


MANDATORY_TESTS = [
    # Confirmatory family: exactly the five hypotheses frozen in pilot.yaml.
    MandatoryTest(
        "H1_full_vs_dual_prior_interface_normalized_nll",
        "confirmatory",
        True,
        "Full DM-ICF versus dual structural priors; primary endpoint is complex-level interface normalized NLL.",
        "confirmatory",
    ),
    MandatoryTest(
        "H2_full_vs_partner_blind_and_geometry_only",
        "confirmatory",
        True,
        "Full DM-ICF versus same-data partner-blind and geometry-only capacity controls.",
        "confirmatory",
    ),
    MandatoryTest(
        "H3_contextual_field_vs_global_C_only",
        "confirmatory",
        True,
        "Contextual C+DeltaC+alpha field versus the global-C-only component model.",
        "confirmatory",
    ),
    MandatoryTest(
        "H4_partner_scramble_interface_delta_nll",
        "confirmatory",
        True,
        "Composition-preserving partner scrambling; interface DeltaNLL is model-interventional evidence of partner dependence.",
        "confirmatory",
    ),
    MandatoryTest(
        "H5_alpha_top_edge_vs_distance_matched_edge_removal",
        "confirmatory",
        True,
        "Remove top-alpha edges versus distance-matched lower-alpha edges and compare target NLL degradation.",
        "confirmatory",
    ),
    # Descriptive prediction metrics.
    MandatoryTest("core_sequence_metrics", "prediction", False, "Protein/RNA raw and entropy-normalized NLL and recovery, split by canonical interface/non-interface.", "descriptive"),
    MandatoryTest("conditional_protein", "prediction", False, "RNA-known Protein design; ProteinMPNN is a one-sided structural reference.", "secondary"),
    MandatoryTest("conditional_rna", "prediction", False, "Protein-known RNA design; MPNN-fixbb/NA-MPNN is a one-sided structural reference.", "secondary"),
    MandatoryTest("joint_mixed_order", "prediction", False, "Teacher-forced sequential pseudo-NLL under fixed mixed and directional orders.", "secondary"),
    # Mechanistic / interpretability analyses.
    MandatoryTest("counterfactual_partner_mutation", "mechanism", False, "Bidirectional single partner-token perturbation; local KL versus distance.", "secondary"),
    MandatoryTest("global_c_vs_pmi", "interpretability", False, "Post-hoc C anchor versus independent heavy-atom PMI; PMI never initializes the primary model.", "secondary"),
    MandatoryTest("c_seed_stability", "interpretability", False, "Across-seed correlation and sign stability of learned C.", "secondary"),
    MandatoryTest("delta_c_context", "interpretability", False, "DeltaC magnitude, mean-drift audit, geometry stratification and contextual reversal cases.", "secondary"),
    MandatoryTest("alpha_relevance", "interpretability", False, "Alpha entropy, effective neighbours, distance relation and non-nearest relevance.", "secondary"),
    # Robustness / inference.
    MandatoryTest("coordinate_noise_robustness", "robustness", False, "0/0.05/0.10/0.20 A coordinate perturbation sensitivity.", "exploratory"),
    MandatoryTest("pr_edge_dropout_robustness", "robustness", False, "0/5/10/20% cross-edge removal sensitivity.", "exploratory"),
    MandatoryTest("partner_hide_robustness", "robustness", False, "0/10/20/40% known partner-token hiding.", "exploratory"),
    MandatoryTest("geometry_permutation", "robustness", False, "Permute PR geometry among edges while preserving graph topology.", "secondary"),
    MandatoryTest("decoding_order_sensitivity", "inference", False, "Mixed/protein-first/RNA-first order sensitivity.", "secondary"),
    MandatoryTest("spir_ablation", "inference", False, "No refinement versus one-pass SPIR versus repeated refinement using pre-registered candidate budgets.", "secondary"),
    MandatoryTest("sequence_collapse_audit", "inference", False, "Candidate diversity, composition and pre/post-SPIR collapse.", "secondary"),
    MandatoryTest("calibration", "reliability", False, "ECE/Brier/reliability for Protein and RNA separately.", "secondary"),
    # Fairness / data / reproducibility.
    MandatoryTest("official_from_scratch_baselines", "fairness", False, "Exact frozen 1,000 single-molecule IDs; official pinned code; random initialization; full-1,000 refit.", "secondary"),
    MandatoryTest("published_checkpoint_reference", "fairness", False, "Published pretrained checkpoints are a separate reference track and never pooled with primary from-scratch comparisons.", "reference"),
    MandatoryTest("parameter_compute_report", "fairness", False, "Trainable/total parameters, data passes/tokens, peak memory, wall time and GPU-hours.", "descriptive"),
    MandatoryTest("strict_ood_shift_audit", "data", False, "Dev versus bilateral strict-OOD test distributions; final test is never described as IID random.", "descriptive"),
    MandatoryTest("ablation_ladder", "ablation", False, "Scratch -> priors -> C -> DeltaC -> alpha -> joint, interpreted together with H1-H3.", "secondary"),
    MandatoryTest("rich_vs_distance_geometry", "ablation", False, "Rich multi-atom/frame geometry versus distance-only with matched data.", "secondary"),
    MandatoryTest("fixed_vs_learned_alpha", "ablation", False, "Distance-prior alpha versus learned relational relevance.", "secondary"),
    MandatoryTest("fixed_training_pmi_baseline", "ablation", False, "Separate fixed-PMI statistical baseline; primary C remains randomly initialized.", "secondary"),
    MandatoryTest("data_efficiency", "ablation", False, "Nested identical 10/25/50/100% complex subsets.", "secondary"),
    MandatoryTest("seed_stability", "reproducibility", False, "At least three primary training seeds; report training-seed and target variance separately.", "descriptive"),
    MandatoryTest("complex_level_statistics", "statistics", False, "Complex is the statistical unit; paired bootstrap and paired non-parametric secondary tests.", "methodological"),
    MandatoryTest("independent_structure_prediction", "external", False, "Matched predictor/MSA/template/seed budget; predictor confidence is not binding energy.", "exploratory"),
]


def token_metrics(df: pd.DataFrame, alphabet_size: int) -> dict[str, float]:
    required = {"native_log_probability", "native_token", "predicted_token"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction table missing {sorted(missing)}")
    if len(df) == 0:
        return {
            "n": 0,
            "nll": float("nan"),
            "normalized_nll": float("nan"),
            "perplexity": float("nan"),
            "recovery": float("nan"),
        }
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


def expected_calibration_error(
    correct: np.ndarray, confidence: np.ndarray, bins: int = 15
) -> float:
    correct = np.asarray(correct, float)
    confidence = np.asarray(confidence, float)
    if correct.shape != confidence.shape:
        raise ValueError("correct/confidence shape mismatch")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < (hi if hi < 1 else hi + 1e-12))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_multiclass(
    native_probability: np.ndarray,
    max_probability: np.ndarray,
    correct: np.ndarray,
) -> float:
    """Confidence Brier proxy when complete external-baseline logits are unavailable."""
    _ = native_probability
    correct = np.asarray(correct, float)
    max_probability = np.asarray(max_probability, float)
    return float(np.mean((max_probability - correct) ** 2))


def native_probability_brier(native_probability: np.ndarray) -> float:
    p = np.asarray(native_probability, float)
    return float(np.mean((1.0 - p) ** 2))


def paired_bootstrap(
    a: pd.Series,
    b: pd.Series,
    resamples: int = 10000,
    seed: int = 20260905,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> dict[str, float]:
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = (aligned["a"] - aligned["b"]).to_numpy(float)
    if len(diff) < 2:
        raise ValueError("Need >=2 paired targets")
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(resamples)
    for i in range(resamples):
        boot[i] = statistic(diff[rng.integers(0, n, size=n)])
    return {
        "n": n,
        "effect": float(statistic(diff)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "bootstrap_p_two_sided": float(
            min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        ),
    }


def paired_wilcoxon(a: pd.Series, b: pd.Series) -> dict[str, float]:
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    result = stats.wilcoxon(
        aligned["a"], aligned["b"], alternative="two-sided", zero_method="wilcox"
    )
    return {
        "n": int(len(aligned)),
        "statistic": float(result.statistic),
        "p": float(result.pvalue),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def bh_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1], reverse=True)
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 1.0
    for reverse_rank, (name, p) in enumerate(items, start=1):
        rank = m - reverse_rank + 1
        value = min(1.0, p * m / rank)
        running = min(running, value)
        adjusted[name] = running
    return adjusted


def partner_scramble_delta(
    native_nll: pd.Series, scrambled_nll: pd.Series
) -> pd.Series:
    aligned = pd.concat(
        [native_nll.rename("native"), scrambled_nll.rename("scrambled")], axis=1
    ).dropna()
    return aligned["scrambled"] - aligned["native"]


def empirical_pmi(counts_20x4: np.ndarray, pseudocount: float = 0.5) -> np.ndarray:
    counts = np.asarray(counts_20x4, float)
    if counts.shape != (20, 4):
        raise ValueError("Expected 20x4 counts")
    counts = counts + pseudocount
    p = counts / counts.sum()
    pa = p.sum(1, keepdims=True)
    pb = p.sum(0, keepdims=True)
    return np.log(p / (pa * pb))


def matrix_correlations(C: np.ndarray, pmi: np.ndarray) -> dict[str, float]:
    c = np.asarray(C, float).ravel()
    p = np.asarray(pmi, float).ravel()
    pearson = stats.pearsonr(c, p)
    spearman = stats.spearmanr(c, p)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def effective_neighbor_number(alpha: np.ndarray) -> float:
    a = np.asarray(alpha, float)
    a = a / a.sum()
    entropy = -(a * np.log(np.clip(a, 1e-12, 1))).sum()
    return float(np.exp(entropy))


def mandatory_test_registry() -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in MANDATORY_TESTS])
