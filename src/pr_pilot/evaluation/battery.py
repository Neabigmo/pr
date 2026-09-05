"""Held-out 100-complex evaluation primitives and pre-registered test registry.

Only five comparisons are confirmatory/primary. The remaining battery is still
mandatory for a complete pilot but is classified secondary or exploratory so a
large collection of mechanistic diagnostics does not masquerade as dozens of
independent primary hypotheses.
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
    tier: str
    description: str


MANDATORY_TESTS = [
    MandatoryTest("core_sequence_metrics", "prediction", False, "secondary", "Protein/RNA raw and entropy-normalized NLL and recovery, canonical interface/non-interface, macro/micro."),
    MandatoryTest("conditional_protein", "prediction", False, "secondary", "RNA-known Protein design; ProteinMPNN is a one-sided external structural reference."),
    MandatoryTest("conditional_rna", "prediction", False, "secondary", "Protein-known RNA design; NA-MPNN is a one-sided external structural reference."),
    MandatoryTest("joint_mixed_order", "prediction", False, "secondary", "Sequential teacher-forced joint pseudo-NLL under fixed mixed/P-first/R-first orders."),
    MandatoryTest("full_vs_dual_prior_interface_nll", "confirmatory", True, "primary", "H1: full DM-ICF improves canonical-interface normalized NLL over dual structural priors."),
    MandatoryTest("full_vs_partner_identity_controls", "confirmatory", True, "primary", "H2: full DM-ICF improves over partner-blind and geometry-only capacity controls."),
    MandatoryTest("contextual_vs_global_c", "confirmatory", True, "primary", "H3: contextual C+DeltaC+alpha improves over C-only on canonical-interface normalized NLL."),
    MandatoryTest("partner_scramble", "confirmatory", True, "primary", "H4: partner scrambling degrades canonical-interface NLL more than matched non-interface/control effects."),
    MandatoryTest("alpha_edge_removal", "confirmatory", True, "primary", "H5: removing high-alpha edges causes larger model degradation than distance-matched lower-alpha removals; model-interventional evidence only."),
    MandatoryTest("counterfactual_partner_mutation", "mechanism", False, "secondary", "Bidirectional single-token partner perturbation and local KL-distance response."),
    MandatoryTest("global_c_vs_pmi", "interpretability", False, "secondary", "Post-hoc Stage-C anchor vs independent heavy-atom empirical PMI; PMI never initializes main C."),
    MandatoryTest("c_seed_stability", "interpretability", False, "secondary", "Across-seed correlation/sign stability of learned Stage-C anchor."),
    MandatoryTest("delta_c_context", "interpretability", False, "secondary", "Residual magnitude, geometry stratification, mean-drift audit and contextual reversals."),
    MandatoryTest("alpha_relevance", "interpretability", False, "secondary", "Entropy/effective neighbours/distance relation/non-nearest relevance."),
    MandatoryTest("coordinate_noise_robustness", "robustness", False, "exploratory", "0/0.05/0.10/0.20 A coordinate perturbation."),
    MandatoryTest("pr_edge_dropout_robustness", "robustness", False, "exploratory", "0/5/10/20% cross-edge removal."),
    MandatoryTest("partner_hide_robustness", "robustness", False, "exploratory", "0/10/20/40% known partner-token hiding."),
    MandatoryTest("geometry_permutation", "robustness", False, "secondary", "Permute rich PR geometry among edges while preserving topology."),
    MandatoryTest("decoding_order_sensitivity", "inference", False, "secondary", "Mixed-order variance plus Protein-first/RNA-first bias."),
    MandatoryTest("spir_ablation", "inference", False, "secondary", "No refinement vs one-pass SPIR vs repeated refinement with pre-registered budgets."),
    MandatoryTest("sequence_collapse_audit", "inference", False, "secondary", "Candidate diversity/composition and pre/post-SPIR collapse."),
    MandatoryTest("calibration", "reliability", False, "secondary", "ECE/Brier/reliability for Protein and RNA separately."),
    MandatoryTest("partner_blind_complex_control", "fairness", False, "secondary", "Same complex data/architecture, but cross-partner identities never affect logits."),
    MandatoryTest("geometry_only_capacity_control", "fairness", False, "secondary", "Comparable PR geometry capacity with no specific partner-token identity."),
    MandatoryTest("official_from_scratch_baselines", "fairness", False, "secondary", "Exact frozen single-molecule pools; random-init immutable ProteinMPNN/NA-MPNN references."),
    MandatoryTest("published_checkpoint_reference", "fairness", False, "exploratory", "Published pretrained checkpoints reported separately if used."),
    MandatoryTest("parameter_compute_report", "fairness", False, "secondary", "Trainable/total parameters, passes/tokens, peak memory, wall time and GPU-hours."),
    MandatoryTest("strict_ood_shift_audit", "data", False, "secondary", "Development vs strict bilateral-OOD final-test distribution shift."),
    MandatoryTest("ablation_ladder", "ablation", False, "secondary", "Scratch -> priors -> C -> DeltaC -> alpha -> joint -> SPIR."),
    MandatoryTest("rich_vs_distance_geometry", "ablation", False, "secondary", "Rich multi-atom/frame geometry vs distance-only."),
    MandatoryTest("fixed_vs_learned_alpha", "ablation", False, "secondary", "Distance-prior alpha vs learned relevance."),
    MandatoryTest("fixed_training_pmi_baseline", "ablation", False, "secondary", "Separate fixed-PMI statistical-potential reference; main C remains random-init learned."),
    MandatoryTest("data_efficiency", "ablation", False, "secondary", "Nested identical 10/25/50/100% complex subsets."),
    MandatoryTest("seed_stability", "reproducibility", False, "secondary", "At least three primary training seeds."),
    MandatoryTest("complex_level_statistics", "statistics", False, "secondary", "Complex-level paired bootstrap; Holm only across five confirmatory hypotheses."),
    MandatoryTest("independent_structure_prediction", "external", False, "exploratory", "Matched external structure-prediction budget; confidence is not binding energy."),
]


def token_metrics(df: pd.DataFrame, alphabet_size: int) -> dict[str, float]:
    required = {"native_log_probability", "native_token", "predicted_token"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction table missing {sorted(missing)}")
    if len(df) == 0:
        return {"n": 0, "nll": float("nan"), "normalized_nll": float("nan"), "perplexity": float("nan"), "recovery": float("nan")}
    nll = float(-df["native_log_probability"].mean())
    return {"n": int(len(df)), "nll": nll, "normalized_nll": nll / math.log(alphabet_size), "perplexity": float(math.exp(min(nll, 50.0))), "recovery": float((df["native_token"] == df["predicted_token"]).mean())}


def macro_by_complex(df: pd.DataFrame, alphabet_size: int) -> pd.DataFrame:
    rows = []
    for sample_id, part in df.groupby("sample_id", sort=True):
        metrics = token_metrics(part, alphabet_size); metrics["sample_id"] = sample_id; rows.append(metrics)
    return pd.DataFrame(rows)


def expected_calibration_error(correct: np.ndarray, confidence: np.ndarray, bins: int = 15) -> float:
    correct = np.asarray(correct, float); confidence = np.asarray(confidence, float)
    if correct.shape != confidence.shape: raise ValueError("correct/confidence shape mismatch")
    edges = np.linspace(0, 1, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < (hi if hi < 1 else hi + 1e-12))
        if mask.any(): ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_multiclass(native_probability: np.ndarray, max_probability: np.ndarray, correct: np.ndarray) -> float:
    correct = np.asarray(correct, float); max_probability = np.asarray(max_probability, float)
    return float(np.mean((max_probability - correct) ** 2))


def native_probability_brier(native_probability: np.ndarray) -> float:
    p = np.asarray(native_probability, float); return float(np.mean((1.0 - p) ** 2))


def paired_bootstrap(a: pd.Series, b: pd.Series, resamples: int = 10000, seed: int = 20260905, statistic: Callable[[np.ndarray], float] = np.mean) -> dict[str, float]:
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna(); diff = (aligned["a"] - aligned["b"]).to_numpy(float)
    if len(diff) < 2: raise ValueError("Need >=2 paired targets")
    rng = np.random.default_rng(seed); n = len(diff); boot = np.empty(resamples)
    for i in range(resamples): boot[i] = statistic(diff[rng.integers(0, n, size=n)])
    return {"n": n, "effect": float(statistic(diff)), "ci_low": float(np.quantile(boot, .025)), "ci_high": float(np.quantile(boot, .975)), "bootstrap_p_two_sided": float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))}


def paired_wilcoxon(a: pd.Series, b: pd.Series) -> dict[str, float]:
    aligned = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna(); result = stats.wilcoxon(aligned["a"], aligned["b"], alternative="two-sided", zero_method="wilcox")
    return {"n": int(len(aligned)), "statistic": float(result.statistic), "p": float(result.pvalue)}


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1]); m = len(items); adjusted = {}; running = 0.0
    for rank, (name, p) in enumerate(items):
        value = min(1.0, (m - rank) * p); running = max(running, value); adjusted[name] = running
    return adjusted


def bh_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1], reverse=True); m = len(items); adjusted = {}; running = 1.0
    for reverse_rank, (name, p) in enumerate(items, start=1):
        rank = m - reverse_rank + 1; value = min(1.0, p * m / rank); running = min(running, value); adjusted[name] = running
    return adjusted


def partner_scramble_delta(native_nll: pd.Series, scrambled_nll: pd.Series) -> pd.Series:
    aligned = pd.concat([native_nll.rename("native"), scrambled_nll.rename("scrambled")], axis=1).dropna(); return aligned["scrambled"] - aligned["native"]


def empirical_pmi(counts_20x4: np.ndarray, pseudocount: float = .5) -> np.ndarray:
    counts = np.asarray(counts_20x4, float)
    if counts.shape != (20, 4): raise ValueError("Expected 20x4 counts")
    counts = counts + pseudocount; p = counts / counts.sum(); pa = p.sum(1, keepdims=True); pb = p.sum(0, keepdims=True); return np.log(p / (pa * pb))


def matrix_correlations(C: np.ndarray, pmi: np.ndarray) -> dict[str, float]:
    c = np.asarray(C, float).ravel(); p = np.asarray(pmi, float).ravel(); pearson = stats.pearsonr(c, p); spearman = stats.spearmanr(c, p)
    return {"pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue), "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue)}


def effective_neighbor_number(alpha: np.ndarray) -> float:
    a = np.asarray(alpha, float); a = a / a.sum(); entropy = -(a * np.log(np.clip(a, 1e-12, 1))).sum(); return float(np.exp(entropy))


def mandatory_test_registry() -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in MANDATORY_TESTS])
