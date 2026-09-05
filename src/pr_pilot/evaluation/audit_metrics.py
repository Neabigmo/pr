"""Strict metrics and descriptive field diagnostics for the Review-4 protocol.

These routines are additive. Existing evaluation entrypoints must explicitly call
these functions or export the inputs for tools/run_review4_statistics.py. Merely
importing this module does not upgrade historical result files.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import logsumexp


def multiclass_brier_score(probabilities, targets, *, reduction: str = "mean"):
    """Mean sum_k (p_k - 1[k=y])**2; no division by alphabet size.

    Range [0, 2]. Uniform K-class prediction scores 1 - 1/K. This is NOT
    (max_probability - correctness)**2, which is a top-label binary score.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("probabilities must be a nonempty [N,K>=2] matrix")
    if y.shape != (p.shape[0],) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("targets must be integer [N]")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.allclose(p.sum(1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Probability rows must sum to one; do not silently renormalize")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Target outside alphabet")
    scores = np.sum(p * p, axis=1) - 2.0 * p[np.arange(len(y)), y] + 1.0
    # Floating-point roundoff only; all semantic checks happened above.
    scores = np.clip(scores, 0.0, 2.0)
    if reduction == "none":
        return scores
    if reduction != "mean":
        raise ValueError("reduction must be mean or none")
    return float(scores.mean())


def top_label_brier_score(confidence, correct) -> float:
    """Binary Brier of confidence in the chosen label, explicitly named."""
    p, y = np.asarray(confidence, float), np.asarray(correct)
    if p.ndim != 1 or p.size == 0 or y.shape != p.shape:
        raise ValueError("Need aligned nonempty vectors")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("confidence must be in [0,1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("correct must be binary")
    return float(np.mean((p - y) ** 2))


def order_mixture_log_probability(sequence_log_probabilities, weights=None) -> float:
    """log sum_o w_o p_o(sequence), not mean_o log p_o(sequence).

    Inputs are UNNORMALIZED sequence log probabilities (sum over tokens), not
    per-token NLL, entropy-normalized loss, or post-SPIR pseudo-likelihood.
    This is the exact mixture over the supplied finite set of orders, not an
    exact integral over every possible permutation.
    """
    lp = np.asarray(sequence_log_probabilities, float)
    if lp.ndim != 1 or lp.size == 0 or np.isnan(lp).any() or (lp > 1e-8).any():
        raise ValueError("Need a nonempty vector of nonpositive log probabilities")
    if weights is None:
        log_w = np.full(lp.shape, -math.log(lp.size))
    else:
        w = np.asarray(weights, float)
        if w.shape != lp.shape or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("Invalid mixture weights")
        if not np.isclose(w.sum(), 1.0, atol=1e-8, rtol=0):
            raise ValueError("Mixture weights must sum to one")
        log_w = np.full(w.shape, -np.inf)
        np.log(w, out=log_w, where=w > 0)
    return float(logsumexp(lp + log_w))


def compatibility_components(matrix) -> dict[str, np.ndarray | float]:
    """Descriptive ANOVA decomposition under UNIFORM alphabet weighting.

    matrix = grand + row_main + column_main + interaction. Double centering
    extracts only interaction; it is not a forward-path gauge transformation
    when frozen priors cannot absorb the row/column effects.
    """
    x = np.asarray(matrix, float)
    if x.shape != (20, 4) or not np.isfinite(x).all():
        raise ValueError("Expected finite 20x4 matrix")
    grand = float(x.mean())
    row = x.mean(1, keepdims=True) - grand
    col = x.mean(0, keepdims=True) - grand
    interaction = x - grand - row - col
    return {"grand": grand, "row_main": row, "column_main": col,
            "interaction": interaction}


def residual_drift_audit(global_c, delta_c, edge_weights=None) -> dict:
    """Descriptive edge-weighted moments, never a residue-level significance test.

    A norm-zero C/residual produces None for an undefined ratio, not a fake zero.
    For population claims construct equal-pair/family weights before calling.
    """
    c, dc = np.asarray(global_c, float), np.asarray(delta_c, float)
    if c.shape != (20, 4) or dc.ndim != 3 or dc.shape[1:] != (20, 4) or len(dc) == 0:
        raise ValueError("Expected C[20,4] and nonempty DeltaC[E,20,4]")
    if not np.isfinite(c).all() or not np.isfinite(dc).all():
        raise ValueError("Nonfinite compatibility values")
    w = np.ones(len(dc), dtype=float) if edge_weights is None else np.asarray(edge_weights, float)
    if w.shape != (len(dc),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative and have positive mass")
    w = w / w.sum()
    mean_dc = np.einsum("e,eab->ab", w, dc)
    rms_dc = float(np.sqrt(np.dot(w, (dc * dc).sum((1, 2)))))
    c_norm = float(np.linalg.norm(c))
    return {
        "n_edges": len(dc), "c_frobenius": c_norm,
        "delta_rms_frobenius": rms_dc,
        "mean_delta_frobenius": float(np.linalg.norm(mean_dc)),
        "delta_rms_to_c": rms_dc / c_norm if c_norm > 0 else None,
        "mean_delta_to_delta_rms": float(np.linalg.norm(mean_dc)) / rms_dc if rms_dc > 0 else None,
        "weighting": "uniform_edges" if edge_weights is None else "caller_supplied",
        "mean_delta": mean_dc.tolist(),
    }


def directional_coefficient_gap(alpha_p, alpha_r, lambda_p=1.0, lambda_r=1.0) -> dict:
    """A necessary pairwise-field reciprocity diagnostic, NOT a joint-model proof.

    Both arrays refer to the same edges but different destination normalizations.
    A nonzero gap disproves a shared symmetric pairwise coefficient in general.
    A zero gap does not prove decoder-conditionals are globally compatible.
    """
    ap, ar = np.asarray(alpha_p, float), np.asarray(alpha_r, float)
    if ap.ndim != 1 or ap.shape != ar.shape or ap.size == 0:
        raise ValueError("Expected aligned nonempty edge vectors")
    if not np.isfinite(ap).all() or not np.isfinite(ar).all() or (ap < 0).any() or (ar < 0).any():
        raise ValueError("Invalid relevance values")
    if not math.isfinite(lambda_p) or not math.isfinite(lambda_r) or min(lambda_p, lambda_r) <= 0:
        raise ValueError("Gains must be finite and positive")
    gap = lambda_p * ap - lambda_r * ar
    return {"mean_abs_gap": float(np.abs(gap).mean()),
            "max_abs_gap": float(np.abs(gap).max()),
            "interpretation": "necessary_pairwise_reciprocity_diagnostic_only"}
