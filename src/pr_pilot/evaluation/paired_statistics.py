"""Strict pairing and sign-flip inference on independent predeclared units."""
from __future__ import annotations

from itertools import product
import math
import numpy as np
import pandas as pd


def strict_align_pairs(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Never silently delete failed targets or treat duplicate rows as replicates."""
    if not isinstance(a, pd.Series) or not isinstance(b, pd.Series):
        raise TypeError("a and b must be indexed Series")
    if not a.index.is_unique or not b.index.is_unique or a.index.hasnans or b.index.hasnans:
        raise ValueError("Pairing indices must be unique and nonmissing")
    if len(a) != len(b) or not a.index.difference(b.index).empty or not b.index.difference(a.index).empty:
        raise ValueError("Paired target sets differ; define an intersection manifest BEFORE evaluation")
    aligned = pd.DataFrame({"a": a, "b": b.reindex(a.index)}, index=a.index)
    if not np.isfinite(aligned.to_numpy(float)).all():
        raise ValueError("Missing/nonfinite paired score; do not drop failed targets")
    return aligned


def signflip_test(differences, *, alternative="less", resamples=10000,
                  seed=20260905, statistic=np.mean) -> dict:
    """Paired sign-flip test; MC p=(b+1)/(B+1), exact enumeration for n<=16.

    Assumption: independent analysis units with sign-exchangeable differences
    under the null. This is not evidence of biological causality. Cluster/seed
    aggregation must take place BEFORE this call. Bounds memory by chunking.
    """
    d = np.asarray(differences, float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Need >=2 finite independent-unit differences")
    if alternative not in {"less", "greater", "two-sided"}:
        raise ValueError("Invalid alternative")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    observed = float(statistic(d))
    if not math.isfinite(observed):
        raise ValueError("Statistic is nonfinite")
    tol = 1e-12 * max(1., abs(observed))

    def is_extreme(value):
        if alternative == "less":
            return value <= observed + tol
        if alternative == "greater":
            return value >= observed - tol
        return abs(value) >= abs(observed) - tol

    exact = len(d) <= 16
    if exact:
        count = sum(is_extreme(float(statistic(d * np.asarray(signs))))
                    for signs in product((-1., 1.), repeat=len(d)))
        total = 2 ** len(d)
        p = count / total
    else:
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(resamples):
            count += is_extreme(float(statistic(d * rng.choice([-1., 1.], size=len(d)))))
        total = resamples
        p = (count + 1) / (total + 1)
    return {"p": float(p), "effect": observed, "n_units": len(d),
            "alternative": alternative, "method": "exact_signflip" if exact else "monte_carlo_signflip_plus_one",
            "null_draws": total}


def mean_bootstrap_ci(differences, *, resamples=10000, seed=20260905) -> tuple[float, float]:
    d = np.asarray(differences, float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("Need >=2 finite independent units")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples)
    for start in range(0, resamples, 512):
        end = min(start + 512, resamples)
        boot[start:end] = d[rng.integers(0, len(d), size=(end-start, len(d)))].mean(1)
    return float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
