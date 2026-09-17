"""Five-hypothesis analysis with an immutable roster and explicit H2 conjunction.

Input effects are prepared by an upstream evaluator; this module does not claim
that their scientific contrasts or data provenance have been independently
validated. It rejects incomplete/duplicated exports and separates seeds from
independent biological units. See docs/EXPERIMENT_VALIDATION_V2.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pr_pilot.evaluation.battery import holm_adjust
from pr_pilot.evaluation.paired_statistics import signflip_test, mean_bootstrap_ci

HYPOTHESES = {
    "full_vs_dual_prior_interface_nll": ["primary"],
    "full_vs_partner_identity_controls": ["partner_blind", "geometry_only"],
    "contextual_vs_global_c": ["primary"],
    "partner_scramble": ["primary"],
    "alpha_edge_removal": ["primary"],
}


def analyze_confirmatory(effects: pd.DataFrame, roster: pd.DataFrame, seeds: list[int],
                         *, resamples: int = 10000, random_seed: int = 20260905,
                         alpha: float = 0.05) -> dict:
    """All effects use negative = improvement under the prespecified contrast.

    Aggregation: seed mean per complex -> complex mean within independence group
    -> equal-weight mean across groups. This estimates the average GROUP effect,
    not the pooled residue or uniformly weighted complex effect.
    H2 requires both controls beaten: max(component p) is an intersection-union
    p value, followed by Holm across the five hypotheses. Confidence intervals
    are descriptive marginal intervals, not simultaneous Holm intervals.
    """
    required = {"hypothesis", "component", "sample_id", "group_id", "seed", "effect"}
    if not required.issubset(effects.columns):
        raise ValueError(f"Missing effect columns: {sorted(required - set(effects.columns))}")
    if not {"sample_id", "group_id"}.issubset(roster.columns) or roster.empty:
        raise ValueError("A frozen sample_id/group_id roster is required")
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
        raise ValueError("Seeds must be a nonempty unique integer list")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    if roster[["sample_id", "group_id"]].isna().any().any() or not roster.sample_id.is_unique:
        raise ValueError("Roster IDs must be complete and sample IDs unique")
    if effects[list(required)].isna().any().any() or not np.isfinite(effects.effect.to_numpy(float)).all():
        raise ValueError("Missing/nonfinite effects; failed targets cannot be silently excluded")
    if set(effects.hypothesis) != set(HYPOTHESES):
        raise ValueError("Exactly the five prespecified hypotheses must be supplied")
    mapping = roster.set_index("sample_id").group_id.to_dict()
    sample_ids = set(mapping)
    expected_pairs = {(sid, seed) for sid in sample_ids for seed in seeds}
    result, pvalues = {}, {}
    for h_index, (hypothesis, components) in enumerate(HYPOTHESES.items()):
        part = effects[effects.hypothesis == hypothesis]
        if set(part.component) != set(components):
            raise ValueError(f"{hypothesis}: incorrect component set")
        component_results = {}
        for c_index, component in enumerate(components):
            sub = part[part.component == component]
            if sub.duplicated(["sample_id", "seed"]).any():
                raise ValueError(f"{hypothesis}/{component}: duplicate sample/seed rows")
            actual_pairs = set(zip(sub.sample_id, sub.seed))
            if actual_pairs != expected_pairs:
                raise ValueError(f"{hypothesis}/{component}: roster/seed coverage mismatch; "
                                 f"missing={len(expected_pairs-actual_pairs)}, extra={len(actual_pairs-expected_pairs)}")
            if any(mapping[sid] != gid for sid, gid in zip(sub.sample_id, sub.group_id)):
                raise ValueError("Independence-group assignment differs from frozen roster")
            per_sample = sub.groupby("sample_id", sort=True).effect.mean()
            grouped = pd.DataFrame({"effect": per_sample,
                                    "group_id": [mapping[sid] for sid in per_sample.index]})
            unit_effects = grouped.groupby("group_id", sort=True).effect.mean()
            seed_here = random_seed + h_index * 1009 + c_index * 101
            test = signflip_test(unit_effects.to_numpy(), alternative="less",
                                 resamples=resamples, seed=seed_here)
            lo, hi = mean_bootstrap_ci(unit_effects.to_numpy(), resamples=resamples, seed=seed_here+1)
            component_results[component] = {
                **test, "ci_low": lo, "ci_high": hi,
                "n_complexes": len(per_sample), "n_training_seeds": len(seeds),
                "per_seed_mean_complex_effect": {str(k): float(v) for k, v in sub.groupby("seed").effect.mean().items()},
                "unit_effects": {str(k): float(v) for k, v in unit_effects.items()},
            }
        pvalues[hypothesis] = max(entry["p"] for entry in component_results.values())
        result[hypothesis] = {"components": component_results, "p_unadjusted": pvalues[hypothesis],
                              "combination": "intersection_union_max_p" if len(components)>1 else "single_component"}
    adjusted = holm_adjust(pvalues)
    for h, value in result.items():
        value["p_holm"] = adjusted[h]
        value["reject_in_prespecified_direction"] = bool(
            adjusted[h] <= alpha and all(c["effect"] < 0 for c in value["components"].values())
        )
    return {
        "schema_version": "review4.v1", "analysis_status": "numeric_analysis_only_provenance_not_certified",
        "effect_direction": "negative_is_improvement", "estimand": "equal_independence_group_mean",
        "seed_handling": "average_within_sample_not_independent_replication",
        "null_assumption": "independent_groups_with_sign_exchangeable_differences",
        "ci_interpretation": "marginal_percentile_95_not_simultaneous",
        "multiplicity": "Holm_across_five_after_H2_intersection_union",
        "alpha": alpha, "hypotheses": result,
    }
