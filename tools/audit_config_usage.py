#!/usr/bin/env python3
"""Audit pilot.yaml leaves as runtime-consumed or explicitly declarative.

Research configs should not accumulate knobs that look active but have no semantic
contract. Runtime discovery is intentionally conservative (exact leaf-key string
search in project Python); protocol assertions that are hard-coded by design are
listed explicitly below. Anything else fails CI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import yaml


DECLARATIVE_PREFIXES = (
    "fairness_controls.",
    "evaluation.run_",
)
DECLARATIVE_EXACT = {
    "experiment.output_root",
    "experiment.fail_on_test_leakage",
    "sampling.final_refit_on_full_dev",
    "sampling.final_refit_rule",
    "sampling.forbid_predicted_structures",
    "structure_filters.require_biological_assembly",
    "leakage.protein_strict_cluster",
    "leakage.rna_sequence_cluster",
    "leakage.require_rfam_holdout_when_available",
    "leakage.purge_final_test_from_single_molecule_pretraining",
    "leakage.prior_validation_cluster_disjoint",
    "geometry.prohibit_rna_base_identity_atoms",
    "loss.global_c_label_smoothing",
    "loss.delta_c_norm_penalty",
    "optimization.optimizer",
    "optimization.schedule",
    "training_stages.global_c.lambda_fixed_to_one",
    "training_stages.delta_c.zero_init_output",
    "training_stages.delta_c.lambda_fixed_to_one",
    "training_stages.alpha.zero_init_score_residual",
    "training_stages.alpha.lambda_fixed_to_one",
    "training_stages.alpha.freeze_context_field",
    "training_stages.joint.learn_bounded_lambda",
    "training_stages.joint.freeze_global_c",
    "inference.random_mixed_order",
    "evaluation.statistical_unit",
    "evaluation.joint_teacher_forced_orders",
    "evaluation.primary_multiple_testing",
    "evaluation.exploratory_multiple_testing",
}


def _flatten(obj, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, path))
    else:
        out[prefix] = obj
    return out


def _source_text(repo_root: Path) -> str:
    files = list((repo_root / "src").rglob("*.py")) + list((repo_root / "tools").rglob("*.py"))
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in files
        if path.name != Path(__file__).name
    )


def classify(config: dict, repo_root: Path) -> dict[str, dict]:
    source = _source_text(repo_root)
    result: dict[str, dict] = {}
    for path, value in _flatten(config).items():
        leaf = path.rsplit(".", 1)[-1]
        pattern = re.compile(rf"[\"']{re.escape(leaf)}[\"']")
        if pattern.search(source):
            status = "runtime_consumed"
        elif path in DECLARATIVE_EXACT or any(
            path.startswith(prefix) for prefix in DECLARATIVE_PREFIXES
        ):
            status = "declarative_assertion"
        else:
            status = "unknown_or_dead"
        result[path] = {"value": value, "status": status}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = classify(config, args.repo_root.resolve())
    unknown = [path for path, item in result.items() if item["status"] == "unknown_or_dead"]
    summary = {
        "runtime_consumed": sum(
            item["status"] == "runtime_consumed" for item in result.values()
        ),
        "declarative_assertion": sum(
            item["status"] == "declarative_assertion" for item in result.values()
        ),
        "unknown_or_dead": unknown,
        "keys": result,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if unknown:
        raise SystemExit("Config audit failed: unknown/dead keys remain")


if __name__ == "__main__":
    main()
