#!/usr/bin/env python3
"""Audit pilot YAML leaves against executable code or explicit assertions.

A research config must not contain switches that look tunable but are silently
ignored. Every leaf is either referenced by executable Python or deliberately
classified as a protocol/assertion value. Unknown leaves fail the audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


DECLARATIVE_PREFIXES = {
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
    "geometry.prohibit_rna_base_identity_atoms",
    "training_stages.global_c.conditional_task_ratio",
    "training_stages.global_c.lambda_fixed_to_one",
    "training_stages.delta_c.zero_init_output",
    "training_stages.delta_c.lambda_fixed_to_one",
    "training_stages.alpha.zero_init_score_residual",
    "training_stages.alpha.lambda_fixed_to_one",
    "training_stages.alpha.freeze_interaction_and_delta",
    "training_stages.joint.learn_bounded_lambda",
    "training_stages.joint.freeze_global_c",
    "fairness_controls",
    "evaluation.statistical_unit",
    "evaluation.primary_multiple_testing",
    "evaluation.exploratory_multiple_testing",
    "evaluation.primary_hypotheses",
    "evaluation.runtime_profile_complexes",
}


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaves(child, path)
    else:
        yield prefix, value


def _declared(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in DECLARATIVE_PREFIXES)


def audit(config: Path, repo_root: Path) -> dict:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    python_files = list((repo_root / "src").rglob("*.py")) + list((repo_root / "tools").rglob("*.py"))
    corpus = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in python_files)
    rows = []
    unknown = []
    for path, value in _leaves(cfg):
        key = path.rsplit(".", 1)[-1]
        if _declared(path):
            status = "declarative_assertion"
        elif f'"{key}"' in corpus or f"'{key}'" in corpus:
            status = "runtime_referenced"
        else:
            status = "UNKNOWN_OR_DEAD"
            unknown.append(path)
        rows.append({"path": path, "value": value, "status": status})
    return {"config": str(config), "files_scanned": len(python_files), "leaves": rows, "unknown": unknown}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.config.resolve(), args.repo_root.resolve())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if report["unknown"]:
        raise SystemExit("Unknown/dead config leaves: " + ", ".join(report["unknown"]))
    print(json.dumps({"status": "PASS", "leaves": len(report["leaves"]), "unknown": 0}, indent=2))


if __name__ == "__main__":
    main()
