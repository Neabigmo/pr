"""Cost-audited wrapper around the frozen final-100 mechanistic battery.

The core conditional/joint metrics are already evaluated for every primary seed by
the orchestrator. The heavyweight order/SPIR candidate battery is executed only on
the predeclared analysis seed and uses ``evaluation.ablation_candidates_per_complex``
rather than silently multiplying the 64-candidate primary design budget across all
ablation cells.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import copy
import json

import yaml

from pr_pilot.evaluation import full_suite_legacy as _legacy


def run_full_suite(
    cfg: dict,
    checkpoint: Path,
    test_manifest: Path,
    out_dir: Path,
    dev_manifest: Path | None = None,
    device: str | None = None,
) -> dict:
    local = copy.deepcopy(cfg)
    primary_budget = int(local["inference"]["candidates_per_complex"])
    ablation_budget = int(
        local.get("evaluation", {}).get("ablation_candidates_per_complex", primary_budget)
    )
    if ablation_budget <= 0 or ablation_budget > primary_budget:
        raise ValueError(
            "ablation_candidates_per_complex must be positive and no larger than the primary budget"
        )
    local["inference"]["candidates_per_complex"] = ablation_budget
    summary = _legacy.run_full_suite(
        local, checkpoint, test_manifest, out_dir, dev_manifest, device
    )
    budget = {
        "primary_design_candidates_per_complex": primary_budget,
        "mechanistic_ablation_candidates_per_cell": ablation_budget,
        "scope": "heavy full suite is intended for the predeclared analysis seed only",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_budget.json").write_text(
        json.dumps(budget, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary["evaluation_budget"] = budget
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--dev", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(
        json.dumps(
            run_full_suite(
                cfg,
                args.checkpoint,
                args.test,
                args.out,
                args.dev,
                args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
