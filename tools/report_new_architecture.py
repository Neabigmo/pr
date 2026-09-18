#!/usr/bin/env python3
"""Evaluate the locked scientific Adapter architecture and summarize ablations.

This tool is deliberately separate from the legacy conditional evaluator.  The
scientific checkpoints store their architecture in ``metadata.spec`` rather
than the legacy ``config`` field, and the evaluator must preserve that
distinction.  It reports six teacher-forced conditions:
Protein/RNA x all/active/interface.

The holdout command is intended to be run only after the development search
has been locked.  It never trains either frozen prior and never selects a
model from holdout metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from run_conditional_adapter_pilot import _attach_selected_edges, _forward_payload, _load_cache
from run_scientific_adapter_pilot import _config
from pr_pilot.adapter_pilot.model import ReciprocalAdapter


CONDITIONS = (("protein", "all"), ("protein", "active"), ("protein", "interface"),
              ("rna", "all"), ("rna", "active"), ("rna", "interface"))


def _score_rows(model: ReciprocalAdapter, data: list[dict], device: torch.device) -> list[dict]:
    rows: list[dict] = []
    with torch.no_grad():
        for payload in data:
            out = _forward_payload(model, payload, device, 0.0, 0, token_off=False)
            for polymer in ("protein", "rna"):
                labels = payload[f"{polymer}_native"].to(device)
                base = payload[f"{polymer}_base"].to(device)
                logits = out[f"{polymer}_logits"]
                adapter_logp = F.log_softmax(logits, dim=-1)
                adapter_nll = -adapter_logp[torch.arange(len(labels), device=device), labels]
                prior_nll = -base[torch.arange(len(labels), device=device), labels]
                adapter_hit = (adapter_logp.argmax(-1) == labels).float()
                masks = {
                    "all": torch.ones_like(labels, dtype=torch.bool),
                    "active": payload[f"_selected_{polymer}_active"].to(device),
                    "interface": payload[f"{polymer}_interface"].to(device),
                }
                for subset, mask in masks.items():
                    if not bool(mask.any()):
                        continue
                    rows.append({
                        "sample_id": str(payload["sample_id"]),
                        "polymer": polymer,
                        "subset": subset,
                        "adapter_nll": float(adapter_nll[mask].mean().cpu()),
                        "prior_nll": float(prior_nll[mask].mean().cpu()),
                        "adapter_recovery": float(adapter_hit[mask].mean().cpu()),
                        "prior_recovery": float((base.argmax(-1)[mask] == labels[mask]).float().mean().cpu()),
                    })
    return rows


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def _bootstrap(rows: list[dict], seed: int, repeats: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    delta_nll = np.asarray([row["prior_nll"] - row["adapter_nll"] for row in rows], dtype=np.float64)
    delta_recovery = np.asarray([row["adapter_recovery"] - row["prior_recovery"] for row in rows], dtype=np.float64)
    if len(delta_nll) == 0:
        return {"n": 0}
    indices = rng.integers(0, len(delta_nll), size=(repeats, len(delta_nll)))
    nll_means = delta_nll[indices].mean(axis=1)
    recovery_means = delta_recovery[indices].mean(axis=1)
    return {
        "n": int(len(rows)),
        "delta_nll_prior_minus_adapter": float(delta_nll.mean()),
        "delta_nll_ci95": [float(x) for x in np.quantile(nll_means, [0.025, 0.975])],
        "delta_recovery_adapter_minus_prior": float(delta_recovery.mean()),
        "delta_recovery_ci95": [float(x) for x in np.quantile(recovery_means, [0.025, 0.975])],
    }


def _condition_summary(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(f"{row['polymer']}_{row['subset']}", []).append(row)
    result: dict[str, dict] = {}
    for polymer, subset in CONDITIONS:
        key = f"{polymer}_{subset}"
        values = grouped.get(key, [])
        adapter = _mean(values, "adapter_nll")
        prior = _mean(values, "prior_nll")
        result[key] = {
            "n_complexes": len(values),
            "adapter_nll": adapter,
            "prior_nll": prior,
            "ratio_adapter_over_prior": float(adapter / max(prior, 1e-12)),
            "delta_nll_prior_minus_adapter": float(prior - adapter),
            "adapter_recovery": _mean(values, "adapter_recovery"),
            "prior_recovery": _mean(values, "prior_recovery"),
            "delta_recovery_adapter_minus_prior": float(_mean(values, "adapter_recovery") - _mean(values, "prior_recovery")),
            "paired_bootstrap_10000": _bootstrap(values, 20260917 + len(key)),
        }
    return result


def run_holdout(args: argparse.Namespace) -> dict:
    search_summary = json.loads(Path(args.search_summary).read_text(encoding="utf-8"))
    expected_spec = search_summary["selected"]
    data = _load_cache(Path(args.test_cache), "test")
    if len(data) != 86:
        raise ValueError(f"expected frozen 86-complex holdout, found {len(data)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fold_results = []
    all_rows: list[dict] = []
    for fold in range(3):
        checkpoint_path = Path(args.checkpoint_root) / f"fold{fold}" / "best.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        spec = payload["metadata"]["spec"]
        if spec != expected_spec:
            raise ValueError(f"fold{fold} spec does not match locked search spec")
        model = ReciprocalAdapter(_config(spec)).to(args.device)
        model.load_state_dict(payload["model"])
        model.eval()
        fold_data = [dict(item) for item in data]
        _attach_selected_edges(fold_data, float(spec["radius"]), int(args.neighbors), int(spec["r2p_k"]), int(spec["p2r_k"]), True)
        rows = _score_rows(model, fold_data, args.device)
        (out / f"fold{fold}_per_complex.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        fold_results.append({"fold": fold, "checkpoint": str(checkpoint_path), "spec": spec, "metrics": _condition_summary(rows)})
        all_rows.extend(rows)
    aggregate = {}
    for polymer, subset in CONDITIONS:
        key = f"{polymer}_{subset}"
        values = [fold["metrics"][key] for fold in fold_results]
        aggregate[key] = {field: float(np.mean([float(item[field]) for item in values])) for field in (
            "adapter_nll", "prior_nll", "ratio_adapter_over_prior", "delta_nll_prior_minus_adapter",
            "adapter_recovery", "prior_recovery", "delta_recovery_adapter_minus_prior")}
        aggregate[key]["folds"] = 3
    aggregate_rows: list[dict] = []
    grouped_rows: dict[tuple[str, str, str], list[dict]] = {}
    for row in all_rows:
        grouped_rows.setdefault((row["sample_id"], row["polymer"], row["subset"]), []).append(row)
    for (sample_id, polymer, subset), values in grouped_rows.items():
        aggregate_rows.append({
            "sample_id": sample_id,
            "polymer": polymer,
            "subset": subset,
            "adapter_nll": float(np.mean([item["adapter_nll"] for item in values])),
            "prior_nll": float(np.mean([item["prior_nll"] for item in values])),
            "adapter_recovery": float(np.mean([item["adapter_recovery"] for item in values])),
            "prior_recovery": float(np.mean([item["prior_recovery"] for item in values])),
        })
    aggregate_bootstrap = {}
    for polymer, subset in CONDITIONS:
        key = f"{polymer}_{subset}"
        aggregate_bootstrap[key] = _bootstrap(
            [row for row in aggregate_rows if row["polymer"] == polymer and row["subset"] == subset],
            20260917 + len(key),
            int(args.bootstrap_repeats),
        )
    result = {
        "protocol": {"test_read": True, "test_complexes": 86, "prior_retrained": False, "adapter_retrained": False,
                     "condition_definition": "teacher-forced Protein/RNA x all/active/interface",
                     "checkpoint_selection_source": str(Path(args.search_summary).resolve()),
                     "holdout_not_used_for_selection": True},
        "spec": expected_spec,
        "cache": str(Path(args.test_cache).resolve()),
        "folds": fold_results,
        "aggregate_mean_of_three_folds": aggregate,
        "aggregate_paired_bootstrap": aggregate_bootstrap,
    }
    (out / "holdout_six_condition_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def run_ablation(args: argparse.Namespace) -> dict:
    source = json.loads(Path(args.search_summary).read_text(encoding="utf-8"))
    rows = []
    for stage, stage_data in source["stages"].items():
        for candidate in stage_data["candidates"]:
            rows.append({
                "stage": stage,
                "name": candidate["name"],
                "spec": candidate["spec"],
                "protein_interface_ratio": candidate["metrics"]["protein_interface_ratio"],
                "rna_interface_ratio": candidate["metrics"]["rna_interface_ratio"],
                "worst_direction_ratio": max(candidate["metrics"]["protein_interface_ratio"], candidate["metrics"]["rna_interface_ratio"]),
                "specificity_mean": candidate["metrics"].get("specificity_mean"),
                "is_stage_selected": candidate["name"] == stage_data["selected"]["name"],
                "test_read": False,
            })
    result = {"protocol": {"development_only": True, "test_read": False, "single_seed": source["seed"],
                            "selection_metric": "worst_direction_ratio on development validation"},
              "selected": source["selected"], "rows": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "development_ablation_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    holdout = sub.add_parser("holdout")
    holdout.add_argument("--search-summary", type=Path, required=True)
    holdout.add_argument("--test-cache", type=Path, required=True)
    holdout.add_argument("--checkpoint-root", type=Path, required=True)
    holdout.add_argument("--out", type=Path, required=True)
    holdout.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    holdout.add_argument("--neighbors", type=int, default=32)
    holdout.add_argument("--bootstrap-repeats", type=int, default=10000)
    holdout.set_defaults(func=run_holdout)
    ablation = sub.add_parser("ablation")
    ablation.add_argument("--search-summary", type=Path, required=True)
    ablation.add_argument("--out", type=Path, required=True)
    ablation.set_defaults(func=run_ablation)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
