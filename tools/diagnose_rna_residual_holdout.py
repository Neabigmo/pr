#!/usr/bin/env python3
"""Post-test, per-complex diagnostics for the RNA residual-cap pilot.

This script is descriptive only.  It reads the already frozen holdout after
CV and checkpoint selection, never selects a checkpoint, and does not train.
It reports ratio-of-means, mean per-complex ratios, median prior-minus-adapter
gain, adapter-better fraction, and RNA prior-NLL quartiles.
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.model import ReciprocalAdapter
from run_adapter_v2_cv import _attach_selected_edges, _forward_payload, _load_cache
from run_rna_residual_cap import build_spec
from run_scientific_adapter_pilot import _config


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _per_complex(model, payload, device: torch.device) -> dict[str, float]:
    with torch.no_grad():
        native = _forward_payload(model, payload, device, 14.979730606, 32, token_off=False)
        protein_tokens = payload["protein_native"].roll(1) if len(payload["protein_native"]) > 1 else payload["protein_native"]
        rna_tokens = payload["rna_native"].roll(1) if len(payload["rna_native"]) > 1 else payload["rna_native"]
        shuffled = _forward_payload(
            model,
            payload,
            device,
            14.979730606,
            32,
            partner_token_override={"protein": protein_tokens, "rna": rna_tokens},
        )
        labels = payload["rna_native"].to(device)
        interface = payload["rna_interface"].to(device)
        base = payload["rna_base"].to(device)
        index = torch.arange(len(labels), device=device)
        adapter_nll = -F.log_softmax(native["rna_logits"], dim=-1)[index, labels]
        prior_nll = -base[index, labels]
        shuffle_nll = -F.log_softmax(shuffled["rna_logits"], dim=-1)[index, labels]
        if not bool(interface.any()):
            return {"sample_id": str(payload["sample_id"]), "has_interface": 0.0}
        mask = interface
        adapter = float(adapter_nll[mask].mean().cpu())
        prior = float(prior_nll[mask].mean().cpu())
        shuffle = float(shuffle_nll[mask].mean().cpu())
        return {
            "sample_id": str(payload["sample_id"]),
            "has_interface": 1.0,
            "adapter_interface_nll": adapter,
            "prior_interface_nll": prior,
            "shuffle_interface_nll": shuffle,
            "ratio": adapter / max(prior, 1e-8),
            "gain_prior_minus_adapter": prior - adapter,
            "native_minus_shuffle": adapter - shuffle,
        }


def _summarize(rows: list[dict[str, float]]) -> dict:
    rows = [row for row in rows if row.get("has_interface", 0.0) > 0.0]
    prior = np.asarray([row["prior_interface_nll"] for row in rows], dtype=float)
    adapter = np.asarray([row["adapter_interface_nll"] for row in rows], dtype=float)
    shuffle = np.asarray([row["shuffle_interface_nll"] for row in rows], dtype=float)
    gain = prior - adapter
    ratio = adapter / np.maximum(prior, 1e-8)
    result = {
        "n_complexes": int(len(rows)),
        "ratio_of_means": float(adapter.mean() / max(prior.mean(), 1e-8)),
        "mean_per_complex_ratio": float(ratio.mean()),
        "median_prior_minus_adapter": float(np.median(gain)),
        "adapter_better_fraction": float(np.mean(gain > 0.0)),
        "native_minus_shuffle_mean": float(np.mean(adapter - shuffle)),
        "native_better_than_shuffle_fraction": float(np.mean(adapter < shuffle)),
        "prior_interface_nll_mean": float(prior.mean()),
        "adapter_interface_nll_mean": float(adapter.mean()),
        "shuffle_interface_nll_mean": float(shuffle.mean()),
        "quartiles": {},
    }
    edges = np.quantile(prior, [0.0, 0.25, 0.5, 0.75, 1.0])
    # Stable index quartiles avoid boundary ambiguity when several complexes
    # have identical prior NLLs.
    order = np.argsort(prior, kind="stable")
    for q, indices in enumerate(np.array_split(order, 4), start=1):
        q_prior = prior[indices]
        q_adapter = adapter[indices]
        q_gain = q_prior - q_adapter
        result["quartiles"][f"Q{q}"] = {
            "n": int(len(indices)),
            "prior_nll_mean": float(q_prior.mean()),
            "adapter_nll_mean": float(q_adapter.mean()),
            "ratio_of_means": float(q_adapter.mean() / max(q_prior.mean(), 1e-8)),
            "mean_per_complex_ratio": float(np.mean(q_adapter / np.maximum(q_prior, 1e-8))),
            "median_prior_minus_adapter": float(np.median(q_gain)),
            "adapter_better_fraction": float(np.mean(q_gain > 0.0)),
            "prior_nll_range": [float(q_prior.min()), float(q_prior.max())],
        }
    result["quartile_edges"] = [float(x) for x in edges]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.out
    data = _load_cache(args.test_cache, "test")
    _attach_selected_edges(data, 14.979730606, 32, 8, 12, True)
    groups = {}
    for group_dir in sorted((root / "cv").iterdir()):
        if not group_dir.is_dir():
            continue
        fold_rows: dict[str, list[dict[str, float]]] = {}
        for fold in range(3):
            checkpoint = group_dir / f"fold{fold}" / "best.pt"
            payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
            mode = {
                "M0_current_scalar_gate": "baseline",
                "M1_rna_fixed_gate_0p25": "fixed_quarter",
                "M2_rna_capped_gate_0p25": "capped_quarter",
            }[group_dir.name]
            model = ReciprocalAdapter(_config(build_spec(mode))).to(args.device)
            model.load_state_dict(payload["model"])
            model.eval()
            rows = [_per_complex(model, item, torch.device(args.device)) for item in data]
            for row in rows:
                fold_rows.setdefault(row["sample_id"], []).append(row)
            del model, payload
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
        averaged = []
        for sample_id, items in sorted(fold_rows.items()):
            usable = [item for item in items if item.get("has_interface", 0.0) > 0.0]
            if not usable:
                averaged.append({"sample_id": sample_id, "has_interface": 0.0})
                continue
            keys = ("adapter_interface_nll", "prior_interface_nll", "shuffle_interface_nll")
            row = {"sample_id": sample_id, "has_interface": 1.0}
            for key in keys:
                row[key] = _mean([float(item[key]) for item in usable])
            row["ratio"] = row["adapter_interface_nll"] / max(row["prior_interface_nll"], 1e-8)
            row["gain_prior_minus_adapter"] = row["prior_interface_nll"] - row["adapter_interface_nll"]
            row["native_minus_shuffle"] = row["adapter_interface_nll"] - row["shuffle_interface_nll"]
            averaged.append(row)
        groups[group_dir.name] = {"summary": _summarize(averaged), "per_complex": averaged}
    result = {
        "protocol": {"test_read_only": True, "holdout_used_for_selection": False, "fold_checkpoints": "best.pt"},
        "groups": groups,
    }
    destination = root / "holdout_per_complex_diagnostics.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({name: value["summary"] for name, value in groups.items()}, indent=2, ensure_ascii=False))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
