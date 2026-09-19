#!/usr/bin/env python3
"""One-time P0--P3 evaluation for the locked A2 model on a new blind set."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.explicit_matrix import ExplicitMatrixConfig, ExplicitSelectionAdapter  # noqa: E402
from run_conditional_adapter_pilot import _load_cache  # noqa: E402
from run_explicit_selection_matrix import RADIUS, _attach_selected_edges, _forward, _shuffle_tokens  # noqa: E402


CONDITIONS = ("P1_native", "P2_token_off", "P3_partner_shuffle")
SUBSETS = ("all", "active", "interface")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _mask(payload: dict, polymer: str, subset: str) -> torch.Tensor:
    if subset == "all":
        return torch.ones(len(payload[f"{polymer}_native"]), dtype=torch.bool)
    if subset == "active":
        return payload[f"_selected_{polymer}_active"].bool()
    if subset == "interface":
        return payload[f"{polymer}_interface"].bool()
    raise ValueError(subset)


def _metric(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    if not bool(mask.any()):
        return float("nan"), float("nan")
    nll = -F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)
    recovery = (logits.argmax(dim=-1) == labels).float()
    return float(nll[mask].mean().cpu()), float(recovery[mask].mean().cpu())


def _finite_mean(values: list[float]) -> float:
    values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


def _bootstrap(rows: list[dict], adapter_key: str, prior_key: str, seed: int, repeats: int) -> dict:
    adapter = np.asarray([float(row[adapter_key]) for row in rows], dtype=float)
    prior = np.asarray([float(row[prior_key]) for row in rows], dtype=float)
    valid = np.isfinite(adapter) & np.isfinite(prior) & (prior > 0)
    adapter, prior = adapter[valid], prior[valid]
    if not len(adapter):
        return {"n": 0, "repeats": int(repeats)}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(adapter), size=(int(repeats), len(adapter)))
    ratio = adapter[indices].mean(axis=1) / np.maximum(prior[indices].mean(axis=1), 1e-12)
    delta = prior[indices].mean(axis=1) - adapter[indices].mean(axis=1)
    return {
        "n": int(len(adapter)),
        "repeats": int(repeats),
        "ratio_of_means": float(adapter.mean() / max(prior.mean(), 1e-12)),
        "ratio_ci95": [float(x) for x in np.quantile(ratio, [0.025, 0.975])],
        "delta_prior_minus_adapter": float(prior.mean() - adapter.mean()),
        "delta_ci95": [float(x) for x in np.quantile(delta, [0.025, 0.975])],
        "adapter_better_fraction": float(np.mean(adapter < prior)),
    }


def _summarize(rows: list[dict], condition: str, seed: int, bootstrap_repeats: int) -> dict:
    output: dict[str, object] = {"condition": condition}
    for polymer in ("protein", "rna"):
        for subset in SUBSETS:
            key = f"{polymer}_{subset}"
            adapter = np.asarray([row[f"{condition}_{key}_nll"] for row in rows], dtype=float)
            prior = np.asarray([row[f"P0_{key}_nll"] for row in rows], dtype=float)
            adapter_rec = np.asarray([row[f"{condition}_{key}_recovery"] for row in rows], dtype=float)
            prior_rec = np.asarray([row[f"P0_{key}_recovery"] for row in rows], dtype=float)
            output[key] = {
                "n_complexes": len(rows),
                "adapter_nll": float(np.nanmean(adapter)),
                "prior_nll": float(np.nanmean(prior)),
                "ratio_of_means": float(np.nanmean(adapter) / max(float(np.nanmean(prior)), 1e-12)),
                "mean_of_per_complex_ratios": float(np.nanmean(adapter / np.maximum(prior, 1e-12))),
                "delta_prior_minus_adapter": float(np.nanmean(prior - adapter)),
                "adapter_recovery": float(np.nanmean(adapter_rec)),
                "prior_recovery": float(np.nanmean(prior_rec)),
                "delta_recovery_adapter_minus_prior": float(np.nanmean(adapter_rec - prior_rec)),
                "adapter_better_fraction": float(np.mean(adapter < prior)),
                "paired_bootstrap_10000": _bootstrap(rows, f"{condition}_{key}_nll", f"P0_{key}_nll", seed + len(key), bootstrap_repeats),
            }
    return output


def _quartiles(rows: list[dict], condition: str) -> dict:
    order = np.argsort(np.asarray([row["P0_rna_interface_nll"] for row in rows]), kind="stable")
    bins = np.empty(len(rows), dtype=np.int64)
    bins[order] = np.minimum(3, (np.arange(len(rows)) * 4) // max(len(rows), 1))
    result = {}
    for q in range(4):
        group = [row for row, index in zip(rows, bins) if int(index) == q]
        adapter = np.asarray([row[f"{condition}_rna_interface_nll"] for row in group], dtype=float)
        prior = np.asarray([row["P0_rna_interface_nll"] for row in group], dtype=float)
        result[f"Q{q + 1}"] = {
            "n_complexes": len(group),
            "ratio_of_means": float(np.mean(adapter) / max(float(np.mean(prior)), 1e-12)),
            "delta_prior_minus_adapter": float(np.mean(prior - adapter)),
            "adapter_better_fraction": float(np.mean(adapter < prior)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("test_read") is not False:
        raise RuntimeError("blind lock is already consumed")
    manifest_sha = _sha256(args.manifest)
    if lock.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("blind lock does not match manifest hash")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    data = _load_cache(args.test_cache, "test")
    if len(data) != len(manifest) or {str(row["sample_id"]) for row in data} != set(manifest["sample_id"].astype(str)):
        raise ValueError("blind manifest/cache mismatch")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec = dict(checkpoint["metadata"]["spec"])
    config = ExplicitMatrixConfig(**dict(spec["model_config"]))
    device = torch.device(args.device)
    model = ExplicitSelectionAdapter(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _attach_selected_edges(data, float(spec["radius"]), 32, int(spec["r2p_k"]), int(spec["p2r_k"]), True)

    rows: list[dict] = []
    composition_checks = 0
    composition_failures = 0
    with torch.inference_mode():
        for index, payload in enumerate(data):
            p_labels = payload["protein_native"].to(device)
            r_labels = payload["rna_native"].to(device)
            native = _forward(model, payload, device)
            token_off = _forward(model, payload, device, token_off=True)
            row = {"sample_id": str(payload["sample_id"]), "protein_length": len(p_labels), "rna_length": len(r_labels)}
            for condition, output in (("P1_native", native), ("P2_token_off", token_off)):
                for polymer, labels in (("protein", p_labels), ("rna", r_labels)):
                    for subset in SUBSETS:
                        mask = _mask(payload, polymer, subset).to(device)
                        nll, recovery = _metric(output[f"{polymer}_logits"], labels, mask)
                        row[f"{condition}_{polymer}_{subset}_nll"] = nll
                        row[f"{condition}_{polymer}_{subset}_recovery"] = recovery
            shuffle_nll = {f"{polymer}_{subset}": [] for polymer in ("protein", "rna") for subset in SUBSETS}
            shuffle_rec = {f"{polymer}_{subset}": [] for polymer in ("protein", "rna") for subset in SUBSETS}
            for repeat in range(int(args.shuffle_repeats)):
                shuffled = _shuffle_tokens(payload, int(args.seed), repeat)
                if sorted(shuffled["protein"].tolist()) != sorted(payload["protein_native"].tolist()):
                    composition_failures += 1
                if sorted(shuffled["rna"].tolist()) != sorted(payload["rna_native"].tolist()):
                    composition_failures += 1
                composition_checks += 2
                output = _forward(model, payload, device, token_override=shuffled)
                for polymer, labels in (("protein", p_labels), ("rna", r_labels)):
                    for subset in SUBSETS:
                        mask = _mask(payload, polymer, subset).to(device)
                        nll, recovery = _metric(output[f"{polymer}_logits"], labels, mask)
                        shuffle_nll[f"{polymer}_{subset}"].append(nll)
                        shuffle_rec[f"{polymer}_{subset}"].append(recovery)
            for polymer in ("protein", "rna"):
                for subset in SUBSETS:
                    key = f"{polymer}_{subset}"
                    row[f"P3_partner_shuffle_{key}_nll"] = _finite_mean(shuffle_nll[key])
                    row[f"P3_partner_shuffle_{key}_recovery"] = _finite_mean(shuffle_rec[key])
            for polymer, labels in (("protein", p_labels), ("rna", r_labels)):
                base = payload[f"{polymer}_base"].to(device)
                for subset in SUBSETS:
                    mask = _mask(payload, polymer, subset).to(device)
                    nll, recovery = _metric(base, labels, mask)
                    row[f"P0_{polymer}_{subset}_nll"] = nll
                    row[f"P0_{polymer}_{subset}_recovery"] = recovery
            rows.append(row)
            print(json.dumps({"event": "blind_complex_complete", "completed": index + 1, "total": len(data)}), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_complex.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = {condition: _summarize(rows, condition, int(args.seed), int(args.bootstrap_repeats)) for condition in CONDITIONS}
    comparisons = {condition: {"protein_interface": summaries[condition]["protein_interface"], "rna_interface": summaries[condition]["rna_interface"], "rna_quartiles": _quartiles(rows, condition)} for condition in CONDITIONS}
    for left, right in (("P1_native", "P2_token_off"), ("P1_native", "P3_partner_shuffle")):
        comparisons[f"{left}_vs_{right}"] = {}
        for polymer in ("protein", "rna"):
            for subset in SUBSETS:
                key = f"{polymer}_{subset}"
                left_values = np.asarray([row[f"{left}_{key}_nll"] for row in rows], dtype=float)
                right_values = np.asarray([row[f"{right}_{key}_nll"] for row in rows], dtype=float)
                comparisons[f"{left}_vs_{right}"][key] = {
                    "left_nll": float(np.mean(left_values)),
                    "right_nll": float(np.mean(right_values)),
                    "left_minus_right": float(np.mean(left_values - right_values)),
                    "native_better_fraction": float(np.mean(left_values < right_values)),
                }
    summary = {
        "protocol": {
            "test_read": True,
            "blind_complexes": len(rows),
            "prior_retrained": False,
            "adapter_variant": "A2",
            "adapter_refit": True,
            "single_seed": int(args.seed),
            "conditions": "P0 frozen priors; P1 native A2; P2 token-off; P3 20 composition-preserving partner shuffles",
            "shuffle_repeats": int(args.shuffle_repeats),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "manifest": str(args.manifest),
            "manifest_sha256": manifest_sha,
            "checkpoint": str(args.checkpoint),
        },
        "spec": spec,
        "conditions": summaries,
        "comparisons": comparisons,
        "audits": {
            "manifest_cache_id_match": True,
            "composition_checks": composition_checks,
            "composition_failures": composition_failures,
            "active_mask_source": "same frozen cache masks for P0-P3",
        },
    }
    (out / "blind_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lock["test_read"] = True
    lock["summary"] = str(out / "blind_summary.json")
    args.lock.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "blind_evaluation_complete", "out": str(out), "test_read": True}), flush=True)


if __name__ == "__main__":
    main()
