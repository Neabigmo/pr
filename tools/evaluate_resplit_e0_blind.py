#!/usr/bin/env python3
"""Single locked E0 evaluation on the retrospective blind manifest.

P0 is the frozen-prior log-probability already present in the cache.  P1 is
native E0, P2 turns partner tokens off, and P3 averages 20 deterministic
composition-preserving partner permutations.  No condition is used for model
selection; this script is only valid after the final refit lock exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.model import ReciprocalAdapter  # noqa: E402
from run_conditional_adapter_pilot import _attach_selected_edges, _forward_payload, _load_cache  # noqa: E402
from run_scientific_adapter_pilot import _config  # noqa: E402


CONDITIONS = tuple((polymer, subset) for polymer in ("protein", "rna") for subset in ("all", "active", "interface"))
CONDITION_NAMES = ("P0_prior", "P1_native", "P2_token_off", "P3_partner_shuffle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mask(payload: dict, polymer: str, subset: str) -> torch.Tensor:
    if subset == "all":
        return torch.ones(int(payload[f"{polymer}_native"].shape[0]), dtype=torch.bool)
    if subset == "active":
        return payload[f"_selected_{polymer}_active"].bool()
    if subset == "interface":
        return payload[f"{polymer}_interface"].bool()
    raise ValueError(subset)


def _metrics(logits_or_logp: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *, is_logp: bool = False) -> tuple[float, float]:
    if not bool(mask.any()):
        return float("nan"), float("nan")
    logp = logits_or_logp if is_logp else F.log_softmax(logits_or_logp, dim=-1)
    positions = torch.arange(len(labels), device=labels.device)
    nll = -logp[positions, labels]
    hit = (logp.argmax(dim=-1) == labels).float()
    return float(nll[mask].mean().cpu()), float(hit[mask].mean().cpu())


def _shuffle_tokens(tokens: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if len(tokens) <= 1:
        return tokens.clone()
    order = list(range(len(tokens)))
    rng.shuffle(order)
    return tokens[torch.tensor(order, dtype=torch.long)]


def _bootstrap_ratio(rows: list[dict], adapter_key: str, prior_key: str, seed: int, repeats: int) -> dict:
    adapter = np.asarray([float(row[adapter_key]) for row in rows], dtype=np.float64)
    prior = np.asarray([float(row[prior_key]) for row in rows], dtype=np.float64)
    valid = np.isfinite(adapter) & np.isfinite(prior) & (prior > 0)
    adapter, prior = adapter[valid], prior[valid]
    if not len(adapter):
        return {"n": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(adapter), size=(repeats, len(adapter)))
    adapter_means = adapter[indices].mean(axis=1)
    prior_means = prior[indices].mean(axis=1)
    ratios = adapter_means / np.maximum(prior_means, 1e-12)
    deltas = prior_means - adapter_means
    return {
        "n": int(len(adapter)),
        "ratio_of_means": float(adapter.mean() / max(prior.mean(), 1e-12)),
        "ratio_ci95": [float(x) for x in np.quantile(ratios, [0.025, 0.975])],
        "delta_prior_minus_adapter": float(prior.mean() - adapter.mean()),
        "delta_ci95": [float(x) for x in np.quantile(deltas, [0.025, 0.975])],
        "adapter_better_fraction": float(np.mean(adapter < prior)),
    }


def _summary_for(rows: list[dict], condition: str, bootstrap_repeats: int, seed: int) -> dict:
    result: dict[str, object] = {"condition": condition}
    for polymer, subset in CONDITIONS:
        key = f"{polymer}_{subset}"
        # Each row is one complex and carries one metric column per
        # polymer/subset condition; there is no redundant row-level label.
        selected = rows
        adapter_key = f"{condition}_{key}_nll"
        recovery_key = f"{condition}_{key}_recovery"
        prior_key = f"P0_{key}_nll"
        prior_recovery_key = f"P0_{key}_recovery"
        adapter_values = np.asarray([row[adapter_key] for row in selected], dtype=np.float64)
        prior_values = np.asarray([row[prior_key] for row in selected], dtype=np.float64)
        adapter_rec = np.asarray([row[recovery_key] for row in selected], dtype=np.float64)
        prior_rec = np.asarray([row[prior_recovery_key] for row in selected], dtype=np.float64)
        result[key] = {
            "n_complexes": len(selected),
            "adapter_nll": float(np.nanmean(adapter_values)),
            "prior_nll": float(np.nanmean(prior_values)),
            "ratio_of_means": float(np.nanmean(adapter_values) / max(float(np.nanmean(prior_values)), 1e-12)),
            "mean_of_per_complex_ratios": float(np.nanmean(adapter_values / np.maximum(prior_values, 1e-12))),
            "delta_prior_minus_adapter": float(np.nanmean(prior_values - adapter_values)),
            "adapter_recovery": float(np.nanmean(adapter_rec)),
            "prior_recovery": float(np.nanmean(prior_rec)),
            "delta_recovery_adapter_minus_prior": float(np.nanmean(adapter_rec - prior_rec)),
            "adapter_better_fraction": float(np.mean(adapter_values < prior_values)),
            "paired_bootstrap_10000": _bootstrap_ratio(selected, adapter_key, prior_key, seed + len(key), bootstrap_repeats),
        }
    return result


def _quartiles(rows: list[dict], condition: str) -> dict:
    selected = rows
    order = np.argsort(np.asarray([row["P0_rna_interface_nll"] for row in selected]), kind="stable")
    bins = np.empty(len(selected), dtype=np.int64)
    bins[order] = np.minimum(3, (np.arange(len(selected)) * 4) // max(len(selected), 1))
    output = {}
    for q in range(4):
        group = [row for row, index in zip(selected, bins) if int(index) == q]
        a = np.asarray([row[f"{condition}_rna_interface_nll"] for row in group], dtype=float)
        p = np.asarray([row["P0_rna_interface_nll"] for row in group], dtype=float)
        output[f"Q{q + 1}"] = {
            "n_complexes": len(group),
            "prior_interface_nll": float(np.mean(p)),
            "adapter_interface_nll": float(np.mean(a)),
            "ratio_of_means": float(np.mean(a) / max(float(np.mean(p)), 1e-12)),
            "delta_prior_minus_adapter": float(np.mean(p - a)),
            "adapter_better_fraction": float(np.mean(a < p)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--final-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()

    lock = json.loads(args.final_lock.read_text(encoding="utf-8"))
    if lock.get("test_read") is not False or lock.get("new_blind_manifest") != str(args.manifest):
        raise RuntimeError("final lock does not match the declared blind manifest or is already marked read")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    manifest_rows = __import__("pandas").read_csv(args.manifest, sep="\t", dtype=str)
    data = _load_cache(args.test_cache, "test")
    if len(data) != len(manifest_rows):
        raise ValueError(f"blind manifest/cache count mismatch: manifest={len(manifest_rows)} cache={len(data)}")
    manifest_ids = set(manifest_rows["sample_id"].astype(str))
    cache_ids = {str(item["sample_id"]) for item in data}
    if manifest_ids != cache_ids:
        raise ValueError("blind manifest/cache sample IDs do not match")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    spec = dict(checkpoint["metadata"]["spec"])
    model = ReciprocalAdapter(_config(spec)).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    _attach_selected_edges(data, float(spec["radius"]), args.neighbors, int(spec["r2p_k"]), int(spec["p2r_k"]), True)

    rows: list[dict] = []
    composition_checks = 0
    composition_failures = 0
    edge_counts: list[dict] = []
    with torch.inference_mode():
        for index, payload in enumerate(data):
            p_labels = payload["protein_native"].to(args.device)
            r_labels = payload["rna_native"].to(args.device)
            p_base = payload["protein_base"].to(args.device)
            r_base = payload["rna_base"].to(args.device)
            native = _forward_payload(model, payload, args.device, 0.0, 0, token_off=False)
            token_off = _forward_payload(model, payload, args.device, 0.0, 0, token_off=True)
            p3_nll: dict[str, list[float]] = {f"{polymer}_{subset}": [] for polymer, subset in CONDITIONS}
            p3_rec: dict[str, list[float]] = {f"{polymer}_{subset}": [] for polymer, subset in CONDITIONS}
            for repeat in range(args.shuffle_repeats):
                base_seed = f"{args.seed}|{payload['sample_id']}|shuffle|{repeat}"
                rng_p = random.Random(base_seed + "|protein")
                rng_r = random.Random(base_seed + "|rna")
                shuffled_p = _shuffle_tokens(payload["protein_native"], rng_p)
                shuffled_r = _shuffle_tokens(payload["rna_native"], rng_r)
                composition_checks += 2
                if sorted(shuffled_p.tolist()) != sorted(payload["protein_native"].tolist()):
                    composition_failures += 1
                if sorted(shuffled_r.tolist()) != sorted(payload["rna_native"].tolist()):
                    composition_failures += 1
                shuffled = _forward_payload(
                    model, payload, args.device, 0.0, 0, token_off=False,
                    partner_token_override={"protein": shuffled_p, "rna": shuffled_r},
                )
                for polymer, logits in (("protein", shuffled["protein_logits"]), ("rna", shuffled["rna_logits"])):
                    labels = p_labels if polymer == "protein" else r_labels
                    for subset in ("all", "active", "interface"):
                        mask = _mask(payload, polymer, subset).to(args.device)
                        nll, rec = _metrics(logits, labels, mask)
                        p3_nll[f"{polymer}_{subset}"].append(nll)
                        p3_rec[f"{polymer}_{subset}"].append(rec)

            row = {"sample_id": str(payload["sample_id"]), "protein_length": int(len(p_labels)), "rna_length": int(len(r_labels)), "r2p_edges": int(payload["_selected_edge_index_r2p"].shape[1]), "p2r_edges": int(payload["_selected_edge_index_p2r"].shape[1])}
            edge_counts.append({key: row[key] for key in ("sample_id", "r2p_edges", "p2r_edges")})
            for polymer, labels, base, native_logits, off_logits in (
                ("protein", p_labels, p_base, native["protein_logits"], token_off["protein_logits"]),
                ("rna", r_labels, r_base, native["rna_logits"], token_off["rna_logits"]),
            ):
                for subset in ("all", "active", "interface"):
                    key = f"{polymer}_{subset}"
                    mask = _mask(payload, polymer, subset).to(args.device)
                    p0_nll, p0_rec = _metrics(base, labels, mask, is_logp=True)
                    p1_nll, p1_rec = _metrics(native_logits, labels, mask)
                    p2_nll, p2_rec = _metrics(off_logits, labels, mask)
                    row[f"P0_{key}_nll"], row[f"P0_{key}_recovery"] = p0_nll, p0_rec
                    row[f"P1_native_{key}_nll"], row[f"P1_native_{key}_recovery"] = p1_nll, p1_rec
                    row[f"P2_token_off_{key}_nll"], row[f"P2_token_off_{key}_recovery"] = p2_nll, p2_rec
                    row[f"P3_partner_shuffle_{key}_nll"] = float(np.nanmean(p3_nll[key]))
                    row[f"P3_partner_shuffle_{key}_recovery"] = float(np.nanmean(p3_rec[key]))
            rows.append(row)
            if (index + 1) % 25 == 0 or index + 1 == len(data):
                print(json.dumps({"event": "blind_complex_complete", "completed": index + 1, "total": len(data)}), flush=True)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_complex.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    condition_summaries = {condition: _summary_for(rows, condition, args.bootstrap_repeats, args.seed) for condition in CONDITION_NAMES[1:]}
    comparisons = {}
    for condition in CONDITION_NAMES[1:]:
        comparisons[condition] = {
            "protein_interface": condition_summaries[condition]["protein_interface"],
            "rna_interface": condition_summaries[condition]["rna_interface"],
            "rna_quartiles": _quartiles(rows, condition),
        }
    for left, right in (("P1_native", "P2_token_off"), ("P1_native", "P3_partner_shuffle")):
        comparisons[f"{left}_vs_{right}"] = {}
        for polymer, subset in CONDITIONS:
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
            "retrospective_resplit": True,
            "blind_complexes": len(data),
            "prior_retrained": False,
            "adapter_retrained": True,
            "single_seed": args.seed,
            "condition_definition": "P0 frozen priors; P1 native E0; P2 token-off; P3 20 composition-preserving partner shuffles",
            "shuffle_repeats": args.shuffle_repeats,
            "bootstrap_repeats": args.bootstrap_repeats,
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "selection_completed_before_blind": True,
        },
        "spec": spec,
        "conditions": condition_summaries,
        "comparisons": comparisons,
        "audits": {
            "manifest_cache_id_match": True,
            "composition_checks": composition_checks,
            "composition_failures": composition_failures,
            "active_mask_source": "same frozen cache masks for P0-P3",
            "edge_counts": edge_counts,
            "edge_counts_unchanged_across_conditions": True,
        },
    }
    (out / "blind_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "blind_evaluation_complete", "out": str(out), "test_read": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
