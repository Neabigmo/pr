#!/usr/bin/env python3
"""Run the partner-use Adapter V2 grouped-CV pilot.

The script intentionally reads only the 1,000-complex development cache
(former train+val) and its development manifests. The frozen 86-complex test
set is not accepted as an input to this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pr_pilot.adapter_pilot.model import AdapterConfig, ReciprocalAdapter
from pr_pilot.data.manifest import _labels as _manifest_labels
from pr_pilot.data.manifest import _rna_holdout_labels
from pr_pilot.data.manifest import bilateral_components

from run_conditional_adapter_pilot import (
    _attach_selected_edges,
    _batched_loss,
    _collate_payloads,
    _forward_payload,
    _load_cache,
    _seed_everything,
)


DEFAULT_CACHE = Path(r"F:\111临时\PR PILOT\pilot_conditional_adapter_20260916\cache\noise0p0")
DEFAULT_MANIFESTS = Path(r"F:\111临时\PR PILOT\remote_return_20260911\manifests\round_20260905_exception_v2")
DEFAULT_OUT = Path(r"F:\111临时\PR PILOT\pilot_adapter_v2_20260917")
RADIUS = 14.357456359863283


def _hash_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _distribution(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "q05": float("nan"), "q95": float("nan"), "rms": float("nan")}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "rms": float(np.sqrt(np.mean(array * array))),
    }


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def build_grouped_folds(manifests: Path, n_folds: int = 3, seed: int = 20260917) -> list[dict]:
    """Assign bilateral P30/R80/Rfam components to balanced CV folds."""
    if "test" in str(manifests).lower():
        raise ValueError("Adapter V2 grouped CV must not read a test manifest")
    train_path = manifests / "complex_train.tsv"
    val_path = manifests / "complex_val.tsv"
    frame = pd.concat([pd.read_csv(train_path, sep="\t"), pd.read_csv(val_path, sep="\t")], ignore_index=True)
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("development manifests contain duplicate sample_id")
    components = bilateral_components(frame)
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    group_of = {sample_id: group_id for group_id, component in enumerate(components) for sample_id in component}
    if len(group_of) != len(frame):
        raise ValueError("bilateral component assignment does not cover every development complex")
    group_stats: list[dict] = []
    for group_id, component in enumerate(components):
        rows = frame[frame["sample_id"].isin(component)]
        group_stats.append(
            {
                "group_id": group_id,
                "sample_ids": sorted(component),
                "n": len(rows),
                "protein_length": float(rows["protein_length"].sum()),
                "rna_length": float(rows["rna_length"].sum()),
                "interface_size": float(rows["interface_residue_pairs"].sum()),
            }
        )
    features = ["n", "protein_length", "rna_length", "interface_size"]
    target = {feature: sum(item[feature] for item in group_stats) / n_folds for feature in features}
    assigned: list[list[dict]] = [[] for _ in range(n_folds)]
    totals = [{feature: 0.0 for feature in features} for _ in range(n_folds)]
    ordered = sorted(group_stats, key=lambda item: (-item["n"], _hash_key(seed, str(item["group_id"]))))
    for item in ordered:
        scores = []
        for fold in range(n_folds):
            projected = {feature: totals[fold][feature] + item[feature] for feature in features}
            load_score = projected["n"] / max(target["n"], 1.0)
            feature_score = sum(((projected[feature] - target[feature]) / max(target[feature], 1.0)) ** 2 for feature in features[1:])
            scores.append((load_score, feature_score, fold))
        fold = min(scores)[2]
        assigned[fold].append(item)
        for feature in features:
            totals[fold][feature] += item[feature]

    fold_records = []
    for fold_id, groups in enumerate(assigned):
        val_ids = sorted(sample_id for item in groups for sample_id in item["sample_ids"])
        train_ids = sorted(set(frame["sample_id"]) - set(val_ids))
        fold_records.append(
            {
                "fold": fold_id,
                "train_sample_ids": train_ids,
                "val_sample_ids": val_ids,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "group_ids": [item["group_id"] for item in groups],
                "val_totals": totals[fold_id],
            }
        )
    # Explicitly re-check the leakage contract across all folds.
    for left in range(n_folds):
        for right in range(left + 1, n_folds):
            left_rows = frame[frame["sample_id"].isin(fold_records[left]["val_sample_ids"])]
            right_rows = frame[frame["sample_id"].isin(fold_records[right]["val_sample_ids"])]
            left_protein = set().union(*(_manifest_labels(value) for value in left_rows["protein_cluster_p30"]))
            right_protein = set().union(*(_manifest_labels(value) for value in right_rows["protein_cluster_p30"]))
            left_rna = set().union(*(_rna_holdout_labels(row) for _, row in left_rows.iterrows()))
            right_rna = set().union(*(_rna_holdout_labels(row) for _, row in right_rows.iterrows()))
            if left_protein & right_protein:
                raise AssertionError(f"protein P30 leakage between folds {left} and {right}")
            if left_rna & right_rna:
                raise AssertionError(f"RNA R80/Rfam leakage between folds {left} and {right}")
    return fold_records


def _config_for(experiment: str) -> AdapterConfig:
    if experiment not in {"C0", "C1", "C2", "C3", "C4"}:
        raise ValueError(f"unknown experiment {experiment}")
    v2 = experiment != "C0"
    return AdapterConfig(
        geometry="G2",
        aggregation="A2",
        hidden_dim=128,
        hidden_projection_dim=64,
        edge_dim=64,
        token_dim=64 if v2 else 32,
        message_dim=128,
        layers=1,
        dropout=0.1,
        rbf_bins=16,
        separate_edge_encoders=False,
        sequence_independent_attention=v2,
        partner_centered_residual=experiment in {"C2", "C3", "C4"},
        modality_projector=experiment in {"C3", "C4"},
        conservative_gate=experiment == "C4",
        gate_init=0.1,
    )


def _interface_distance(mask: torch.Tensor) -> np.ndarray:
    positions = torch.where(mask)[0].cpu().numpy()
    if positions.size == 0:
        return np.full((len(mask),), np.inf)
    return np.min(np.abs(np.arange(len(mask))[:, None] - positions[None, :]), axis=1)


def _distance_bin(distance: float) -> str:
    if distance == 0:
        return "0"
    if distance <= 2:
        return "1-2"
    if distance <= 5:
        return "3-5"
    return "6+"


def _confidence_bin(value: float, boundaries: np.ndarray) -> str:
    index = int(np.searchsorted(boundaries[1:-1], value, side="right"))
    return ("q1", "q2", "q3", "q4")[min(index, 3)]


def evaluate_dataset(
    model: ReciprocalAdapter,
    data: list[dict],
    device: torch.device,
    radius: float,
    neighbors: int,
    include_permutation: bool = False,
) -> dict:
    metric_values: dict[str, list[float]] = {}
    distance_rows: dict[str, list[tuple[str, float]]] = {"protein": [], "rna": []}
    confidence_rows: dict[str, list[tuple[float, float]]] = {"protein": [], "rna": []}

    def add(key: str, value: float) -> None:
        metric_values.setdefault(key, []).append(float(value))

    with torch.no_grad():
        for payload in data:
            native = _forward_payload(model, payload, device, radius, neighbors, token_off=False)
            token_off = _forward_payload(model, payload, device, radius, neighbors, token_off=True)
            permuted = None
            if include_permutation:
                p_tokens = payload["protein_native"].roll(1) if len(payload["protein_native"]) > 1 else payload["protein_native"]
                r_tokens = payload["rna_native"].roll(1) if len(payload["rna_native"]) > 1 else payload["rna_native"]
                permuted = _forward_payload(model, payload, device, radius, neighbors, partner_token_override={"protein": p_tokens, "rna": r_tokens})
            for polymer, base, native_key, interface, active, delta_key, null_key, gate_key, off_key, perm_key in (
                ("protein", payload["protein_base"], "protein_native", payload["protein_interface"], payload["_selected_protein_active"], "protein_delta", "protein_null_weight", "protein_gate", "protein_logits", "protein_logits"),
                ("rna", payload["rna_base"], "rna_native", payload["rna_interface"], payload["_selected_rna_active"], "rna_delta", "rna_null_weight", "rna_gate", "rna_logits", "rna_logits"),
            ):
                native_logits = native[native_key.replace("_native", "_logits")]
                off_logits = token_off[off_key]
                perm_logits = permuted[perm_key] if permuted is not None else None
                base = base.to(device)
                labels = payload[native_key].to(device)
                interface = interface.to(device)
                active = active.to(device)
                native_logp = F.log_softmax(native_logits, dim=-1)
                off_logp = F.log_softmax(off_logits, dim=-1)
                perm_logp = F.log_softmax(perm_logits, dim=-1) if perm_logits is not None else None
                nll = -native_logp[torch.arange(len(labels), device=device), labels]
                prior_nll = -base[torch.arange(len(labels), device=device), labels]
                off_nll = -off_logp[torch.arange(len(labels), device=device), labels]
                perm_nll = -perm_logp[torch.arange(len(labels), device=device), labels] if perm_logp is not None else None
                for subset, mask in (("all", torch.ones_like(active)), ("active", active), ("interface", interface)):
                    if bool(mask.any()):
                        add(f"{polymer}_{subset}_nll", float(nll[mask].mean().cpu()))
                        add(f"{polymer}_{subset}_prior_nll", float(prior_nll[mask].mean().cpu()))
                        add(f"{polymer}_{subset}_recovery", float((native_logp[mask].argmax(-1) == labels[mask]).float().mean().cpu()))
                        if subset == "interface":
                            add(f"{polymer}_token_off_interface_nll", float(off_nll[mask].mean().cpu()))
                            if perm_nll is not None:
                                add(f"{polymer}_partner_permutation_interface_nll", float(perm_nll[mask].mean().cpu()))
                null = native[null_key].detach().float().cpu()
                add(f"{polymer}_null_weight_mean", float(null.mean()))
                add(f"{polymer}_null_weight_median", float(null.median()))
                # Scalar gates are used by legacy V2; confidence gates are
                # position-wise.  Report one comparable mean without making
                # the evaluator depend on the residual implementation.
                add(f"{polymer}_gate", float(native[gate_key].detach().float().mean().cpu()))
                delta = native[delta_key]
                delta_rms = delta.pow(2).mean(dim=-1).sqrt()
                prior_rms = base.pow(2).mean(dim=-1).sqrt()
                add(f"{polymer}_delta_rms", float(delta_rms[interface].mean().cpu()) if bool(interface.any()) else 0.0)
                add(f"{polymer}_prior_logits_rms", float(prior_rms[interface].mean().cpu()) if bool(interface.any()) else 0.0)
                if bool(interface.any()):
                    add(f"{polymer}_delta_prior_rms_ratio", float((delta_rms[interface] / prior_rms[interface].clamp_min(1e-8)).mean().cpu()))
                distance = _interface_distance(interface)
                distance_rows[polymer].extend((_distance_bin(float(distance[index])), float(delta_rms[index].detach().cpu())) for index in range(len(distance)))
                confidence_rows[polymer].extend((float(prior_nll[index].detach().cpu()), float(delta_rms[index].detach().cpu())) for index in range(len(prior_nll)))

    result = {key: _mean(values) for key, values in metric_values.items()}
    for polymer in ("protein", "rna"):
        result[f"{polymer}_interface_ratio"] = result[f"{polymer}_interface_nll"] / max(result[f"{polymer}_interface_prior_nll"], 1e-8)
        result[f"{polymer}_native_minus_token_off_interface_nll"] = result[f"{polymer}_interface_nll"] - result[f"{polymer}_token_off_interface_nll"]
        if include_permutation:
            result[f"{polymer}_native_minus_permutation_interface_nll"] = result[f"{polymer}_interface_nll"] - result[f"{polymer}_partner_permutation_interface_nll"]
        distance_summary = {}
        for label in ("0", "1-2", "3-5", "6+"):
            values = [value for bin_name, value in distance_rows[polymer] if bin_name == label]
            distance_summary[label] = {"n": len(values), "mean_delta_rms": _mean(values)}
        result[f"{polymer}_delta_rms_by_distance_to_interface"] = distance_summary
        confidence_values = confidence_rows[polymer]
        boundaries = np.quantile([value for value, _ in confidence_values], [0.0, 0.25, 0.5, 0.75, 1.0]) if confidence_values else np.zeros(5)
        confidence_summary = {}
        for label in ("q1", "q2", "q3", "q4"):
            values = [delta for confidence, delta in confidence_values if _confidence_bin(confidence, boundaries) == label]
            confidence_summary[label] = {"n": len(values), "mean_delta_rms": _mean(values)}
        result[f"{polymer}_delta_rms_by_prior_confidence_nll_bin"] = confidence_summary
    result["relative_interface_nll"] = 0.5 * (result["protein_interface_ratio"] + result["rna_interface_ratio"])
    result["worst_direction_ratio"] = max(result["protein_interface_ratio"], result["rna_interface_ratio"])
    result["partner_gate_pass"] = bool(
        result["protein_interface_ratio"] < 1.0
        and result["rna_interface_ratio"] < 1.0
        and result["protein_native_minus_token_off_interface_nll"] < 0.0
        and result["rna_native_minus_token_off_interface_nll"] < 0.0
    )
    result["native_better_than_token_off"] = {
        "protein": result["protein_native_minus_token_off_interface_nll"] < 0.0,
        "rna": result["rna_native_minus_token_off_interface_nll"] < 0.0,
    }
    return result


def hidden_and_embedding_diagnostics(model: ReciprocalAdapter, data: list[dict]) -> dict:
    values: dict[str, list[float]] = {
        "protein_hidden_raw_norm": [],
        "rna_hidden_raw_norm": [],
        "protein_hidden_adapter_input_norm": [],
        "rna_hidden_adapter_input_norm": [],
        "edge_embedding_norm": [],
        "rna_token_embedding_norm": [],
        "protein_token_embedding_norm": [],
    }
    device = next(model.parameters()).device
    with torch.no_grad():
        for payload in data:
            p_h = payload["protein_hidden"].to(device)
            r_h = payload["rna_hidden"].to(device)
            values["protein_hidden_raw_norm"].extend(p_h.norm(dim=-1).tolist())
            values["rna_hidden_raw_norm"].extend(r_h.norm(dim=-1).tolist())
            p_input, r_input = model._hidden_inputs(p_h, r_h)
            values["protein_hidden_adapter_input_norm"].extend(p_input.norm(dim=-1).tolist())
            values["rna_hidden_adapter_input_norm"].extend(r_input.norm(dim=-1).tolist())
            for edge_index_key, geometry_key, encoder in (
                ("_selected_edge_index_r2p", "_selected_geometry_r2p", model.edge_encoder_r2p or model.shared_edge_encoder),
                ("_selected_edge_index_p2r", "_selected_geometry_p2r", model.edge_encoder_p2r or model.shared_edge_encoder),
            ):
                del edge_index_key
                geometry = model._select_geometry(payload[geometry_key].to(device))
                if len(geometry):
                    values["edge_embedding_norm"].extend(encoder(geometry).norm(dim=-1).tolist())
            values["rna_token_embedding_norm"].extend(model.r2p.token_embedding(payload["rna_native"].to(device)).norm(dim=-1).tolist())
            values["protein_token_embedding_norm"].extend(model.p2r.token_embedding(payload["protein_native"].to(device)).norm(dim=-1).tolist())
    return {key: _distribution(value) for key, value in values.items()}


def _selection(metrics: dict) -> tuple[float, bool, dict]:
    ratio_ok = metrics["protein_interface_ratio"] < 1.0 and metrics["rna_interface_ratio"] < 1.0
    token_ok = metrics["protein_native_minus_token_off_interface_nll"] < 0.0 and metrics["rna_native_minus_token_off_interface_nll"] < 0.0
    gate_pass = bool(ratio_ok and token_ok)
    ratio_penalty = 0.0 if ratio_ok else 10.0
    token_penalty = 0.0 if token_ok else 1.0 + sum(max(0.0, metrics[f"{polymer}_native_minus_token_off_interface_nll"]) for polymer in ("protein", "rna"))
    score = max(metrics["protein_interface_ratio"], metrics["rna_interface_ratio"]) + ratio_penalty + token_penalty
    return score, gate_pass, {"ratio_ok": ratio_ok, "token_ok": token_ok, "gate_pass": gate_pass, "score": score}


def train_fold(
    experiment: str,
    fold_id: int,
    train_data: list[dict],
    val_data: list[dict],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    seed = int(args.seed) + fold_id
    _seed_everything(seed)
    _attach_selected_edges(train_data, args.radius, args.neighbors, args.r2p_k, args.p2r_k, True)
    _attach_selected_edges(val_data, args.radius, args.neighbors, args.r2p_k, args.p2r_k, True)
    model = ReciprocalAdapter(_config_for(experiment)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    best_score = float("inf")
    best_gate = False
    bad = 0
    history = []
    order_rng = random.Random(seed)
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = list(range(len(train_data)))
        order_rng.shuffle(order)
        losses = []
        for start in range(0, len(order), int(args.batch_size)):
            payloads = [train_data[index] for index in order[start : start + int(args.batch_size)]]
            packed = _collate_payloads(payloads, args.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                packed["protein_base"], packed["rna_base"], packed["protein_hidden"], packed["rna_hidden"],
                packed["protein_native"], packed["rna_native"], packed["edge_index_r2p"], packed["edge_geometry_r2p"],
                packed["edge_index_p2r"], packed["edge_geometry_p2r"],
            )
            p_loss = _batched_loss(out["protein_logits"], packed["protein_native"], packed["protein_active"], packed["protein_sample"], packed["batch_size"])
            r_loss = _batched_loss(out["rna_logits"], packed["rna_native"], packed["rna_active"], packed["rna_sample"], packed["batch_size"])
            p_prior = _batched_loss(packed["protein_base"].detach(), packed["protein_native"], packed["protein_active"], packed["protein_sample"], packed["batch_size"], log_probs=True)
            r_prior = _batched_loss(packed["rna_base"].detach(), packed["rna_native"], packed["rna_active"], packed["rna_sample"], packed["batch_size"], log_probs=True)
            loss = 0.5 * (p_loss / p_prior.clamp_min(1e-8) + r_loss / r_prior.clamp_min(1e-8))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_metrics = evaluate_dataset(model, val_data, args.device, args.radius, args.neighbors, include_permutation=False)
        score, gate_pass, gate_detail = _selection(val_metrics)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_score": score, **gate_detail, **{key: value for key, value in val_metrics.items() if not isinstance(value, (dict, list))}}
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "config": _config_for(experiment).__dict__,
            "experiment": experiment,
            "fold": fold_id,
            "epoch": epoch,
            "seed": seed,
            "radius": args.radius,
            "neighbors": args.neighbors,
            "r2p_neighbors": args.r2p_k,
            "p2r_neighbors": args.p2r_k,
            "direction_specific": True,
            "loss_mode": "prior_normalized",
            "selection_metric": "partner_use_gated_score",
            "selection_value": score,
            "partner_gate_pass": gate_pass,
        }
        torch.save(checkpoint, last_path)
        if score < best_score - 1e-7:
            best_score = score
            best_gate = gate_pass
            bad = 0
            torch.save(checkpoint, best_path)
        else:
            bad += 1
        if bad >= int(args.patience):
            break
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    model.eval()
    best_metrics = evaluate_dataset(model, val_data, args.device, args.radius, args.neighbors, include_permutation=True)
    diagnostics = {
        "validation": best_metrics,
        "hidden_and_embedding_norms": hidden_and_embedding_diagnostics(model, val_data),
        "fold": fold_id,
        "experiment": experiment,
        "test_read": False,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    summary = {
        "experiment": experiment,
        "fold": fold_id,
        "train_complexes": len(train_data),
        "val_complexes": len(val_data),
        "epochs_run": len(history),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_selection_score": best_score,
        "best_partner_gate_pass": best_gate,
        "best_validation": best_metrics,
        "config": _config_for(experiment).__dict__,
        "radius": args.radius,
        "neighbors": args.neighbors,
        "r2p_neighbors": args.r2p_k,
        "p2r_neighbors": args.p2r_k,
        "seed": seed,
        "batch_size": args.batch_size,
        "checkpoint": str(best_path),
        "diagnostics": str(out_dir / "diagnostics.json"),
        "test_read": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return summary


def _aggregate_cv(fold_summaries: list[dict]) -> dict:
    metrics = [summary["best_validation"] for summary in fold_summaries]
    scalar_keys = sorted(key for key, value in metrics[0].items() if isinstance(value, (int, float)) and np.isfinite(value))
    mean_metrics = {key: float(np.mean([float(item[key]) for item in metrics])) for key in scalar_keys}
    gate_count = sum(bool(summary["best_partner_gate_pass"]) for summary in fold_summaries)
    mean_gate = {
        "protein_ratio_lt_1": mean_metrics.get("protein_interface_ratio", float("inf")) < 1.0,
        "rna_ratio_lt_1": mean_metrics.get("rna_interface_ratio", float("inf")) < 1.0,
        "protein_native_better_than_off": mean_metrics.get("protein_native_minus_token_off_interface_nll", float("inf")) < 0.0,
        "rna_native_better_than_off": mean_metrics.get("rna_native_minus_token_off_interface_nll", float("inf")) < 0.0,
    }
    return {
        "folds": len(fold_summaries),
        "gate_pass_folds": gate_count,
        "mean_metrics": mean_metrics,
        "mean_gate_conditions": mean_gate,
        "promotion_gate_pass": gate_count >= 2 and all(mean_gate.values()),
        "fold_summaries": fold_summaries,
    }


def run_cv(args) -> dict:
    cache_root = Path(args.cache_root)
    manifests = Path(args.manifests)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "test" in str(manifests).lower() or "test" in str(cache_root).lower():
        raise ValueError("Adapter V2 CV refuses test manifests or test caches")
    data = _load_cache(cache_root, "train") + _load_cache(cache_root, "val")
    by_id = {str(payload["sample_id"]): payload for payload in data}
    folds = build_grouped_folds(manifests, 3, int(args.seed))
    if set(by_id) != set(sample_id for fold in folds for sample_id in fold["train_sample_ids"]):
        raise ValueError("cache sample IDs do not match the 1,000 development manifest IDs")
    (out / "folds.json").write_text(json.dumps(folds, indent=2, ensure_ascii=False), encoding="utf-8")
    all_results = {}
    for experiment in args.experiments:
        fold_summaries = []
        for fold in folds:
            target = out / experiment / f"fold{fold['fold']}"
            if (target / "summary.json").exists() and not args.force:
                summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
            else:
                train_data = [by_id[sample_id] for sample_id in fold["train_sample_ids"]]
                val_data = [by_id[sample_id] for sample_id in fold["val_sample_ids"]]
                summary = train_fold(experiment, int(fold["fold"]), train_data, val_data, target, args)
            fold_summaries.append(summary)
            print(json.dumps({"experiment": experiment, "fold": fold["fold"], "gate": summary["best_partner_gate_pass"], "ratio": {k: summary["best_validation"][k] for k in ("protein_interface_ratio", "rna_interface_ratio")}}, ensure_ascii=False), flush=True)
        all_results[experiment] = _aggregate_cv(fold_summaries)
        (out / experiment / "cv_summary.json").write_text(json.dumps(all_results[experiment], indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    summary = {
        "experiments": all_results,
        "protocol": {
            "development_complexes": len(data),
            "folds": 3,
            "grouping": "bilateral connected components over Protein P30, RNA R80, and Rfam",
            "stratification_targets": ["complex_count", "protein_length", "rna_length", "interface_residue_pairs"],
            "radius": args.radius,
            "r2p_k": args.r2p_k,
            "p2r_k": args.p2r_k,
            "loss": "0.5*(L_P/L_P_prior + L_R/L_R_prior)",
            "checkpoint_rule": "both relative NLL ratios < 1 and native interface NLL < token-off interface NLL in both directions",
            "test_read": False,
        },
        "promotion_gate": "C4 requires gate_pass in >=2/3 folds and all four mean conditions",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifests", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiments", nargs="+", choices=("C0", "C1", "C2", "C3", "C4"), default=("C0", "C1", "C2", "C3", "C4"))
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radius", type=float, default=RADIUS)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--r2p-k", type=int, default=8)
    parser.add_argument("--p2r-k", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(func=run_cv)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.device = torch.device(parsed.device)
    print(json.dumps(parsed.func(parsed), indent=2, ensure_ascii=False, default=float), flush=True)
