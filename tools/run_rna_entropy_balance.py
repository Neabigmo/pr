#!/usr/bin/env python3
"""Run the pre-registered E0--E3 RNA entropy/balancing experiment.

The four groups share one frozen ProteinMPNN/NA-MPNN cache, one grouped
development split, one seed, and identical optimizer settings.  E1/E3 use
the entropy-scaled RNA residual; E2/E3 use a training-fold-only geometric
mean sampler over GC fraction, RNA length, and P->R mean degree.  No test or
blind manifest is opened by this runner.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.model import ReciprocalAdapter
from pr_pilot.training.checkpointing import CheckpointManager
from run_adapter_v2_cv import _attach_selected_edges, _batched_loss, _collate_payloads, _load_cache
from run_conditional_adapter_pilot import _forward_payload
from run_scientific_adapter_pilot import _cache_index, _config, _load_fold_payload, _seed_everything
from run_adapter_v2_cv import build_grouped_folds


SEED = 20260917
MANIFESTS = Path(r"I:\PR_PILOT_SCIENTIFIC\20260917\manifest_length_v1")
DEV_CACHE = Path(r"I:\PR_PILOT_SCIENTIFIC\20260918\rna_residual_cap\cache_g2")
OUT = Path(r"I:\PR_PILOT_SCIENTIFIC\20260919\rna_entropy_balance")
RADIUS = 14.979730606
NEIGHBORS = 32
R2P_K = 8
P2R_K = 12
LR = 3e-4
EPOCHS = 60
PATIENCE = 18
BATCH_SIZE = 16

EXPERIMENTS = {
    "E0_original_current": {"rna_gate_mode": "baseline", "balanced": False},
    "E1_original_entropy": {"rna_gate_mode": "entropy_scaled", "balanced": False},
    "E2_balanced_current": {"rna_gate_mode": "baseline", "balanced": True},
    "E3_balanced_entropy": {"rna_gate_mode": "entropy_scaled", "balanced": True},
}

_FOLD_INDEX: dict[str, Path] | None = None


def build_spec(name: str) -> dict:
    item = EXPERIMENTS[name]
    return {
        "geometry": "G2",
        "r2p_k": R2P_K,
        "p2r_k": P2R_K,
        "radius": RADIUS,
        "aggregation": "A0",
        "interaction": "multiplicative",
        "residual": "scalar_gate",
        "separate_edge_encoders": True,
        "modality_projector": True,
        "rna_gate_mode": item["rna_gate_mode"],
        "balanced_sampling": bool(item["balanced"]),
        "sampler_variables": ["gc_fraction", "rna_length", "p2r_mean_degree"] if item["balanced"] else [],
    }


def args_namespace(device: str, resume: bool) -> argparse.Namespace:
    return argparse.Namespace(
        seed=SEED,
        device=torch.device(device),
        radius=RADIUS,
        neighbors=NEIGHBORS,
        r2p_k=R2P_K,
        p2r_k=P2R_K,
        lr=LR,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        resume=bool(resume),
    )


def _json_number(value: object) -> float | int | bool | None:
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return value.item() if hasattr(value, "item") else value
    return None


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        number = _json_number(value)
        if isinstance(number, (int, float)) and not isinstance(number, bool):
            result[key] = float(number)
    return result


def _rank_quartiles(values: np.ndarray) -> np.ndarray:
    """Deterministic equal-count quartiles; ties cannot empty a bin."""
    order = np.argsort(values, kind="stable")
    bins = np.empty(len(values), dtype=np.int64)
    bins[order] = np.minimum(3, (np.arange(len(values), dtype=np.int64) * 4) // max(len(values), 1))
    return bins


def _sample_features(payload: dict) -> dict[str, float]:
    rna = payload["rna_native"].detach().cpu().numpy()
    gc = float(np.mean((rna == 2) | (rna == 3))) if len(rna) else 0.0
    p2r = payload["_selected_edge_index_p2r"]
    degree = float(p2r.shape[1]) / max(float(len(rna)), 1.0)
    return {"gc_fraction": gc, "rna_length": float(len(rna)), "p2r_mean_degree": degree}


def _balanced_weights(train_data: list[dict]) -> tuple[np.ndarray, dict]:
    names = ("gc_fraction", "rna_length", "p2r_mean_degree")
    values = {name: np.asarray([_sample_features(item)[name] for item in train_data], dtype=np.float64) for name in names}
    components: dict[str, np.ndarray] = {}
    report: dict[str, object] = {"clip": [0.5, 2.0], "variables": {}}
    for name in names:
        bins = _rank_quartiles(values[name])
        counts = np.bincount(bins, minlength=4).astype(np.float64)
        inverse = len(values[name]) / (4.0 * np.maximum(counts[bins], 1.0))
        components[name] = inverse / max(float(np.mean(inverse)), 1e-12)
        report["variables"][name] = {
            "quartile_boundaries": np.quantile(values[name], [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
            "bin_counts": counts.astype(int).tolist(),
            "value_min": float(values[name].min()),
            "value_max": float(values[name].max()),
        }
    raw = np.exp(np.mean(np.stack([np.log(components[name]) for name in names], axis=0), axis=0))
    weights = np.clip(raw, 0.5, 2.0)
    report["raw_weight_mean"] = float(raw.mean())
    report["weight_mean"] = float(weights.mean())
    report["weight_min"] = float(weights.min())
    report["weight_max"] = float(weights.max())
    return weights, report


def _training_entropy_tau(train_data: list[dict]) -> float:
    """Return the raw prior-entropy median for one training fold."""
    values: list[torch.Tensor] = []
    for payload in train_data:
        log_probs = payload["rna_base"].detach().float()
        probabilities = log_probs.exp().clamp_min(torch.finfo(log_probs.dtype).tiny)
        values.append((-(probabilities * log_probs).sum(dim=-1)).cpu())
    if not values:
        raise ValueError("cannot compute entropy tau from an empty training fold")
    tau = float(torch.cat(values).median().item())
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"invalid training entropy tau: {tau}")
    return tau


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)


def evaluate_detailed(model: ReciprocalAdapter, data: list[dict], device: torch.device, include_shuffle: bool) -> dict:
    rows: list[dict] = []
    with torch.no_grad():
        for payload in data:
            native = _forward_payload(model, payload, device, RADIUS, NEIGHBORS, token_off=False)
            shuffled = None
            if include_shuffle:
                p_tokens = payload["protein_native"].roll(1) if len(payload["protein_native"]) > 1 else payload["protein_native"]
                r_tokens = payload["rna_native"].roll(1) if len(payload["rna_native"]) > 1 else payload["rna_native"]
                shuffled = _forward_payload(
                    model,
                    payload,
                    device,
                    RADIUS,
                    NEIGHBORS,
                    partner_token_override={"protein": p_tokens, "rna": r_tokens},
                )
            row = {"sample_id": str(payload["sample_id"])}
            for polymer, base_key, label_key, interface_key, active_key, logits_key, shuffle_key in (
                ("protein", "protein_base", "protein_native", "protein_interface", "_selected_protein_active", "protein_logits", "protein_logits"),
                ("rna", "rna_base", "rna_native", "rna_interface", "_selected_rna_active", "rna_logits", "rna_logits"),
            ):
                base = payload[base_key].to(device)
                labels = payload[label_key].to(device)
                interface = payload[interface_key].to(device).bool()
                active = payload[active_key].to(device).bool()
                native_nll = _nll(native[logits_key], labels)
                prior_nll = -base.gather(1, labels[:, None]).squeeze(1)
                for subset, mask in (("all", torch.ones_like(active)), ("active", active), ("interface", interface)):
                    row[f"{polymer}_{subset}_nll"] = float(native_nll[mask].mean().cpu()) if bool(mask.any()) else float("nan")
                    row[f"{polymer}_{subset}_prior_nll"] = float(prior_nll[mask].mean().cpu()) if bool(mask.any()) else float("nan")
                if shuffled is not None:
                    shuffled_nll = _nll(shuffled[shuffle_key], labels)
                    row[f"{polymer}_shuffle_interface_nll"] = float(shuffled_nll[interface].mean().cpu()) if bool(interface.any()) else float("nan")
                row[f"{polymer}_better"] = bool(row[f"{polymer}_interface_nll"] < row[f"{polymer}_interface_prior_nll"])
            rows.append(row)

    def mean_key(key: str) -> float:
        vals = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    metrics: dict[str, object] = {"n_complexes": len(rows)}
    for polymer in ("protein", "rna"):
        for subset in ("all", "active", "interface"):
            metrics[f"{polymer}_{subset}_nll"] = mean_key(f"{polymer}_{subset}_nll")
            metrics[f"{polymer}_{subset}_prior_nll"] = mean_key(f"{polymer}_{subset}_prior_nll")
            metrics[f"{polymer}_{subset}_ratio"] = metrics[f"{polymer}_{subset}_nll"] / max(metrics[f"{polymer}_{subset}_prior_nll"], 1e-8)
        metrics[f"{polymer}_better_fraction"] = float(np.mean([bool(row[f"{polymer}_better"]) for row in rows])) if rows else float("nan")
        if include_shuffle:
            metrics[f"{polymer}_shuffle_interface_nll"] = mean_key(f"{polymer}_shuffle_interface_nll")
            metrics[f"{polymer}_native_minus_shuffle_interface_nll"] = metrics[f"{polymer}_interface_nll"] - metrics[f"{polymer}_shuffle_interface_nll"]

    rna_rows = [row for row in rows if np.isfinite(float(row["rna_interface_prior_nll"]))]
    rna_rows.sort(key=lambda row: float(row["rna_interface_prior_nll"]))
    for index, row in enumerate(rna_rows):
        row["rna_difficulty_quartile"] = min(3, (index * 4) // max(len(rna_rows), 1))
    q_ratios = []
    for q in range(4):
        group = [row for row in rna_rows if row.get("rna_difficulty_quartile") == q]
        a = float(np.mean([row["rna_interface_nll"] for row in group])) if group else float("nan")
        p = float(np.mean([row["rna_interface_prior_nll"] for row in group])) if group else float("nan")
        ratio = a / max(p, 1e-8) if group else float("nan")
        metrics[f"rna_q{q + 1}_n"] = len(group)
        metrics[f"rna_q{q + 1}_ratio"] = ratio
        q_ratios.append(ratio)
    finite_q = [x for x in q_ratios if np.isfinite(x)]
    metrics["rna_stratified_score"] = float(np.mean(finite_q)) if finite_q else float("inf")
    metrics["protein_hard_pass"] = bool(metrics["protein_interface_ratio"] < 1.0)
    metrics["partner_use_pass"] = bool(
        include_shuffle
        and metrics["protein_native_minus_shuffle_interface_nll"] < 0.0
        and metrics["rna_native_minus_shuffle_interface_nll"] < 0.0
    )
    metrics["selection_valid"] = bool(metrics["protein_hard_pass"] and metrics["partner_use_pass"])
    # Primary score is the equal-weight RNA difficulty score.  The tiny
    # secondary term prefers more genuinely improved complexes only on ties;
    # failed hard checks receive a large, explicit penalty.
    penalty = 0.0 if metrics["selection_valid"] else 10.0
    metrics["selection_score"] = float(metrics["rna_stratified_score"] + penalty + 1e-4 * (1.0 - metrics["rna_better_fraction"]))
    return {"metrics": metrics, "complexes": rows}


def _advance_sampling_rng(rng: random.Random, count: int, epochs: int, weights: np.ndarray | None) -> None:
    values = list(range(count))
    for _ in range(max(0, epochs)):
        if weights is None:
            rng.shuffle(values)
        else:
            rng.choices(values, weights=weights.tolist(), k=count)


def _train_fold(spec: dict, fold: dict, by_id: dict[str, Path], args: argparse.Namespace, target: Path) -> dict:
    seed = int(args.seed)
    _seed_everything(seed)
    fold_spec = dict(spec)
    selector_spec = dict(fold_spec)
    if selector_spec["rna_gate_mode"] == "entropy_threshold_scalar":
        # The selector is used only to trim cached geometry before tau is
        # computed from this fold's training residues.
        selector_spec["rna_entropy_tau"] = 1.0
    selector = ReciprocalAdapter(_config(selector_spec))

    def load_payload(sample_id: str) -> dict:
        payload = _load_fold_payload(by_id[sample_id])
        geometry = selector._select_geometry(payload["edge_geometry"])
        payload["edge_geometry"] = geometry.contiguous() if geometry.shape[-1] < payload["edge_geometry"].shape[-1] else geometry
        return payload

    train_data = [load_payload(sample_id) for sample_id in fold["train_sample_ids"]]
    val_data = [load_payload(sample_id) for sample_id in fold["val_sample_ids"]]
    del selector
    if fold_spec["rna_gate_mode"] == "entropy_threshold_scalar":
        fold_spec["rna_entropy_tau"] = _training_entropy_tau(train_data)
    _attach_selected_edges(train_data, RADIUS, NEIGHBORS, R2P_K, P2R_K, True)
    _attach_selected_edges(val_data, RADIUS, NEIGHBORS, R2P_K, P2R_K, True)
    weights = None
    sampler_report = {"enabled": bool(spec["balanced_sampling"])}
    if spec["balanced_sampling"]:
        weights, sampler_report = _balanced_weights(train_data)
        sampler_report["enabled"] = True

    model = ReciprocalAdapter(_config(fold_spec)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)
    manager = CheckpointManager(target, "selection_score")
    start_epoch = 1
    if args.resume and (target / "last.pt").exists():
        start_epoch = manager.restore_last(model, optimizer, map_location=args.device)
    order_rng = random.Random(seed + 1009 * int(fold["fold"]))
    _advance_sampling_rng(order_rng, len(train_data), start_epoch - 1, weights)
    bad = 0
    epochs_run = start_epoch - 1
    for epoch in range(start_epoch, int(args.epochs) + 1):
        started = time.perf_counter()
        model.train()
        if weights is None:
            order = list(range(len(train_data)))
            order_rng.shuffle(order)
        else:
            order = order_rng.choices(list(range(len(train_data))), weights=weights.tolist(), k=len(train_data))
        losses = []
        for start in range(0, len(order), int(args.batch_size)):
            packed = _collate_payloads([train_data[index] for index in order[start : start + int(args.batch_size)]], args.device)
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
        detailed = evaluate_detailed(model, val_data, args.device, include_shuffle=True)
        scalar = _scalar_metrics(detailed["metrics"])
        scalar.update({
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "epoch_seconds": time.perf_counter() - started,
            "train_complexes_per_second": len(train_data) / max(time.perf_counter() - started, 1e-8),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)) if args.device.type == "cuda" else 0.0,
        })
        metadata = {
            "spec": fold_spec,
            "fold": int(fold["fold"]),
            "seed": seed,
            "r2p_k": R2P_K,
            "p2r_k": P2R_K,
            "radius": RADIUS,
            "sampler": sampler_report,
            "selection_protocol": "mean RNA Q1-Q4 ratio; protein ratio<1; native partner NLL<composition-preserving shuffle",
        }
        improved = manager.save_epoch(model, optimizer, None, epoch, scalar, metadata)
        epochs_run = epoch
        bad = 0 if improved else bad + 1
        print(json.dumps({"event": "epoch", "experiment": spec["name"], "fold": fold["fold"], "epoch": epoch, "selection_score": scalar["selection_score"], "rna_q": [scalar[f"rna_q{i}_ratio"] for i in range(1, 5)], "valid": bool(detailed["metrics"]["selection_valid"])}, ensure_ascii=False), flush=True)
        if bad >= int(args.patience):
            break

    best_path = target / "best.pt"
    if not best_path.exists():
        raise RuntimeError(f"no best checkpoint produced for {target}")
    best = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(best["model"])
    model.eval()
    detailed = evaluate_detailed(model, val_data, args.device, include_shuffle=True)
    summary = {
        "experiment": spec["name"],
        "spec": fold_spec,
        "fold": int(fold["fold"]),
        "seed": seed,
        "train_complexes": len(train_data),
        "val_complexes": len(val_data),
        "epochs_run": epochs_run,
        "best_epoch": int(best["epoch"]),
        "selection": detailed["metrics"],
        "validation_complexes": detailed["complexes"],
        "sampler": sampler_report,
        "checkpoint": str(best_path),
        "last_checkpoint": str(target / "last.pt"),
        "test_read": False,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else x), encoding="utf-8")
    return summary


def _init_worker(index: dict[str, str]) -> None:
    global _FOLD_INDEX
    torch.set_num_threads(2)
    _FOLD_INDEX = {sample_id: Path(path) for sample_id, path in index.items()}


def _train_worker(job: tuple[dict, dict, argparse.Namespace, Path]) -> dict:
    if _FOLD_INDEX is None:
        raise RuntimeError("fold cache index not initialized")
    spec, fold, args, target = job
    return _train_fold(spec, fold, _FOLD_INDEX, args, target)


def _numeric_mean(records: list[dict], key: str) -> float:
    values = [float(item["selection"][key]) for item in records if np.isfinite(float(item["selection"].get(key, np.nan)))]
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()
    if args.fold_workers < 1 or args.fold_workers > 2:
        raise ValueError("fold-workers must be 1 or 2; benchmark before increasing")
    OUT.mkdir(parents=True, exist_ok=True)
    folds = build_grouped_folds(MANIFESTS, 3, SEED)
    cache_index = _cache_index(DEV_CACHE)
    expected = {sample_id for fold in folds for sample_id in fold["train_sample_ids"]}
    if set(cache_index) != expected:
        raise ValueError(f"cache IDs do not match grouped development folds: cache={len(cache_index)} expected={len(expected)}")
    protocol = {
        "seed": SEED,
        "manifests": str(MANIFESTS),
        "development_cache": str(DEV_CACHE),
        "test_read": False,
        "experiments": {name: build_spec(name) for name in EXPERIMENTS},
        "optimizer": {"lr": LR, "weight_decay": 1e-3, "batch_size": BATCH_SIZE, "epochs": args.epochs, "patience": args.patience},
        "selection": "S_RNA=mean(Q1,Q2,Q3,Q4); hard protein interface ratio<1; native interface NLL<composition-preserving partner shuffle",
        "balanced_sampler": "training fold only; rank quartiles; per-variable inverse-frequency normalized to mean 1; geometric mean; clip [0.5,2.0]; replacement; same N per epoch",
        "forbidden_inputs": ["old 86 holdout metrics", "new blind metrics"],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    all_groups: dict[str, list[dict]] = {name: [] for name in EXPERIMENTS}
    started_all = time.time()
    for name in EXPERIMENTS:
        spec = build_spec(name)
        spec["name"] = name
        jobs = []
        for fold in folds:
            target = OUT / "cv" / name / f"fold{fold['fold']}"
            if not args.force and (target / "summary.json").exists():
                summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                all_groups[name].append(summary)
                print(json.dumps({"event": "reuse_fold", "experiment": name, "fold": fold["fold"], "score": summary["selection"]["selection_score"]}, ensure_ascii=False), flush=True)
            else:
                jobs.append((spec, fold, args_namespace(args.device, args.resume), target))
        if args.fold_workers == 1:
            _init_worker({key: str(value) for key, value in cache_index.items()})
            for job in jobs:
                summary = _train_worker(job)
                all_groups[name].append(summary)
                print(json.dumps({"event": "fold_complete", "experiment": name, "fold": summary["fold"], "score": summary["selection"]["selection_score"]}, ensure_ascii=False), flush=True)
        elif jobs:
            with ProcessPoolExecutor(max_workers=args.fold_workers, initializer=_init_worker, initargs=({key: str(value) for key, value in cache_index.items()},)) as executor:
                future_map = {executor.submit(_train_worker, job): job[1]["fold"] for job in jobs}
                for future in as_completed(future_map):
                    summary = future.result()
                    all_groups[name].append(summary)
                    print(json.dumps({"event": "fold_complete", "experiment": name, "fold": summary["fold"], "score": summary["selection"]["selection_score"]}, ensure_ascii=False), flush=True)
        all_groups[name].sort(key=lambda item: int(item["fold"]))
        if len(all_groups[name]) != 3:
            raise RuntimeError(f"incomplete {name} CV: {len(all_groups[name])}/3")
        group_summary = {
            "experiment": name,
            "spec": spec,
            "folds": all_groups[name],
            "mean_selection_score": _numeric_mean(all_groups[name], "selection_score"),
            "mean_rna_stratified_score": _numeric_mean(all_groups[name], "rna_stratified_score"),
            "mean_protein_interface_ratio": _numeric_mean(all_groups[name], "protein_interface_ratio"),
            "mean_rna_better_fraction": _numeric_mean(all_groups[name], "rna_better_fraction"),
            "all_folds_hard_pass": all(bool(item["selection"]["protein_hard_pass"]) for item in all_groups[name]),
            "all_folds_partner_use_pass": all(bool(item["selection"]["partner_use_pass"]) for item in all_groups[name]),
            "test_read": False,
        }
        (OUT / "cv" / name / "group_summary.json").write_text(json.dumps(group_summary, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else x), encoding="utf-8")
        print(json.dumps({"event": "group_complete", "experiment": name, "score": group_summary["mean_selection_score"], "hard_pass": group_summary["all_folds_hard_pass"], "partner_pass": group_summary["all_folds_partner_use_pass"]}, ensure_ascii=False), flush=True)

    candidates = []
    for name in EXPERIMENTS:
        group = json.loads((OUT / "cv" / name / "group_summary.json").read_text(encoding="utf-8"))
        if group["all_folds_hard_pass"] and group["all_folds_partner_use_pass"]:
            candidates.append(group)
    if not candidates:
        candidates = [json.loads((OUT / "cv" / name / "group_summary.json").read_text(encoding="utf-8")) for name in EXPERIMENTS]
    selected = min(candidates, key=lambda item: (float(item["mean_selection_score"]), -float(item["mean_rna_better_fraction"])))
    final = {
        "protocol": protocol,
        "groups": {name: json.loads((OUT / "cv" / name / "group_summary.json").read_text(encoding="utf-8")) for name in EXPERIMENTS},
        "selected_experiment": selected["experiment"],
        "selection_basis": "development CV only; equal-weight RNA Q1-Q4 score, then improved-complex fraction; protein and native-vs-shuffle gates",
        "test_read": False,
        "runtime_seconds": time.time() - started_all,
    }
    (OUT / "cv_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else x), encoding="utf-8")
    print(json.dumps({"event": "cv_complete", "selected_experiment": selected["experiment"], "test_read": False, "out": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
