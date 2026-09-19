#!/usr/bin/env python3
"""Run the frozen explicit AA--base selection-matrix protocol.

This runner is intentionally separate from the historical architecture-search
runner.  It trains exactly A0/A1/A2 on the same 860-complex development split,
uses one seed, never opens a test cache, and retains only best.pt/last.pt,
metrics.jsonl, and summary artifacts per fold.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import random
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.explicit_matrix import ExplicitMatrixConfig, ExplicitSelectionAdapter
from pr_pilot.adapter_pilot.model import AdapterConfig, ReciprocalAdapter
from pr_pilot.training.checkpointing import CheckpointManager
from run_adapter_v2_cv import (
    _attach_selected_edges,
    _batched_loss,
    _collate_payloads,
    _forward_payload,
    build_grouped_folds,
)

SINGLE_SEED = 20260919
RADIUS = 14.979730606
DEFAULT_MANIFESTS = Path(r"I:\PR_PILOT_SCIENTIFIC\20260919\resplit_e0\manifest")
DEFAULT_CACHE = Path(r"I:\PR_PILOT_SCIENTIFIC\20260919\cache\g2_structure_only")
DEFAULT_OUT = Path(r"I:\PR_PILOT_SCIENTIFIC\20260919\reports\explicit_selection_matrix")
_WORKER_INDEX: dict[str, Path] | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cache_index(cache_root: Path) -> dict[str, Path]:
    files = sorted((cache_root / "train").glob("*.pt")) + sorted((cache_root / "val").glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no development cache files in {cache_root}")
    result: dict[str, Path] = {}
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        sample_id = str(payload["sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate cache sample_id: {sample_id}")
        result[sample_id] = path
    return result


def _load_payload(source: Path | dict) -> dict:
    if isinstance(source, Path):
        return torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    return dict(source)


def _fold_payloads(by_id: dict[str, Path], sample_ids: list[str]) -> list[dict]:
    result = []
    for sample_id in sample_ids:
        payload = _load_payload(by_id[sample_id])
        # The cache is G2 already, but materialize a compact tensor so a fold
        # never retains an unnecessary mmap view of a wider geometry cache.
        payload["edge_geometry"] = payload["edge_geometry"][:, :114].contiguous()
        result.append(payload)
    return result


def _a0_config() -> AdapterConfig:
    """Exact locked E0 configuration used as the retrained baseline."""
    return AdapterConfig(
        geometry="G2", aggregation="A0", interaction="multiplicative", residual="scalar_gate",
        hidden_dim=128, hidden_projection_dim=64, edge_dim=64, token_dim=64,
        message_dim=128, layers=1, dropout=0.1, rbf_bins=16,
        separate_edge_encoders=True, sequence_independent_attention=True,
        partner_centered_residual=False, modality_projector=True,
        conservative_gate=True, gate_init=0.1,
    )


def _make_model(variant: str) -> torch.nn.Module:
    if variant == "A0":
        return ReciprocalAdapter(_a0_config())
    geometry = "G0" if variant == "B3" else "G2"
    return ExplicitSelectionAdapter(ExplicitMatrixConfig(variant=variant, geometry=geometry))


def _forward(model: torch.nn.Module, payload: dict, device: torch.device, token_override: dict[str, torch.Tensor] | None = None, token_off: bool = False) -> dict[str, torch.Tensor]:
    if isinstance(model, ReciprocalAdapter):
        return _forward_payload(
            model, payload, device, RADIUS, 32, token_off=token_off,
            partner_token_override=token_override,
        )
    p_tokens = token_override.get("protein", payload["protein_native"]) if token_override else payload["protein_native"]
    r_tokens = token_override.get("rna", payload["rna_native"]) if token_override else payload["rna_native"]
    return model(
        payload["protein_base"].to(device), payload["rna_base"].to(device),
        payload["protein_hidden"].to(device), payload["rna_hidden"].to(device),
        p_tokens.to(device), r_tokens.to(device),
        payload["_selected_edge_index_r2p"].to(device), payload["_selected_geometry_r2p"].to(device),
        payload["_selected_edge_index_p2r"].to(device), payload["_selected_geometry_p2r"].to(device),
        token_off=token_off,
    )


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)


def _shuffle_tokens(payload: dict, seed: int, replicate: int) -> dict[str, torch.Tensor]:
    # The frozen cache does not carry residue-to-chain offsets.  A global
    # composition-preserving permutation is therefore used consistently for
    # both directions; target labels, coordinates, masks, and selected edges
    # remain unchanged. The limitation is recorded in the protocol summary.
    sample_hash = int.from_bytes(hashlib.sha256(str(payload["sample_id"]).encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(int(seed) + 100003 * int(replicate) + sample_hash)
    p_order = torch.from_numpy(rng.permutation(len(payload["protein_native"]))).long()
    r_order = torch.from_numpy(rng.permutation(len(payload["rna_native"]))).long()
    return {"protein": payload["protein_native"][p_order], "rna": payload["rna_native"][r_order]}


def _finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def evaluate_detailed(
    model: torch.nn.Module,
    data: list[dict],
    device: torch.device,
    seed: int,
    shuffle_repeats: int,
) -> dict:
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for payload in data:
            native = _forward(model, payload, device)
            p_labels = payload["protein_native"].to(device)
            r_labels = payload["rna_native"].to(device)
            p_interface = payload["protein_interface"].to(device).bool()
            r_interface = payload["rna_interface"].to(device).bool()
            p_active = payload["_selected_protein_active"].to(device).bool()
            r_active = payload["_selected_rna_active"].to(device).bool()
            p_base = payload["protein_base"].to(device)
            r_base = payload["rna_base"].to(device)
            p_nll = _nll(native["protein_logits"], p_labels)
            r_nll = _nll(native["rna_logits"], r_labels)
            p_prior = -p_base.gather(1, p_labels[:, None]).squeeze(1)
            r_prior = -r_base.gather(1, r_labels[:, None]).squeeze(1)
            row = {"sample_id": str(payload["sample_id"])}
            for name, nll, prior, active, interface in (
                ("protein", p_nll, p_prior, p_active, p_interface),
                ("rna", r_nll, r_prior, r_active, r_interface),
            ):
                for subset, mask in (("all", torch.ones_like(active)), ("active", active), ("interface", interface)):
                    row[f"{name}_{subset}_nll"] = float(nll[mask].mean().cpu()) if bool(mask.any()) else float("nan")
                    row[f"{name}_{subset}_prior_nll"] = float(prior[mask].mean().cpu()) if bool(mask.any()) else float("nan")
                row[f"{name}_better"] = bool(row[f"{name}_interface_nll"] < row[f"{name}_interface_prior_nll"])
                row[f"{name}_interface_recovery"] = float((native[f"{name}_logits"].argmax(-1)[interface] == (p_labels if name == "protein" else r_labels)[interface]).float().mean().cpu()) if bool(interface.any()) else float("nan")

            p_shuffle: list[float] = []
            r_shuffle: list[float] = []
            for replicate in range(int(shuffle_repeats)):
                shuffled = _forward(model, payload, device, _shuffle_tokens(payload, seed, replicate))
                p_shuffle.append(float(_nll(shuffled["protein_logits"], p_labels)[p_interface].mean().cpu()) if bool(p_interface.any()) else float("nan"))
                r_shuffle.append(float(_nll(shuffled["rna_logits"], r_labels)[r_interface].mean().cpu()) if bool(r_interface.any()) else float("nan"))
            row["protein_shuffle_interface_nll"] = _finite_mean(p_shuffle)
            row["rna_shuffle_interface_nll"] = _finite_mean(r_shuffle)
            row["protein_native_minus_shuffle_interface_nll"] = row["protein_interface_nll"] - row["protein_shuffle_interface_nll"]
            row["rna_native_minus_shuffle_interface_nll"] = row["rna_interface_nll"] - row["rna_shuffle_interface_nll"]
            rows.append(row)

    metrics: dict[str, float | bool | int] = {"n_complexes": len(rows)}
    for name in ("protein", "rna"):
        for subset in ("all", "active", "interface"):
            metrics[f"{name}_{subset}_nll"] = _finite_mean([row[f"{name}_{subset}_nll"] for row in rows])
            metrics[f"{name}_{subset}_prior_nll"] = _finite_mean([row[f"{name}_{subset}_prior_nll"] for row in rows])
            metrics[f"{name}_{subset}_ratio"] = metrics[f"{name}_{subset}_nll"] / max(metrics[f"{name}_{subset}_prior_nll"], 1e-8)
        metrics[f"{name}_better_fraction"] = float(np.mean([bool(row[f"{name}_better"]) for row in rows])) if rows else float("nan")
        metrics[f"{name}_shuffle_interface_nll"] = _finite_mean([row[f"{name}_shuffle_interface_nll"] for row in rows])
        metrics[f"{name}_native_minus_shuffle_interface_nll"] = metrics[f"{name}_interface_nll"] - metrics[f"{name}_shuffle_interface_nll"]

    ranked = sorted(rows, key=lambda row: float(row["rna_interface_prior_nll"]))
    for index, row in enumerate(ranked):
        row["rna_difficulty_quartile"] = min(3, (index * 4) // max(len(ranked), 1))
    q_ratios: list[float] = []
    for q in range(4):
        group = [row for row in ranked if row["rna_difficulty_quartile"] == q]
        q_nll = _finite_mean([row["rna_interface_nll"] for row in group])
        q_prior = _finite_mean([row["rna_interface_prior_nll"] for row in group])
        ratio = q_nll / max(q_prior, 1e-8) if group else float("nan")
        metrics[f"rna_q{q + 1}_n"] = len(group)
        metrics[f"rna_q{q + 1}_ratio"] = ratio
        q_ratios.append(ratio)
    metrics["rna_stratified_score"] = _finite_mean(q_ratios)
    metrics["protein_hard_pass"] = bool(metrics["protein_interface_ratio"] < 1.0)
    metrics["partner_use_pass"] = bool(
        metrics["protein_native_minus_shuffle_interface_nll"] < 0.0
        and metrics["rna_native_minus_shuffle_interface_nll"] < 0.0
    )
    metrics["selection_valid"] = bool(metrics["protein_hard_pass"] and metrics["partner_use_pass"])
    metrics["selection_score"] = float(
        metrics["rna_stratified_score"]
        + (0.0 if metrics["selection_valid"] else 10.0)
        + 1e-4 * (1.0 - float(metrics["rna_better_fraction"]))
    )
    return {"metrics": metrics, "complexes": rows}


def _paired_bootstrap(rows: list[dict], seed: int, repeats: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    n = len(rows)
    if n == 0:
        return {"n": 0, "repeats": repeats}
    p_delta = np.asarray([r["protein_interface_nll"] - r["protein_interface_prior_nll"] for r in rows], dtype=np.float64)
    r_delta = np.asarray([r["rna_interface_nll"] - r["rna_interface_prior_nll"] for r in rows], dtype=np.float64)
    p_shuffle = np.asarray([r["protein_native_minus_shuffle_interface_nll"] for r in rows], dtype=np.float64)
    r_shuffle = np.asarray([r["rna_native_minus_shuffle_interface_nll"] for r in rows], dtype=np.float64)
    index = rng.integers(0, n, size=(int(repeats), n))
    def ci(values: np.ndarray) -> list[float]:
        boot = values[index].mean(axis=1)
        return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return {
        "n": n,
        "repeats": int(repeats),
        "protein_delta_nll_ci95": ci(p_delta),
        "rna_delta_nll_ci95": ci(r_delta),
        "protein_native_minus_shuffle_ci95": ci(p_shuffle),
        "rna_native_minus_shuffle_ci95": ci(r_shuffle),
    }


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            result[key] = float(value)
    return result


def _advance_order_rng(rng: random.Random, count: int, completed_epochs: int) -> None:
    order = list(range(count))
    for _ in range(max(0, completed_epochs)):
        rng.shuffle(order)


def _train_fold(variant: str, fold: dict, by_id: dict[str, Path], args: argparse.Namespace, target: Path) -> dict:
    seed = int(args.seed)
    _seed_everything(seed)
    train_data = _fold_payloads(by_id, fold["train_sample_ids"])
    val_data = _fold_payloads(by_id, fold["val_sample_ids"])
    _attach_selected_edges(train_data, RADIUS, 32, 8, 12, True)
    _attach_selected_edges(val_data, RADIUS, 32, 8, 12, True)
    model = _make_model(variant).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)
    manager = CheckpointManager(target, "selection_score")
    start_epoch = 1
    if args.resume and (target / "last.pt").exists():
        start_epoch = manager.restore_last(model, optimizer, map_location=args.device)
    order_rng = random.Random(seed + 1009 * int(fold["fold"]))
    _advance_order_rng(order_rng, len(train_data), start_epoch - 1)
    best_score = float("inf")
    if (target / "best.pt").exists():
        best_score = float(torch.load(target / "best.pt", map_location="cpu", weights_only=False)["metrics"]["selection_score"])
    bad = 0
    epochs_run = max(0, start_epoch - 1)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        started = time.perf_counter()
        model.train()
        order = list(range(len(train_data)))
        order_rng.shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), int(args.batch_size)):
            payloads = [train_data[index] for index in order[start : start + int(args.batch_size)]]
            packed = _collate_payloads(payloads, args.device)
            optimizer.zero_grad(set_to_none=True)
            if variant == "A0":
                out = model(
                    packed["protein_base"], packed["rna_base"], packed["protein_hidden"], packed["rna_hidden"],
                    packed["protein_native"], packed["rna_native"], packed["edge_index_r2p"], packed["edge_geometry_r2p"],
                    packed["edge_index_p2r"], packed["edge_geometry_p2r"],
                )
            else:
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
        validation = evaluate_detailed(model, val_data, args.device, seed + 7000 * int(fold["fold"]), int(args.selection_shuffle_repeats))
        metrics = _scalar_metrics(validation["metrics"])
        metrics.update({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "epoch_seconds": time.perf_counter() - started,
            "train_complexes_per_second": len(train_data) / max(time.perf_counter() - started, 1e-8),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)) if args.device.type == "cuda" else 0.0,
        })
        manager.save_epoch(model, optimizer, None, epoch, metrics, {
            "variant": variant, "fold": int(fold["fold"]), "seed": seed,
            "test_read": False, "hidden_contract": "sequence_free_encoder_side",
        })
        epochs_run = epoch
        if metrics["selection_score"] < best_score - 1e-7:
            best_score = metrics["selection_score"]
            bad = 0
        else:
            bad += 1
        if bad >= int(args.patience):
            break

    best_path = target / "best.pt"
    if not best_path.exists():
        raise RuntimeError(f"no best checkpoint produced for {target}")
    best = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(best["model"])
    model.eval()
    validation = evaluate_detailed(model, val_data, args.device, seed + 7000 * int(fold["fold"]), int(args.shuffle_repeats))
    rows_path = target / "validation_per_complex.jsonl"
    rows_path.write_text("\n".join(json.dumps(row, ensure_ascii=False, default=float) for row in validation["complexes"]) + "\n", encoding="utf-8")
    summary = {
        "variant": variant, "fold": int(fold["fold"]), "seed": seed,
        "train_complexes": len(train_data), "val_complexes": len(val_data),
        "epochs_run": epochs_run, "best_epoch": int(best["epoch"]),
        "best_selection_score": float(best_score), "best_validation": validation["metrics"],
        "bootstrap": _paired_bootstrap(validation["complexes"], seed + 9000 + int(fold["fold"])),
        "checkpoint": str(best_path), "last_checkpoint": str(target / "last.pt"),
        "validation_per_complex": str(rows_path), "test_read": False,
        "protocol": {"geometry": "G2", "radius": RADIUS, "r2p_k": 8, "p2r_k": 12, "mean_selected_neighbors": True, "shuffle_repeats": int(args.shuffle_repeats)},
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return summary


def _init_worker(cache_root: Path) -> None:
    global _WORKER_INDEX
    torch.set_num_threads(2)
    _WORKER_INDEX = _cache_index(cache_root)


def _worker(job: tuple[str, dict, argparse.Namespace, Path]) -> dict:
    if _WORKER_INDEX is None:
        raise RuntimeError("cache index was not initialized")
    variant, fold, args, target = job
    return _train_fold(variant, fold, _WORKER_INDEX, args, target)


def _aggregate_variant(summaries: list[dict]) -> dict:
    ordered = sorted(summaries, key=lambda item: int(item["fold"]))
    keys = sorted(key for key, value in ordered[0]["best_validation"].items() if isinstance(value, (int, float)) and np.isfinite(value))
    mean_metrics = {key: float(np.mean([float(item["best_validation"][key]) for item in ordered])) for key in keys}
    return {"folds": len(ordered), "mean_metrics": mean_metrics, "fold_summaries": ordered}


def run_cv(args: argparse.Namespace) -> dict:
    if "test" in str(args.manifests).lower() or "test" in str(args.cache_root).lower():
        raise ValueError("explicit matrix CV refuses test manifests or test caches")
    by_id = _cache_index(Path(args.cache_root))
    folds = build_grouped_folds(Path(args.manifests), 3, int(args.seed))
    expected = set(sample_id for fold in folds for sample_id in fold["train_sample_ids"])
    if set(by_id) != expected:
        raise ValueError(f"cache/manifest mismatch: cache={len(by_id)} expected={len(expected)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "explicit_dynamic_selection_matrix",
        "seed": int(args.seed), "development_complexes": len(expected), "folds": 3,
        "variants": list(args.experiments),
        "A0": "locked E0 multiplicative + separate directional edge encoders + modality projectors + learned scalar gates",
        "A1": "M_ij=C, zero-initialized global 20x4 matrix + learned scalar gates",
        "A2": "M_ij=C+DeltaC_ij, shared G2 edge encoder and 192-128-80 MLP; output zero initialized",
        "B1": "M_ij=DeltaC_ij; global C removed, otherwise A2",
        "B2": "M_ij=C+DeltaC_ij; structural hidden removed, G2 edge-only dynamic input",
        "B3": "M_ij=C+DeltaC_ij; G2 replaced by G0 distance/RBF geometry",
        "hidden_contract": "sequence_free_encoder_side; no decoder or teacher-forced target hidden",
        "geometry": "G2", "radius": RADIUS, "r2p_k": 8, "p2r_k": 12,
        "geometry_by_variant": {variant: ("G0" if variant == "B3" else "G2") for variant in args.experiments},
        "loss": "0.5*(LP/LPprior + LR/LRprior)",
        "selection": "RNA equal-weight Q1-Q4 ratio; hard RP<1 and native<20-shuffle in both directions",
        "shuffle": "20 fixed composition-preserving global permutations; target labels/masks/geometry/edges fixed",
        "test_read": False, "old_215_status": "diagnostic_only_never_used_for_selection",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    executor = ProcessPoolExecutor(max_workers=int(args.fold_workers), initializer=_init_worker, initargs=(Path(args.cache_root),)) if int(args.fold_workers) > 1 else None
    all_results: dict[str, dict] = {}
    try:
        for variant in args.experiments:
            summaries: list[dict] = []
            jobs = []
            for fold in folds:
                target = out / variant / f"fold{fold['fold']}"
                if (target / "summary.json").exists() and not args.force:
                    summaries.append(json.loads((target / "summary.json").read_text(encoding="utf-8")))
                else:
                    jobs.append((variant, fold, args, target))
            if executor is None or len(jobs) <= 1:
                for job in jobs:
                    summaries.append(_train_fold(job[0], job[1], by_id, args, job[3]))
            else:
                futures = [executor.submit(_worker, job) for job in jobs]
                for future in as_completed(futures):
                    summaries.append(future.result())
            all_results[variant] = _aggregate_variant(summaries)
            (out / variant / "cv_summary.json").write_text(json.dumps(all_results[variant], indent=2, ensure_ascii=False, default=float), encoding="utf-8")
            for summary in sorted(summaries, key=lambda item: int(item["fold"])):
                print(json.dumps({"variant": variant, "fold": summary["fold"], "score": summary["best_selection_score"], "metrics": {key: summary["best_validation"].get(key) for key in ("protein_interface_ratio", "rna_interface_ratio", "rna_stratified_score", "selection_valid")}}, ensure_ascii=False), flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    comparisons = {}
    for left, right in (("A1", "A0"), ("A2", "A1"), ("B1", "A2"), ("B2", "A2"), ("B3", "A2")):
        if left in all_results and right in all_results:
            comparisons[f"{left}_vs_{right}"] = {
                "protein_interface_ratio_delta": all_results[left]["mean_metrics"]["protein_interface_ratio"] - all_results[right]["mean_metrics"]["protein_interface_ratio"],
                "rna_interface_ratio_delta": all_results[left]["mean_metrics"]["rna_interface_ratio"] - all_results[right]["mean_metrics"]["rna_interface_ratio"],
                "rna_stratified_score_delta": all_results[left]["mean_metrics"]["rna_stratified_score"] - all_results[right]["mean_metrics"]["rna_stratified_score"],
                "lower_is_better": True,
            }
    result = {"protocol": protocol, "experiments": all_results, "comparisons": comparisons, "test_read": False}
    (out / "cv_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"completed": list(all_results), "comparisons": comparisons, "test_read": False}, ensure_ascii=False, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--experiments", nargs="+", choices=("A0", "A1", "A2", "B1", "B2", "B3"), default=("A0", "A1", "A2"))
    parser.add_argument("--seed", type=int, default=SINGLE_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fold-workers", type=int, default=1)
    parser.add_argument("--selection-shuffle-repeats", type=int, default=20)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.device = torch.device(parsed.device)
    print(json.dumps(run_cv(parsed), indent=2, ensure_ascii=False, default=float), flush=True)
