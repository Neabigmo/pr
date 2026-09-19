#!/usr/bin/env python3
"""Single-seed scientific Adapter pilot orchestration.

This runner deliberately shares the existing cache, grouped-fold and metric
implementations, but uses a new protocol layer: one seed, direction-specific
K, ratio-first selection, late partner-shuffle promotion, and bounded
checkpoint retention.  It never reads a test cache during ``preflight`` or
``search``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import random
from statistics import median
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.model import AdapterConfig, ReciprocalAdapter
from pr_pilot.adapter_pilot.scientific import (  # noqa: E402
    AGGREGATIONS,
    GEOMETRY_MODES,
    INTERACTIONS,
    K_CONFIGS,
    RESIDUALS,
    SINGLE_SEED,
    ScientificProtocol,
    choose_candidate,
    validate_length_bounds,
    write_protocol,
)
from pr_pilot.training.checkpointing import CheckpointManager  # noqa: E402
from run_adapter_v2_cv import (  # noqa: E402
    RADIUS,
    _aggregate_cv,
    _attach_selected_edges,
    _batched_loss,
    _collate_payloads,
    _forward_payload,
    _load_cache,
    build_grouped_folds,
    evaluate_dataset,
)

DEFAULT_CACHE = Path(r"F:\111临时\PR PILOT\pilot_conditional_adapter_20260916\cache\noise0p0")
DEFAULT_MANIFESTS = Path(r"F:\111临时\PR PILOT\remote_return_20260911\manifests\round_20260905_exception_v2")
DEFAULT_OUT = Path(r"I:\PR_PILOT_SCIENTIFIC\20260917\reports")
RADIUS_OPTIONS = (13.308568573, 14.357456360, 14.979730606)
_FOLD_WORKER_BY_ID: dict[str, Path] | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config(spec: dict) -> AdapterConfig:
    interaction = str(spec.get("interaction", "concat"))
    residual = str(spec.get("residual", "direct"))
    centered = interaction == "centered" or residual == "partner_centered"
    return AdapterConfig(
        geometry=str(spec.get("geometry", "G2")),
        aggregation=str(spec.get("aggregation", "A2")),
        interaction=interaction,
        residual=residual,
        hidden_dim=128,
        hidden_projection_dim=64,
        edge_dim=64,
        token_dim=64,
        message_dim=128,
        layers=1,
        dropout=0.1,
        rbf_bins=16,
        separate_edge_encoders=bool(spec.get("separate_edge_encoders", False)),
        sequence_independent_attention=True,
        partner_centered_residual=centered,
        modality_projector=bool(spec.get("modality_projector", True)),
        conservative_gate=residual == "scalar_gate",
        gate_init=0.1,
        rna_gate_mode=str(spec.get("rna_gate_mode", "baseline")),
        rna_entropy_tau=(None if spec.get("rna_entropy_tau") is None else float(spec["rna_entropy_tau"])),
    )


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            result[key] = float(value)
    return result


def _score(metrics: dict) -> float:
    return max(float(metrics["protein_interface_ratio"]), float(metrics["rna_interface_ratio"]))


def _cache_index(cache_root: Path) -> dict[str, Path]:
    """Index immutable cache files without retaining their tensor payloads.

    The scientific search trains one fold at a time when running in the
    reliable single-worker mode.  Keeping all development payloads resident
    while also materializing selected edges can push the Windows pagefile
    above the size of the cache itself.  An ID-to-file index preserves the
    cache contract and lets each fold own only its active payloads.
    """
    files = sorted((cache_root / "train").glob("*.pt")) + sorted((cache_root / "val").glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no development caches in {cache_root}")
    index: dict[str, Path] = {}
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        sample_id = str(payload["sample_id"])
        if sample_id in index:
            raise ValueError(f"duplicate cache sample_id: {sample_id}")
        index[sample_id] = path
    return index


def _load_fold_payload(source: Path | dict) -> dict:
    """Load a fold payload from an indexed file or accept legacy in-memory data."""
    if isinstance(source, Path):
        return torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    return dict(source)


def _train_fold(
    spec: dict,
    fold: dict,
    by_id: dict[str, dict],
    args: argparse.Namespace,
    target: Path,
) -> dict:
    seed = int(args.seed)
    _seed_everything(seed)
    # Select the model's feature columns before materializing directional
    # edges. Keep the shared source payload immutable across fold jobs.
    geometry_selector = ReciprocalAdapter(_config(spec))
    def fold_payload(sample_id, _selector=geometry_selector):
        payload = _load_fold_payload(by_id[sample_id])
        full_geometry = payload["edge_geometry"]
        selected_geometry = _selector._select_geometry(full_geometry)
        # A narrow view keeps the complete G3 mmap alive.  Materialize only
        # reduced geometries (G0/G1/G2) so a fold does not retain 1049-column
        # edge tensors after selecting its protocol geometry.
        if selected_geometry.shape[-1] < full_geometry.shape[-1]:
            selected_geometry = selected_geometry.contiguous()
        payload["edge_geometry"] = selected_geometry
        return payload
    train_data = [fold_payload(sample_id) for sample_id in fold["train_sample_ids"]]
    val_data = [fold_payload(sample_id) for sample_id in fold["val_sample_ids"]]
    del geometry_selector
    _seed_everything(seed)
    r2p_k = int(spec.get("r2p_k", args.r2p_k))
    p2r_k = int(spec.get("p2r_k", args.p2r_k))
    radius = float(spec.get("radius", args.radius))
    _attach_selected_edges(train_data, radius, args.neighbors, r2p_k, p2r_k, True)
    _attach_selected_edges(val_data, radius, args.neighbors, r2p_k, p2r_k, True)
    model = ReciprocalAdapter(_config(spec)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)
    manager = CheckpointManager(target, "selection_score")
    start_epoch = 1
    if args.resume and (target / "last.pt").exists():
        start_epoch = manager.restore_last(model, optimizer, map_location=args.device)
    order_rng = random.Random(seed + 1009 * int(fold["fold"]))
    bad = 0
    best_score = float("inf")
    if (target / "best.pt").exists():
        best_payload = torch.load(target / "best.pt", map_location="cpu", weights_only=False)
        best_score = float(best_payload["metrics"]["selection_score"])
    epochs_run = max(0, start_epoch - 1)
    _advance_order_rng(order_rng, len(train_data), epochs_run)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        started = time.perf_counter()
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
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
        validation = evaluate_dataset(model, val_data, args.device, radius, args.neighbors, include_permutation=False)
        scalar = _scalar_metrics(validation)
        scalar["train_loss"] = float(np.mean(losses)) if losses else float("nan")
        scalar["selection_score"] = _score(validation)
        scalar["epoch_seconds"] = time.perf_counter() - started
        scalar["train_complexes_per_second"] = len(train_data) / max(scalar["epoch_seconds"], 1e-8)
        scalar["peak_gpu_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2))
            if args.device.type == "cuda" else 0.0
        )
        metadata = {"spec": spec, "fold": int(fold["fold"]), "seed": seed, "r2p_k": r2p_k, "p2r_k": p2r_k, "radius": radius}
        improved = manager.save_epoch(model, optimizer, None, epoch, scalar, metadata)
        epochs_run = epoch
        if improved:
            best_score = scalar["selection_score"]
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
    validation = evaluate_dataset(model, val_data, args.device, radius, args.neighbors, include_permutation=True)
    scalar = _scalar_metrics(validation)
    scalar["specificity_mean"] = float(np.nanmean([
        scalar.get("protein_native_minus_permutation_interface_nll", np.nan),
        scalar.get("rna_native_minus_permutation_interface_nll", np.nan),
    ]))
    summary = {
        "spec": spec,
        "fold": int(fold["fold"]),
        "seed": seed,
        "train_complexes": len(train_data),
        "val_complexes": len(val_data),
        "epochs_run": epochs_run,
        "best_epoch": int(best["epoch"]),
        "selection_score": best_score,
        "validation": scalar,
        "checkpoint": str(best_path),
        "last_checkpoint": str(target / "last.pt"),
        "test_read": False,
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return summary


def _advance_order_rng(order_rng: random.Random, sample_count: int, completed_epochs: int) -> None:
    """Recreate the per-fold shuffle stream before resuming a checkpoint."""
    scratch = list(range(sample_count))
    for _ in range(max(0, completed_epochs)):
        order_rng.shuffle(scratch)


def _init_fold_worker(cache_root: Path) -> None:
    """Index the development cache once per persistent worker."""
    global _FOLD_WORKER_BY_ID
    torch.set_num_threads(2)
    _FOLD_WORKER_BY_ID = _cache_index(Path(cache_root))


def _train_fold_worker(job: tuple[dict, dict, argparse.Namespace, Path]) -> dict:
    """Train one independent fold using the worker's persistent cache."""
    if _FOLD_WORKER_BY_ID is None:
        raise RuntimeError("fold worker cache was not initialized")
    spec, fold, args, target = job
    return _train_fold(spec, fold, _FOLD_WORKER_BY_ID, args, target)


def _stage_specs(stage: str, base: dict) -> list[dict]:
    if stage == "geometry":
        return [{**base, "geometry": value} for value in GEOMETRY_MODES]
    if stage == "k":
        return [{**base, "r2p_k": r2p, "p2r_k": p2r} for r2p, p2r in K_CONFIGS]
    if stage == "radius":
        return [{**base, "radius": value} for value in RADIUS_OPTIONS]
    if stage == "aggregation":
        return [{**base, "aggregation": value} for value in AGGREGATIONS]
    if stage == "interaction":
        return [{**base, "interaction": value} for value in INTERACTIONS]
    if stage == "residual":
        return [{**base, "residual": value} for value in RESIDUALS]
    if stage == "edge_encoder":
        return [{**base, "separate_edge_encoders": value} for value in (False, True)]
    raise ValueError(f"unknown search stage {stage}")


def _spec_name(spec: dict) -> str:
    fields = [spec.get("geometry"), spec.get("r2p_k"), spec.get("p2r_k"), spec.get("radius"), spec.get("aggregation"), spec.get("interaction"), spec.get("residual"), "separate" if spec.get("separate_edge_encoders") else "shared"]
    return "_".join(str(value).replace(".", "p") for value in fields)


def run_preflight(args: argparse.Namespace) -> dict:
    manifests = Path(args.manifests)
    protocol = ScientificProtocol(seed=int(args.seed))
    reports = {}
    for split in ("train", "val"):
        frame = pd.read_csv(manifests / f"complex_{split}.tsv", sep="\t")
        reports[split] = validate_length_bounds(frame, protocol)
        if reports[split]["protein_out_of_range"] or reports[split]["rna_out_of_range"]:
            raise ValueError(f"{split} manifest violates the scientific length protocol: {reports[split]}")
    folds = build_grouped_folds(manifests, 3, int(args.seed))
    result = {
        "protocol": protocol.as_dict(),
        "manifests": str(manifests),
        "development_length_audit": reports,
        "folds": [{"fold": f["fold"], "train": f["n_train"], "val": f["n_val"]} for f in folds],
        "test_read": False,
        "cache_root": str(args.cache_root),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_protocol(out / "preflight.json", protocol, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def run_search(args: argparse.Namespace) -> dict:
    if "test" in str(args.cache_root).lower() or "test" in str(args.manifests).lower():
        raise ValueError("scientific search refuses test caches/manifests")
    by_id = _cache_index(Path(args.cache_root))
    folds = build_grouped_folds(Path(args.manifests), 3, int(args.seed))
    expected = set(sample_id for fold in folds for sample_id in fold["train_sample_ids"])
    if set(by_id) != expected:
        raise ValueError("cache IDs do not exactly match development grouped folds")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_protocol(out / "protocol.json", ScientificProtocol(seed=int(args.seed)), {"test_read": False})
    base = {
        "geometry": "G2", "r2p_k": int(args.r2p_k), "p2r_k": int(args.p2r_k), "radius": float(args.radius),
        "aggregation": "A2", "interaction": "concat", "residual": "partner_centered",
        "separate_edge_encoders": False, "modality_projector": True,
    }
    stage_order = [args.stage] if args.stage != "all" else ["geometry", "k", "radius", "aggregation", "interaction", "residual", "edge_encoder"]
    selected = base
    reports = {}
    executor = (
        ProcessPoolExecutor(
            max_workers=int(args.fold_workers),
            initializer=_init_fold_worker,
            initargs=(Path(args.cache_root),),
        )
        if int(args.fold_workers) > 1 else None
    )
    for stage in stage_order:
        specs = _stage_specs(stage, selected)
        candidates = []
        for spec in specs:
            name = _spec_name(spec)
            fold_summaries = []
            pending = []
            for fold in folds:
                target = out / "search" / stage / name / f"fold{fold['fold']}"
                if (target / "summary.json").exists() and not args.force:
                    fold_summaries.append(json.loads((target / "summary.json").read_text(encoding="utf-8")))
                else:
                    pending.append((spec, fold, args, target))
            if int(args.fold_workers) <= 1 or len(pending) <= 1:
                for job in pending:
                    summary = _train_fold(job[0], job[1], by_id, args, job[3])
                    fold_summaries.append(summary)
            else:
                futures = [executor.submit(_train_fold_worker, job) for job in pending]
                for future in as_completed(futures):
                    fold_summaries.append(future.result())
            for summary in sorted(fold_summaries, key=lambda item: int(item["fold"])):
                print(json.dumps({"stage": stage, "candidate": name, "fold": summary["fold"], "score": summary["selection_score"]}, ensure_ascii=False), flush=True)
            metrics = {
                "protein_interface_ratio": float(np.mean([item["validation"]["protein_interface_ratio"] for item in fold_summaries])),
                "rna_interface_ratio": float(np.mean([item["validation"]["rna_interface_ratio"] for item in fold_summaries])),
                "specificity_mean": float(np.nanmean([item["validation"]["specificity_mean"] for item in fold_summaries])),
            }
            candidates.append({"name": name, "spec": spec, "metrics": metrics, "folds": fold_summaries})
        stage_choice = choose_candidate(candidates)
        selected = dict(stage_choice["spec"])
        reports[stage] = {"selected": stage_choice, "candidates": candidates}
        (out / "search" / stage / "summary.json").parent.mkdir(parents=True, exist_ok=True)
        (out / "search" / stage / "summary.json").write_text(json.dumps(reports[stage], indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    if executor is not None:
        executor.shutdown(wait=True)
    result = {"selected": selected, "stages": reports, "seed": int(args.seed), "test_read": False}
    (out / "search_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"selected": selected, "test_read": False}, indent=2, ensure_ascii=False), flush=True)
    return result


def _train_full(spec: dict, data: list[dict], args: argparse.Namespace, target: Path, epochs: int) -> dict:
    """Refit one locked configuration on all development complexes."""
    seed = int(args.seed)
    _seed_everything(seed)
    radius = float(spec.get("radius", args.radius))
    r2p_k = int(spec.get("r2p_k", args.r2p_k))
    p2r_k = int(spec.get("p2r_k", args.p2r_k))
    _attach_selected_edges(data, radius, args.neighbors, r2p_k, p2r_k, True)
    model = ReciprocalAdapter(_config(spec)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)
    manager = CheckpointManager(target, "train_loss")
    order_rng = random.Random(seed)
    history = []
    for epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        if args.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        model.train()
        order = list(range(len(data)))
        order_rng.shuffle(order)
        losses = []
        for start in range(0, len(order), int(args.batch_size)):
            payloads = [data[index] for index in order[start : start + int(args.batch_size)]]
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
        epoch_seconds = time.perf_counter() - started
        record = {
            "train_loss": float(np.mean(losses)),
            "epoch_seconds": epoch_seconds,
            "train_complexes_per_second": len(data) / max(epoch_seconds, 1e-8),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2))
                if args.device.type == "cuda" else 0.0
            ),
        }
        history.append(record)
        manager.save_epoch(model, optimizer, None, epoch, record, {"spec": spec, "seed": seed, "refit": True})
    final = manager.save_final(model, int(epochs), history[-1], {"spec": spec, "seed": seed, "refit": True})
    summary = {"spec": spec, "seed": seed, "development_complexes": len(data), "epochs": int(epochs), "final": str(final), "history": history, "test_read": False}
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return summary


def run_refit(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    search_path = out / "search_summary.json"
    if not search_path.exists():
        raise FileNotFoundError(f"run search before refit: {search_path}")
    search = json.loads(search_path.read_text(encoding="utf-8"))
    spec = search["selected"]
    residual_stage = search["stages"].get("residual")
    if residual_stage is None:
        raise ValueError("search summary has no completed residual stage")
    epochs = int(round(median(item["best_epoch"] for item in residual_stage["selected"]["folds"])))
    data = _load_cache(Path(args.cache_root), "train") + _load_cache(Path(args.cache_root), "val")
    target = out / "refit" / "final"
    if (target / "summary.json").exists() and not args.force:
        summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    else:
        summary = _train_full(spec, data, args, target, epochs)
    lock = {"spec": spec, "epochs": epochs, "refit_summary": summary, "test_read": False, "holdout_locked": True}
    (out / "final_lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(lock, indent=2, ensure_ascii=False, default=float), flush=True)
    return lock


def run_evaluate(args: argparse.Namespace) -> dict:
    if not args.allow_final_holdout:
        raise ValueError("holdout evaluation requires --allow-final-holdout")
    out = Path(args.out)
    lock_path = out / "final_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("final_lock.json is required before reading a test cache")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checkpoint = out / "refit" / "final" / "final.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    test_data = _load_cache(Path(args.test_cache), "test")
    spec = lock["spec"]
    radius = float(spec.get("radius", args.radius))
    _attach_selected_edges(test_data, radius, args.neighbors, int(spec.get("r2p_k", args.r2p_k)), int(spec.get("p2r_k", args.p2r_k)), True)
    model = ReciprocalAdapter(_config(spec)).to(args.device)
    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    metrics = evaluate_dataset(model, test_data, args.device, radius, args.neighbors, include_permutation=True)
    result = {"checkpoint": str(checkpoint), "test_complexes": len(test_data), "metrics": metrics, "test_used_only_after_final_lock": True, "spec": spec}
    (out / "holdout_evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=float), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifests", type=Path, default=DEFAULT_MANIFESTS)
    common.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    common.add_argument("--out", type=Path, default=DEFAULT_OUT)
    common.add_argument("--seed", type=int, default=SINGLE_SEED)
    preflight = sub.add_parser("preflight", parents=[common])
    preflight.set_defaults(func=run_preflight)
    search = sub.add_parser("search", parents=[common])
    search.add_argument("--stage", choices=("all", "geometry", "k", "radius", "aggregation", "interaction", "residual", "edge_encoder"), default="all")
    search.add_argument("--device", default="cuda:0")
    search.add_argument("--radius", type=float, default=RADIUS)
    search.add_argument("--neighbors", type=int, default=32)
    search.add_argument("--r2p-k", type=int, default=8)
    search.add_argument("--p2r-k", type=int, default=12)
    search.add_argument("--lr", type=float, default=3e-4)
    search.add_argument("--epochs", type=int, default=60)
    search.add_argument("--patience", type=int, default=18)
    search.add_argument("--batch-size", type=int, default=16)
    search.add_argument("--resume", action="store_true")
    search.add_argument("--force", action="store_true")
    search.add_argument("--fold-workers", type=int, default=1)
    search.set_defaults(func=run_search)
    refit = sub.add_parser("refit", parents=[common])
    refit.add_argument("--device", default="cuda:0")
    refit.add_argument("--radius", type=float, default=RADIUS)
    refit.add_argument("--neighbors", type=int, default=32)
    refit.add_argument("--r2p-k", type=int, default=8)
    refit.add_argument("--p2r-k", type=int, default=12)
    refit.add_argument("--lr", type=float, default=3e-4)
    refit.add_argument("--batch-size", type=int, default=16)
    refit.add_argument("--force", action="store_true")
    refit.set_defaults(func=run_refit)
    evaluate = sub.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--test-cache", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--radius", type=float, default=RADIUS)
    evaluate.add_argument("--neighbors", type=int, default=32)
    evaluate.add_argument("--r2p-k", type=int, default=8)
    evaluate.add_argument("--p2r-k", type=int, default=12)
    evaluate.add_argument("--allow-final-holdout", action="store_true")
    evaluate.set_defaults(func=run_evaluate)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if hasattr(parsed, "device"):
        parsed.device = torch.device(parsed.device)
    parsed.func(parsed)
