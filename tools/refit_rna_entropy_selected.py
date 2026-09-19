#!/usr/bin/env python3
"""Refit the pre-registered winner on all development complexes.

This script is deliberately separate from the CV runner.  It reads only the
development cache, derives the fixed epoch count from the completed CV
summary, and never opens either the old holdout or the new blind manifest.
The refit keeps the selected architecture and (if selected) the fold-local
sampler definition; no validation or model choice is performed here.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.model import ReciprocalAdapter
from pr_pilot.training.checkpointing import CheckpointManager
from run_adapter_v2_cv import _attach_selected_edges, _batched_loss, _collate_payloads, _load_cache
from run_rna_entropy_balance import (
    BATCH_SIZE,
    DEV_CACHE,
    EPOCHS,
    NEIGHBORS,
    OUT,
    R2P_K,
    P2R_K,
    RADIUS,
    SEED,
    _advance_sampling_rng,
    _balanced_weights,
    _config,
    _training_entropy_tau,
    build_spec,
    _seed_everything,
)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--result-root", type=Path, default=OUT)
    parser.add_argument("--cache-root", type=Path, default=DEV_CACHE)
    parser.add_argument("--blind-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    cv_path = result_root / "cv_summary.json"
    if not cv_path.exists():
        raise FileNotFoundError(cv_path)
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    selected_name = str(cv["selected_experiment"])
    group = cv["groups"][selected_name]
    best_epochs = [int(fold["best_epoch"]) for fold in group["folds"]]
    epochs = max(1, int(round(float(np.median(best_epochs)))))
    # Use the exact locked group specification.  This also supports E4/E5,
    # which live in a separate result root from the original E0--E3 run.
    spec = dict(group["spec"])
    spec["name"] = selected_name

    target = result_root / "refit" / selected_name
    summary_path = target / "summary.json"
    if summary_path.exists() and not args.force:
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    device = torch.device(args.device)
    seed = int(args.seed)
    _seed_everything(seed)
    cache_root = Path(args.cache_root)
    data = _load_cache(cache_root, "train") + _load_cache(cache_root, "val")
    expected_development = int(json.loads((Path(args.result_root) / "protocol.json").read_text(encoding="utf-8")).get("development_complexes", len(data))) if (Path(args.result_root) / "protocol.json").exists() else len(data)
    if len(data) != expected_development:
        raise ValueError(f"expected {expected_development} development complexes, found {len(data)}")
    _attach_selected_edges(data, RADIUS, NEIGHBORS, R2P_K, P2R_K, True)
    if spec["rna_gate_mode"] == "entropy_threshold_scalar":
        spec["rna_entropy_tau"] = _training_entropy_tau(data)

    weights = None
    sampler_report = {"enabled": bool(spec["balanced_sampling"]), "refit": True}
    if spec["balanced_sampling"]:
        weights, sampler_report = _balanced_weights(data)
        sampler_report["enabled"] = True
        sampler_report["refit"] = True

    model = ReciprocalAdapter(_config(spec)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    manager = CheckpointManager(target, "train_loss")
    start_epoch = 1
    if args.resume and (target / "last.pt").exists():
        start_epoch = manager.restore_last(model, optimizer, map_location=device)
    order_rng = random.Random(seed)
    _advance_sampling_rng(order_rng, len(data), start_epoch - 1, weights)
    history: list[dict] = []
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        if weights is None:
            order = list(range(len(data)))
            order_rng.shuffle(order)
        else:
            order = order_rng.choices(list(range(len(data))), weights=weights.tolist(), k=len(data))
        losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            packed = _collate_payloads([data[index] for index in order[start : start + BATCH_SIZE]], device)
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
        seconds = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "epoch_seconds": seconds,
            "train_complexes_per_second": len(data) / max(seconds, 1e-8),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
        }
        manager.save_epoch(model, optimizer, None, epoch, {k: v for k, v in record.items() if k != "epoch"}, {"spec": spec, "seed": seed, "refit": True, "sampler": sampler_report})
        history.append(record)
        print(json.dumps({"event": "refit_epoch", **record}, default=_json_default), flush=True)

    final = manager.save_final(model, epochs, history[-1], {"spec": spec, "seed": seed, "refit": True, "sampler": sampler_report})
    summary = {
        "selected_experiment": selected_name,
        "spec": spec,
        "seed": seed,
        "development_complexes": len(data),
        "cv_best_epochs": best_epochs,
        "refit_epochs": epochs,
        "final": str(final),
        "best_checkpoint": str(target / "best.pt"),
        "last_checkpoint": str(target / "last.pt"),
        "history": history,
        "test_read": False,
        "blind_read": False,
        "selection_source": str(cv_path),
        "cache_root": str(cache_root),
        "sampler": sampler_report,
    }
    target.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    lock = {
        "selected_experiment": selected_name,
        "refit_summary": str(summary_path),
        "final_checkpoint": str(final),
        "test_read": False,
        "blind_read": False,
        "new_blind_manifest": str(args.blind_manifest) if args.blind_manifest else None,
        "new_blind_status": "locked_not_read_before_final_evaluation" if args.blind_manifest else "not_declared",
    }
    (result_root / "final_lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"event": "refit_complete", **lock}, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
