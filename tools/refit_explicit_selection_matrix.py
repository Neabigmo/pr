#!/usr/bin/env python3
"""Full-development refit for the locked explicit-selection A2 adapter.

The refit consumes only the train/validation cache, uses the pre-registered
median CV epoch count, and never opens a test/blind manifest.  It retains one
atomic resumable ``last.pt`` and one final model checkpoint.
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.explicit_matrix import ExplicitMatrixConfig, ExplicitSelectionAdapter  # noqa: E402
from pr_pilot.training.checkpointing import atomic_torch_save, restore_rng_state  # noqa: E402
from run_adapter_v2_cv import _attach_selected_edges, _batched_loss, _collate_payloads  # noqa: E402
from run_explicit_selection_matrix import RADIUS, _cache_index, _fold_payloads  # noqa: E402


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, default=float) + "\n")
        handle.flush()


def _save_last(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict, spec: dict, order_rng: random.Random) -> None:
    atomic_torch_save(
        {
            "format": "scientific_checkpoint_v1",
            "kind": "last",
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "metrics": metrics,
            "metadata": {"spec": spec, "seed": int(spec["seed"]), "test_read": False, "refit": True},
            "rng": _rng_state(),
            "order_rng_state": order_rng.getstate(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if "test" in str(args.manifests).lower() or "holdout" in str(args.manifests).lower():
        raise ValueError("A2 refit accepts development manifests only")
    if "test" in str(args.cache_root).lower() or "holdout" in str(args.cache_root).lower():
        raise ValueError("A2 refit accepts development caches only")
    if int(args.epochs) < 1 or int(args.batch_size) < 1:
        raise ValueError("epochs and batch-size must be positive")

    _seed_everything(int(args.seed))
    device = torch.device(args.device)
    by_id = _cache_index(Path(args.cache_root))
    sample_ids = sorted(by_id)
    data = _fold_payloads(by_id, sample_ids)
    if len(data) != 860:
        raise ValueError(f"expected the locked 860-complex development pool, found {len(data)}")
    _attach_selected_edges(data, RADIUS, 32, 8, 12, True)

    config = ExplicitMatrixConfig(variant="A2", geometry="G2")
    spec = {
        "variant": "A2",
        "geometry": "G2",
        "radius": float(RADIUS),
        "r2p_k": 8,
        "p2r_k": 12,
        "aggregation": "mean",
        "interaction": "explicit_selection_matrix_C_plus_delta",
        "separate_edge_encoder": False,
        "projector": True,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": 1e-3,
        "test_read": False,
        "model_config": dict(config.__dict__),
    }
    model = ExplicitSelectionAdapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    last_path = out / "last.pt"
    final_path = out / "final.pt"
    start_epoch = 1
    order_rng = random.Random(int(args.seed))
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        if payload.get("kind") != "last" or payload.get("metadata", {}).get("spec", {}).get("variant") != "A2":
            raise ValueError("last checkpoint is not an A2 refit checkpoint")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng"])
        if "order_rng_state" in payload:
            order_rng.setstate(payload["order_rng_state"])
        start_epoch = int(payload["epoch"]) + 1

    history: list[dict] = []
    if metrics_path.exists() and start_epoch > 1:
        history = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for epoch in range(start_epoch, int(args.epochs) + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        order = list(range(len(data)))
        order_rng.shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), int(args.batch_size)):
            payloads = [data[index] for index in order[start : start + int(args.batch_size)]]
            packed = _collate_payloads(payloads, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                packed["protein_base"], packed["rna_base"], packed["protein_hidden"], packed["rna_hidden"],
                packed["protein_native"], packed["rna_native"], packed["edge_index_r2p"], packed["edge_geometry_r2p"],
                packed["edge_index_p2r"], packed["edge_geometry_p2r"],
            )
            p_loss = _batched_loss(output["protein_logits"], packed["protein_native"], packed["protein_active"], packed["protein_sample"], packed["batch_size"])
            r_loss = _batched_loss(output["rna_logits"], packed["rna_native"], packed["rna_active"], packed["rna_sample"], packed["batch_size"])
            p_prior = _batched_loss(packed["protein_base"].detach(), packed["protein_native"], packed["protein_active"], packed["protein_sample"], packed["batch_size"], log_probs=True)
            r_prior = _batched_loss(packed["rna_base"].detach(), packed["rna_native"], packed["rna_active"], packed["rna_sample"], packed["batch_size"], log_probs=True)
            loss = 0.5 * (p_loss / p_prior.clamp_min(1e-8) + r_loss / r_prior.clamp_min(1e-8))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {
            "epoch": int(epoch),
            "train_loss": float(np.mean(losses)),
            "epoch_seconds": float(time.perf_counter() - started),
            "train_complexes_per_second": float(len(data) / max(time.perf_counter() - started, 1e-8)),
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
        }
        history.append(record)
        _append_jsonl(metrics_path, record)
        _save_last(last_path, model, optimizer, epoch, record, spec, order_rng)
        print(json.dumps({"event": "refit_epoch_complete", **record}), flush=True)

    if not history:
        raise RuntimeError("no refit epoch completed")
    atomic_torch_save(
        {
            "format": "scientific_checkpoint_v1",
            "kind": "final",
            "epoch": int(history[-1]["epoch"]),
            "model": model.state_dict(),
            "metrics": history[-1],
            "metadata": {"spec": spec, "seed": int(args.seed), "test_read": False, "refit": True},
        },
        final_path,
    )
    summary = {
        "variant": "A2",
        "seed": int(args.seed),
        "development_complexes": len(data),
        "epochs": int(args.epochs),
        "median_cv_best_epoch": 4,
        "final": str(final_path),
        "last": str(last_path),
        "metrics": str(metrics_path),
        "history": history,
        "spec": spec,
        "test_read": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"event": "refit_complete", "out": str(out), "test_read": False}), flush=True)


if __name__ == "__main__":
    main()
