#!/usr/bin/env python3
"""Train and immediately evaluate the three pre-registered RNA-gate groups.

The only architectural change exposed here is the RNA residual gate:

* M0: current scalar gate, ``g_R = sigmoid(beta_R)``;
* M1: fixed RNA gate, ``g_R = 0.25``;
* M2: capped trainable RNA gate, ``g_R = 0.25 * sigmoid(beta_R)``.

All groups use the same single seed, grouped development folds, cache, optimizer,
batch size, epoch budget, early stopping and checkpoint rules.  The holdout is
read only after every CV fold has produced its checkpoint; it is not used for
selection.  Three worker processes are used by default and are deliberately
limited to one GPU model per worker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
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
from run_adapter_v2_cv import _attach_selected_edges, _load_cache, build_grouped_folds, evaluate_dataset
from run_scientific_adapter_pilot import (
    _config,
    _train_fold_worker,
)


SEED = 20260917
MANIFESTS = Path(r"I:\PR_PILOT_SCIENTIFIC\20260917\manifest_length_v1")
SOURCE_DEV_CACHE = Path(r"I:\PR_PILOT_SCIENTIFIC\20260917\cache\g3_noise0p0")
DEV_CACHE = Path(r"I:\PR_PILOT_SCIENTIFIC\20260918\rna_residual_cap\cache_g2")
TEST_CACHE = Path(r"I:\PR_PILOT_SCIENTIFIC\20260918\new_architecture_rerun\cache\g2_r14p979_k32_test")
OUT = Path(r"I:\PR_PILOT_SCIENTIFIC\20260918\rna_residual_cap")
RADIUS = 14.979730606
NEIGHBORS = 32
R2P_K = 8
P2R_K = 12
LR = 3e-4
EPOCHS = 60
PATIENCE = 18
BATCH_SIZE = 16


EXPERIMENTS = {
    "M0_current_scalar_gate": "baseline",
    "M1_rna_fixed_gate_0p25": "fixed_quarter",
    "M2_rna_capped_gate_0p25": "capped_quarter",
}


def build_spec(mode: str) -> dict:
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
        "rna_gate_mode": mode,
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


def numeric_mean(metrics: list[dict]) -> dict:
    keys = sorted(set().union(*(item.keys() for item in metrics)))
    result = {}
    for key in keys:
        values = [item[key] for item in metrics if key in item]
        if values and all(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) for value in values):
            result[key] = float(np.mean(values))
    return result


def build_cache_path_index(cache_root: Path, output: Path, force: bool) -> dict[str, str]:
    """Resolve sample IDs once; Windows workers reuse this small JSON index."""
    if output.exists() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    files = sorted((cache_root / "train").glob("*.pt")) + sorted((cache_root / "val").glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no development caches in {cache_root}")
    index: dict[str, str] = {}
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        sample_id = str(payload["sample_id"])
        if sample_id in index:
            raise ValueError(f"duplicate cache sample_id: {sample_id}")
        index[sample_id] = str(path)
        del payload
    output.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def init_worker_from_path_index(index: dict[str, str], threads: int = 2) -> None:
    """Install the parent-built index in the imported scientific runner."""
    import run_scientific_adapter_pilot as runner

    torch.set_num_threads(int(threads))
    runner._FOLD_WORKER_BY_ID = {sample_id: Path(path) for sample_id, path in index.items()}


def evaluate_group(job: tuple[str, dict, Path, Path, str]) -> dict:
    """Evaluate one group using all three CV best checkpoints on holdout."""
    name, spec, group_dir, test_cache, device_name = job
    torch.set_num_threads(2)
    device = torch.device(device_name)
    data = _load_cache(test_cache, "test")
    _attach_selected_edges(data, RADIUS, NEIGHBORS, R2P_K, P2R_K, True)
    fold_results = []
    for fold in range(3):
        checkpoint = group_dir / f"fold{fold}" / "best.pt"
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model = ReciprocalAdapter(_config(spec)).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        metrics = evaluate_dataset(model, data, device, RADIUS, NEIGHBORS, include_permutation=True)
        fold_results.append({"fold": fold, "checkpoint": str(checkpoint), "metrics": metrics})
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
    mean = numeric_mean([item["metrics"] for item in fold_results])
    return {
        "experiment": name,
        "spec": spec,
        "test_complexes": len(data),
        "folds": fold_results,
        "mean": mean,
        "holdout_used_for_selection": False,
        "test_read_only_after_cv": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 3:
        raise ValueError("workers must be between 1 and 3")
    OUT.mkdir(parents=True, exist_ok=True)
    folds = build_grouped_folds(MANIFESTS, 3, SEED)
    cache_index = build_cache_path_index(DEV_CACHE, OUT / "development_cache_index.json", args.force)
    expected_ids = {sample_id for fold in folds for sample_id in fold["train_sample_ids"]}
    if set(cache_index) != expected_ids:
        raise ValueError("development cache index does not exactly match grouped folds")
    protocol = {
        "seed": SEED,
        "manifests": str(MANIFESTS),
        "source_development_cache": str(SOURCE_DEV_CACHE),
        "development_cache": str(DEV_CACHE),
        "test_cache": str(TEST_CACHE),
        "groups": EXPERIMENTS,
        "specs": {name: build_spec(mode) for name, mode in EXPERIMENTS.items()},
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "folds": [{"fold": fold["fold"], "n_train": fold["n_train"], "n_val": fold["n_val"]} for fold in folds],
        "holdout_used_for_selection": False,
        "checkpoint_policy": "best.pt and last.pt per experiment/fold; metrics.jsonl and summary.json",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    namespace = args_namespace(args.device, args.resume)
    jobs = []
    for name, mode in EXPERIMENTS.items():
        spec = build_spec(mode)
        for fold in folds:
            target = OUT / "cv" / name / f"fold{fold['fold']}"
            if not args.force and (target / "summary.json").exists():
                print(json.dumps({"event": "reuse_cv_summary", "experiment": name, "fold": fold["fold"]}, ensure_ascii=False), flush=True)
                continue
            jobs.append((name, spec, fold, namespace, target))

    started = time.time()
    summaries: dict[str, list[dict]] = {name: [] for name in EXPERIMENTS}
    if args.workers == 1:
        init_worker_from_path_index(cache_index, threads=8)
        for name, spec, fold, fold_args, target in jobs:
            summary = _train_fold_worker((spec, fold, fold_args, target))
            summaries[name].append(summary)
            print(json.dumps({"event": "cv_fold_complete", "experiment": name, "fold": fold["fold"], "score": summary["selection_score"]}, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker_from_path_index,
            initargs=(cache_index,),
        ) as executor:
            futures = {
                executor.submit(_train_fold_worker, (spec, fold, fold_args, target)): (name, fold["fold"])
                for name, spec, fold, fold_args, target in jobs
            }
            for future in as_completed(futures):
                name, fold_id = futures[future]
                summary = future.result()
                summaries[name].append(summary)
                print(json.dumps({"event": "cv_fold_complete", "experiment": name, "fold": fold_id, "score": summary["selection_score"]}, ensure_ascii=False), flush=True)

    # Include already completed summaries when resuming.
    for name in EXPERIMENTS:
        for fold in folds:
            if not any(int(item["fold"]) == int(fold["fold"]) for item in summaries[name]):
                summary_path = OUT / "cv" / name / f"fold{fold['fold']}" / "summary.json"
                if summary_path.exists():
                    summaries[name].append(json.loads(summary_path.read_text(encoding="utf-8")))
        summaries[name] = sorted(summaries[name], key=lambda item: int(item["fold"]))
        if len(summaries[name]) != 3:
            raise RuntimeError(f"incomplete CV results for {name}: {len(summaries[name])}/3")

    cv_summary = {
        "protocol": protocol,
        "runtime_seconds": time.time() - started,
        "groups": {
            name: {
                "spec": build_spec(mode),
                "folds": summaries[name],
                "mean_validation": numeric_mean([item["validation"] for item in summaries[name]]),
            }
            for name, mode in EXPERIMENTS.items()
        },
        "holdout_used_for_selection": False,
    }
    (OUT / "cv_summary.json").write_text(json.dumps(cv_summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"event": "cv_complete", "runtime_seconds": cv_summary["runtime_seconds"]}, ensure_ascii=False), flush=True)

    # Read holdout only now, after all CV checkpoints exist, and evaluate all
    # three groups without selecting among them using the holdout.
    test_jobs = [
        (name, build_spec(mode), OUT / "cv" / name, TEST_CACHE, args.device)
        for name, mode in EXPERIMENTS.items()
    ]
    test_results = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, 3)) as executor:
        futures = {executor.submit(evaluate_group, job): job[0] for job in test_jobs}
        for future in as_completed(futures):
            name = futures[future]
            result = future.result()
            test_results[name] = result
            print(json.dumps({"event": "holdout_complete", "experiment": name, "rna_ratio": result["mean"].get("rna_interface_ratio"), "protein_ratio": result["mean"].get("protein_interface_ratio")}, ensure_ascii=False), flush=True)
    holdout = {
        "protocol": protocol,
        "groups": test_results,
        "holdout_used_for_selection": False,
        "completed_after_all_cv": True,
    }
    (OUT / "holdout_evaluation.json").write_text(json.dumps(holdout, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"event": "all_complete", "out": str(OUT), "groups": sorted(test_results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
