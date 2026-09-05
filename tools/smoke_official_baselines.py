#!/usr/bin/env python3
"""Tiny real-data loader/train/checkpoint smoke for pinned official baselines.

Run after ``run_official_baselines.py --prepare-only`` and CPU preflight, before
long GPU training. The goal is not performance: it proves our converted structures
are accepted by the *actual pinned upstream loaders*, survive one optimization
step/epoch and produce a readable checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from tools.run_official_baselines import clone_locked


def _run(command: list[str], cwd: Path) -> None:
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _protein_smoke(repo: Path, prepared: Path, out: Path, seed: int) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_seeded_upstream.py")),
        "--seed", str(seed),
        "--script", str(repo / "training" / "training.py"),
        "--",
        "--path_for_training_data", str(prepared / "proteinmpnn"),
        "--path_for_outputs", str(out),
        "--num_epochs", "1",
        "--save_model_every_n_epochs", "1",
        "--reload_data_every_n_epochs", "1",
        "--num_examples_per_epoch", "2",
        "--batch_size", "1000",
        "--max_protein_length", "1000",
        "--backbone_noise", "0.1",
        "--gradient_norm", "1.0",
        "--mixed_precision", "False",
    ]
    _run(command, repo)
    checkpoint = out / "model_weights" / "epoch_last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    for key in ["model_state_dict", "optimizer_state_dict", "epoch", "step"]:
        if key not in payload:
            raise ValueError(f"ProteinMPNN smoke checkpoint lacks {key}")
    return checkpoint


def _na_smoke(repo: Path, prepared: Path, out: Path, seed: int) -> Path:
    source = prepared / "na_mpnn" / "na_mpnn_from_scratch.json"
    if not source.exists():
        raise FileNotFoundError(source)
    cfg = json.loads(source.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    cfg["BASE_FOLDER"] = str(out.resolve())
    cfg["TOTAL_STEPS"] = 1
    cfg["SAVE_EVERY_N_STEPS"] = 1
    cfg["MAX_NUMBER_OF_PDBS_TRAIN"] = min(2, int(cfg["MAX_NUMBER_OF_PDBS_TRAIN"]))
    cfg["MAX_NUMBER_OF_PDBS_VALID"] = min(2, int(cfg["MAX_NUMBER_OF_PDBS_VALID"]))
    cfg["NUM_WORKERS"] = 0
    cfg["MIXED_PRECISION"] = 0
    smoke_cfg = out / "smoke_config.json"
    smoke_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_seeded_upstream.py")),
        "--seed", str(seed),
        "--script", str(repo / "na_run.py"),
        "--",
        str(smoke_cfg),
    ]
    _run(command, repo)
    checkpoint = out / "last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    for key in ["model_state_dict", "optimizer_state_dict", "epoch", "step"]:
        if key not in payload:
            raise ValueError(f"NA-MPNN smoke checkpoint lacks {key}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--third-party-root", type=Path, default=Path("third_party/checkouts"))
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--keep-smoke-checkpoints", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    prepared = args.prepared.resolve()
    out = args.out.resolve()
    upstream = clone_locked(repo_root, args.third_party_root.resolve())
    protein_ckpt = _protein_smoke(upstream["ProteinMPNN"], prepared, out / "ProteinMPNN", args.seed)
    na_ckpt = _na_smoke(upstream["NA-MPNN"], prepared, out / "NA-MPNN", args.seed)
    summary = {
        "status": "PASS",
        "prepared": str(prepared),
        "seed": int(args.seed),
        "ProteinMPNN": {"checkpoint": str(protein_ckpt), "real_upstream_loader_train_save": True},
        "NA-MPNN": {"checkpoint": str(na_ckpt), "real_upstream_loader_train_save": True},
        "scientific_metrics_interpreted": False,
    }
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.keep_smoke_checkpoints:
        # Keep logs/config/summary but delete model binaries so a smoke checkpoint
        # can never be mistaken for a reportable baseline.
        protein_ckpt.unlink(missing_ok=True)
        na_ckpt.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
