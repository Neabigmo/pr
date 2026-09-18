#!/usr/bin/env python3
"""Materialize the exact G2 development cache used by residual-cap training."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.geometry import geometry_dimension
from pr_pilot.training.checkpointing import atomic_torch_save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    g2_dim = geometry_dimension("G2", 16)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted((args.source / "train").glob("*.pt")) + sorted((args.source / "val").glob("*.pt"))
    if not paths:
        raise FileNotFoundError(args.source)
    for index, path in enumerate(paths, 1):
        split = path.parent.name
        target = args.out / split / path.name
        if target.exists() and not args.force:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        geometry = payload["edge_geometry"]
        if geometry.shape[-1] < g2_dim:
            raise ValueError(f"{path}: geometry has {geometry.shape[-1]} columns, need {g2_dim}")
        payload["edge_geometry"] = geometry[:, :g2_dim].contiguous()
        payload.setdefault("metadata", {})["materialized_geometry"] = "G2"
        atomic_torch_save(payload, target)
        del payload
        if index % 50 == 0 or index == len(paths):
            print(f"materialized {index}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
