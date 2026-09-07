#!/usr/bin/env python3
"""Measure concurrent CUDA memory for the two standalone prior models.

This is a resource check only.  It uses the longest frozen training example for
each polymer and performs one real forward/backward pass; it does not create a
checkpoint or alter any training manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestRow, load_protein_row, load_rna_row
from pr_pilot.training.engine import build_model_from_config
from pr_pilot.training.stages import Stage, configure_stage


def _move_graph(graph, device: torch.device):
    for name in (
        "node_x",
        "edge_index",
        "edge_x",
        "sequence",
        "interface",
        "valid",
        "fixed",
        "reference_xyz",
        "chain_index",
    ):
        setattr(graph, name, getattr(graph, name).to(device))
    return graph


def _longest_row(path: Path) -> ManifestRow:
    frame = pd.read_csv(path, sep=None, engine="python")
    if frame.empty:
        raise ValueError(f"Empty manifest: {path}")
    length_column = "length" if "length" in frame else ("protein_length" if "protein_length" in frame else "rna_length")
    index = frame[length_column].astype(int).idxmax()
    row = frame.loc[index]
    return ManifestRow(str(row["sample_id"]), Path(str(row["structure_path"])), row.to_dict())


def run_one(config_path: Path, manifest_path: Path, polymer: str, output: Path, device_name: str) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = Stage.PROTEIN_PRIOR if polymer == "protein" else Stage.RNA_PRIOR
    row = _longest_row(manifest_path)
    geometry = cfg["geometry"]
    adapter = GemmiStructureAdapter(
        int(geometry["rbf_bins"]),
        int(geometry["intra_max_neighbors"]),
        float(geometry["pr_cutoff_angstrom"]),
        int(geometry["pr_max_neighbors"]),
        float(geometry.get("coordinate_noise_angstrom", 0.0)),
        int(cfg["experiment"]["pilot_seed"]),
        bool(geometry["rich_pr_geometry"]),
    )
    device = torch.device(device_name)
    model = build_model_from_config(cfg).to(device)
    configure_stage(model, stage)
    model.train()
    graph = load_protein_row(adapter, row) if polymer == "protein" else load_rna_row(adapter, row)
    graph = _move_graph(graph, device)
    known = graph.fixed & graph.valid
    tokens = graph.sequence
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and torch.cuda.is_bf16_supported(),
    ):
        if polymer == "protein":
            logits, _ = model.protein_prior_logits(graph.node_x, graph.edge_index, graph.edge_x, tokens, known)
        else:
            logits, _ = model.rna_prior_logits(graph.node_x, graph.edge_index, graph.edge_x, tokens, known)
        loss = F.cross_entropy(logits.float(), tokens)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)
    else:
        allocated = reserved = peak = 0
    result = {
        "polymer": polymer,
        "sample_id": row.sample_id,
        "length": int(tokens.shape[0]),
        "loss": float(loss.detach().cpu()),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": int(peak),
        "allocated_bytes_after_backward": int(allocated),
        "reserved_bytes_after_backward": int(reserved),
        "device": str(device),
        "bf16": bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{polymer}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protein-manifest", type=Path, required=True)
    parser.add_argument("--rna-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--polymer", choices=["protein", "rna"])
    args = parser.parse_args()

    if args.polymer:
        manifest = args.protein_manifest if args.polymer == "protein" else args.rna_manifest
        run_one(args.config, manifest, args.polymer, args.out, args.device)
        return

    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config.resolve()),
        "--protein-manifest",
        str(args.protein_manifest.resolve()),
        "--rna-manifest",
        str(args.rna_manifest.resolve()),
        "--out",
        str(args.out.resolve()),
        "--device",
        args.device,
    ]
    env = os.environ.copy()
    processes = []
    handles = []
    for polymer in ("protein", "rna"):
        log = args.out / f"{polymer}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("w", encoding="utf-8")
        handles.append(handle)
        processes.append(subprocess.Popen(common + ["--polymer", polymer], stdout=handle, stderr=subprocess.STDOUT, env=env))
    codes = [process.wait() for process in processes]
    for handle in handles:
        handle.close()
    if any(code != 0 for code in codes):
        raise SystemExit(f"Resource smoke failed with return codes {codes}")
    print(json.dumps({"parallel_processes": 2, "return_codes": codes}, indent=2))


if __name__ == "__main__":
    main()
