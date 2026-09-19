#!/usr/bin/env python3
"""Re-export only sequence-free ProteinMPNN encoder hidden states.

Prior logits and all geometry/masks are copied unchanged from the frozen
development cache. The output is a new cache root so old results cannot be
silently mixed with the explicit-matrix protocol.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pr_pilot.adapter_pilot.priors import ProteinMPNNPrior


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--out-cache", type=Path, required=True)
    parser.add_argument("--protein-checkout", type=Path, default=Path(r"F:\111临时\PR PILOT\third_party_checkouts_local_20260907\ProteinMPNN"))
    parser.add_argument("--protein-checkpoint", type=Path, default=Path(r"F:\111临时\PR PILOT\prior_benchmark_local_20260906\03_compute\official_baselines\seed20260905\development\ProteinMPNN\model_weights\epoch61_step2806.pt"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if "test" in str(args.source_cache).lower() or "test" in str(args.out_cache).lower():
        raise ValueError("structure-only cache export refuses test paths")
    device = torch.device(args.device)
    prior = ProteinMPNNPrior(args.protein_checkout, args.protein_checkpoint, device)
    records = []
    for split in ("train", "val"):
        files = sorted((args.source_cache / split).glob("*.pt"))
        for source_path in files:
            target_path = args.out_cache / split / source_path.name
            if target_path.exists() and target_path.stat().st_size > 0:
                continue
            payload = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
            metadata = dict(payload["metadata"])
            hidden = prior.encode_structure_only(Path(metadata["protein_view"]), metadata["protein_view_chains"])["hidden"].detach().cpu().float()
            if tuple(hidden.shape) != tuple(payload["protein_hidden"].shape):
                raise ValueError(f"hidden shape mismatch for {payload['sample_id']}: {tuple(hidden.shape)} vs {tuple(payload['protein_hidden'].shape)}")
            updated = dict(payload)
            updated["protein_hidden"] = hidden
            metadata["hidden_contract"] = "sequence_free_encoder_side"
            metadata["protein_hidden_source"] = "ProteinMPNN.encoder_before_decoder"
            updated["metadata"] = metadata
            _atomic_save(updated, target_path)
            records.append({"split": split, "sample_id": str(payload["sample_id"]), "cache": str(target_path), "hidden_shape": list(hidden.shape)})
            if len(records) % 25 == 0:
                print(json.dumps({"completed": len(records), "last": records[-1]}, ensure_ascii=False), flush=True)
    args.out_cache.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_cache": str(args.source_cache.resolve()),
        "out_cache": str(args.out_cache.resolve()),
        "newly_exported": len(records),
        "counts": {split: len(list((args.out_cache / split).glob("*.pt"))) for split in ("train", "val")},
        "hidden_contract": "sequence_free_encoder_side",
        "test_read": False,
    }
    (args.out_cache / "metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
