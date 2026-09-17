#!/usr/bin/env python3
"""Partner-use diagnostics for one validation-selected Adapter checkpoint."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pr_pilot.adapter_pilot.model import AdapterConfig, ReciprocalAdapter
from tools.run_conditional_adapter_pilot import _attach_selected_edges, _forward_payload, _load_cache


def _directional_rows(model, data, device, radius, neighbors, variant: str, seed: int):
    rng = random.Random(seed)
    rows = []
    for payload in data:
        altered = dict(payload)
        if variant == "rna_permute":
            order = list(range(len(payload["rna_native"])))
            rng.shuffle(order)
            altered["rna_native"] = payload["rna_native"][torch.tensor(order, dtype=torch.long)]
            target = "protein"
        elif variant == "protein_permute":
            order = list(range(len(payload["protein_native"])))
            rng.shuffle(order)
            altered["protein_native"] = payload["protein_native"][torch.tensor(order, dtype=torch.long)]
            target = "rna"
        else:
            raise ValueError(variant)
        with torch.inference_mode():
            output = _forward_payload(model, altered, device, radius, neighbors)
        if target == "protein":
            logp, native, interface, active = output["protein_logits"], payload["protein_native"], payload["protein_interface"], payload.get("_selected_protein_active", payload["protein_active"])
        else:
            logp, native, interface, active = output["rna_logits"], payload["rna_native"], payload["rna_interface"], payload.get("_selected_rna_active", payload["rna_active"])
        logp = F.log_softmax(logp, -1)
        native, interface, active = native.to(device), interface.to(device), active.to(device)
        nll = -logp[torch.arange(len(native), device=device), native]
        for subset, mask in (("all", torch.ones_like(active)), ("active", active), ("interface", interface)):
            if bool(mask.any()):
                rows.append({"sample_id": payload["sample_id"], "polymer": target, "subset": subset, "variant": variant, "nll": float(nll[mask].mean().cpu())})
    return rows


def main(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ReciprocalAdapter(AdapterConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = _load_cache(Path(args.cache), "val")
    _attach_selected_edges(
        data,
        float(checkpoint["radius"]),
        int(checkpoint["neighbors"]),
        checkpoint.get("r2p_neighbors"),
        checkpoint.get("p2r_neighbors"),
        bool(checkpoint.get("direction_specific", False)),
    )
    variants = ["native", "token_off", "rna_permute", "protein_permute"]
    rows = []
    for payload in data:
        for variant in ("native", "token_off"):
            with torch.inference_mode():
                output = _forward_payload(model, payload, device, float(checkpoint["radius"]), int(checkpoint["neighbors"]), token_off=variant == "token_off")
            p_active = payload.get("_selected_protein_active", payload["protein_active"])
            r_active = payload.get("_selected_rna_active", payload["rna_active"])
            for polymer, logits, native, interface, active in (("protein", output["protein_logits"], payload["protein_native"], payload["protein_interface"], p_active), ("rna", output["rna_logits"], payload["rna_native"], payload["rna_interface"], r_active)):
                logits = F.log_softmax(logits, -1)
                native, interface, active = native.to(device), interface.to(device), active.to(device)
                nll = -logits[torch.arange(len(native), device=device), native]
                for subset, mask in (("all", torch.ones_like(active)), ("active", active), ("interface", interface)):
                    if bool(mask.any()):
                        rows.append({"sample_id": payload["sample_id"], "polymer": polymer, "subset": subset, "variant": variant, "nll": float(nll[mask].mean().cpu())})
    rows.extend(_directional_rows(model, data, device, float(checkpoint["radius"]), int(checkpoint["neighbors"]), "rna_permute", args.seed))
    rows.extend(_directional_rows(model, data, device, float(checkpoint["radius"]), int(checkpoint["neighbors"]), "protein_permute", args.seed + 1))
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "dev_partner_diagnostics.tsv", sep="\t", index=False)
    summary = frame.groupby(["polymer", "subset", "variant"], as_index=False)["nll"].mean()
    summary.to_csv(out / "dev_partner_diagnostics_summary.tsv", sep="\t", index=False)
    pivot = summary.pivot_table(index=["polymer", "subset"], columns="variant", values="nll").reset_index()
    for column in ("token_off", "rna_permute", "protein_permute"):
        if column in pivot:
            pivot[f"delta_{column}_minus_native"] = pivot[column] - pivot["native"]
    result = {"checkpoint": str(args.checkpoint), "validation_complexes": len(data), "summary": pivot.to_dict(orient="records"), "test_read": False}
    (out / "dev_partner_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260917)
    main(parser.parse_args())
