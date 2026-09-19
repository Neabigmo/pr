#!/usr/bin/env python3
"""Inference-only evidence suite for the locked explicit-matrix A2 model.

This runner never retrains a prior and never accepts a test/holdout manifest.
It reuses the fold-specific A2 checkpoints on their development validation
folds. Token perturbations keep prior logits, target labels, masks and graph
fixed. Geometry perturbations rebuild G2 from the source CIF through the
audited geometry worker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.explicit_matrix import ExplicitMatrixConfig, ExplicitSelectionAdapter  # noqa: E402
from run_adapter_v2_cv import _attach_selected_edges, build_grouped_folds  # noqa: E402
from run_explicit_selection_matrix import _cache_index, _fold_payloads  # noqa: E402
from run_priority8_mechanisms import (  # noqa: E402
    DROPOUT_LEVELS,
    DROPOUT_REPEATS,
    NOISE_LEVELS,
    REWIRE_REPEATS,
    ROTATIONS,
    TRANSLATIONS,
    _degree_preserving_rewire,
    _drop_edges,
    _geometry_worker,
    _set_directional_edges,
    _stable_rng,
)

RADIUS = 14.979730606
SEED = 20260919
NEIGHBORS = 32
EXPERIMENTS = (
    "global_shuffle", "interface_shuffle", "local_swap", "edge_rewiring",
    "coordinate_noise", "rigid_body", "edge_dropout", "single_site_mutation",
)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _clone_tokens(payload: dict, protein: torch.Tensor | None = None, rna: torch.Tensor | None = None) -> dict:
    clone = dict(payload)
    if protein is not None:
        clone["protein_native"] = protein.clone()
    if rna is not None:
        clone["rna_native"] = rna.clone()
    return clone


def _load_model(checkpoint: Path, device: torch.device) -> ExplicitSelectionAdapter:
    model = ExplicitSelectionAdapter(ExplicitMatrixConfig(variant="A2")).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _forward(model: ExplicitSelectionAdapter, payload: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return model(
        payload["protein_base"].to(device), payload["rna_base"].to(device),
        payload["protein_hidden"].to(device), payload["rna_hidden"].to(device),
        payload["protein_native"].to(device), payload["rna_native"].to(device),
        payload["_selected_edge_index_r2p"].to(device), payload["_selected_geometry_r2p"].to(device),
        payload["_selected_edge_index_p2r"].to(device), payload["_selected_geometry_p2r"].to(device),
    )


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return -F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1)


def _metric_row(original: dict, altered: dict, native: dict, output: dict, fold: int, meta: dict) -> dict:
    row = {"sample_id": str(original["sample_id"]), "fold": int(fold), **meta}
    for polymer in ("protein", "rna"):
        logits = output[f"{polymer}_logits"]
        native_logits = native[f"{polymer}_logits"]
        labels = original[f"{polymer}_native"].long().to(logits.device)
        interface = original[f"{polymer}_interface"].bool().to(logits.device)
        active = altered[f"_selected_{polymer}_active"].bool().to(logits.device)
        prior = original[f"{polymer}_base"].float().to(logits.device)
        nll = _nll(logits, labels)
        native_nll = _nll(native_logits, labels)
        prior_nll = _nll(prior, labels)
        logp = F.log_softmax(logits, dim=-1)
        native_logp = F.log_softmax(native_logits, dim=-1)
        kl = (native_logp.exp() * (native_logp - logp)).sum(-1)
        for subset, mask in (("all", torch.ones_like(active)), ("active", active), ("interface", interface)):
            if bool(mask.any()):
                row[f"{polymer}_{subset}_nll"] = float(nll[mask].mean().cpu())
                row[f"{polymer}_{subset}_prior_nll"] = float(prior_nll[mask].mean().cpu())
                row[f"{polymer}_{subset}_native_delta_nll"] = float((nll[mask] - native_nll[mask]).mean().cpu())
                row[f"{polymer}_{subset}_kl_to_native"] = float(kl[mask].mean().cpu())
                row[f"{polymer}_{subset}_ratio"] = row[f"{polymer}_{subset}_nll"] / max(row[f"{polymer}_{subset}_prior_nll"], 1e-8)
                row[f"{polymer}_{subset}_recovery"] = float((logits[mask].argmax(-1) == labels[mask]).float().mean().cpu())
            else:
                for suffix in ("nll", "prior_nll", "native_delta_nll", "kl_to_native", "ratio", "recovery"):
                    row[f"{polymer}_{subset}_{suffix}"] = float("nan")
        row[f"{polymer}_active_count"] = int(active.sum())
        row[f"{polymer}_interface_count"] = int(interface.sum())
    return row


def _shuffle_tokens(payload: dict, experiment: str, repeat: int, seed: int) -> tuple[dict, dict]:
    p = payload["protein_native"].clone()
    r = payload["rna_native"].clone()
    key = int.from_bytes(hashlib.sha256(str(payload["sample_id"]).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed + 100003 * repeat + key)
    if experiment == "global_shuffle":
        p = p[torch.from_numpy(rng.permutation(len(p))).long()]
        r = r[torch.from_numpy(rng.permutation(len(r))).long()]
    elif experiment == "interface_shuffle":
        for tokens, mask in ((p, payload["protein_interface"]), (r, payload["rna_interface"])):
            ids = torch.where(mask.bool())[0]
            if len(ids) > 1:
                tokens[ids] = tokens[ids][torch.from_numpy(rng.permutation(len(ids))).long()]
    else:
        raise ValueError(experiment)
    return _clone_tokens(payload, protein=p, rna=r), {"variant": experiment, "repeat": int(repeat)}


def _token_variants(payload: dict, experiment: str, seed: int) -> Iterator[tuple[dict, dict]]:
    if experiment in {"global_shuffle", "interface_shuffle"}:
        for repeat in range(20):
            yield _shuffle_tokens(payload, experiment, repeat, seed)
        return
    if experiment == "local_swap":
        for polymer, native, mask in (("protein", payload["protein_native"], payload["protein_interface"]), ("rna", payload["rna_native"], payload["rna_interface"])):
            ids = torch.where(mask.bool())[0].tolist()
            for index, (a, b) in enumerate((pair for pair in zip(ids, ids[1:]) if pair[1] == pair[0] + 1)):
                altered = native.clone()
                altered[a], altered[b] = altered[b].clone(), altered[a].clone()
                yield _clone_tokens(payload, protein=altered if polymer == "protein" else None, rna=altered if polymer == "rna" else None), {"variant": experiment, "partner_polymer": polymer, "site_a": a, "site_b": b, "repeat": index}
        return
    if experiment == "single_site_mutation":
        for polymer, native, mask, vocab_size in (("protein", payload["protein_native"], payload["protein_interface"], 20), ("rna", payload["rna_native"], payload["rna_interface"], 4)):
            for site in torch.where(mask.bool())[0].tolist():
                original = int(native[site])
                for replacement in range(vocab_size):
                    if replacement == original:
                        continue
                    altered = native.clone()
                    altered[site] = replacement
                    yield _clone_tokens(payload, protein=altered if polymer == "protein" else None, rna=altered if polymer == "rna" else None), {"variant": experiment, "partner_polymer": polymer, "site": site, "from_token": original, "to_token": replacement}
        return
    raise ValueError(experiment)


def _geometry_jobs(payloads: list[dict], manifest_rows: dict[str, dict], kind: str, seed: int, workers: int) -> Iterator[dict]:
    jobs = []
    for payload in payloads:
        sample_id = str(payload["sample_id"])
        jobs.append({
            "sample_id": sample_id, "source_path": str(payload["source_path"]),
            "protein_chains": manifest_rows[sample_id]["protein_chains"],
            "rna_chains": manifest_rows[sample_id]["rna_chains"], "kind": kind,
            "seed": int(seed), "r2p_pairs": payload["_selected_edge_index_r2p"].cpu().numpy().tolist(),
            "p2r_pairs": payload["_selected_edge_index_p2r"].cpu().numpy().tolist(),
            "protein_native": payload["protein_native"].tolist(), "rna_native": payload["rna_native"].tolist(),
        })
    if workers <= 1:
        for job in jobs:
            yield _geometry_worker(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_geometry_worker, job) for job in jobs]
            for future in as_completed(futures):
                yield future.result()


def _geometry_variants(payload: dict, item: dict) -> Iterator[tuple[dict, dict]]:
    for variant in item.get("variants", []):
        altered = _set_directional_edges(payload, variant["r2p"], variant["p2r"])
        meta = {key: value for key, value in variant.items() if key not in {"r2p", "p2r", "geometry", "distance", "edge_index"}}
        yield altered, meta


def _dropout_variants(payload: dict, seed: int) -> Iterator[tuple[dict, dict]]:
    for probability in DROPOUT_LEVELS:
        for repeat in range(DROPOUT_REPEATS):
            r2p = _drop_edges(payload["_selected_edge_index_r2p"], payload["_selected_geometry_r2p"], probability, seed + repeat, 0)
            p2r = _drop_edges(payload["_selected_edge_index_p2r"], payload["_selected_geometry_p2r"], probability, seed + repeat, 1)
            yield _set_directional_edges(payload, r2p, p2r), {"variant": "edge_dropout", "probability": float(probability), "repeat": repeat}


def _run_fold(args, fold: dict, checkpoint: Path, manifest_rows: dict[str, dict], by_id: dict[str, Path]) -> None:
    fold_id = int(fold["fold"])
    out = Path(args.out) / f"fold{fold_id}"
    out.mkdir(parents=True, exist_ok=True)
    payloads = _fold_payloads(by_id, fold["val_sample_ids"])
    _attach_selected_edges(payloads, RADIUS, NEIGHBORS, 8, 12, True)
    device = torch.device(args.device)
    model = _load_model(checkpoint, device)
    original = {str(p["sample_id"]): p for p in payloads}
    manifest_rows = manifest_rows
    detail = out / "details.jsonl"
    if detail.exists() and args.resume:
        detail.unlink()
    handle = detail.open("a", encoding="utf-8")
    native_by_id = {}
    with torch.no_grad():
        for payload in payloads:
            sid = str(payload["sample_id"])
            native = _forward(model, payload, device)
            native_by_id[sid] = {key: value.detach() for key, value in native.items()}
            row = _metric_row(payload, payload, native, native, fold_id, {"variant": "native"})
            handle.write(json.dumps(row, default=_json_default) + "\n")
    handle.flush()
    selected = [args.experiment] if args.experiment else list(EXPERIMENTS)
    for experiment in selected:
        if experiment in {"global_shuffle", "interface_shuffle", "local_swap", "single_site_mutation"}:
            variants = ((payload, altered, meta) for payload in payloads for altered, meta in _token_variants(payload, experiment, args.seed))
        elif experiment == "edge_dropout":
            variants = ((payload, altered, meta) for payload in payloads for altered, meta in _dropout_variants(payload, args.seed))
        else:
            variants = []
            for item in _geometry_jobs(payloads, manifest_rows, experiment, args.seed, args.cpu_workers):
                payload = original[str(item["sample_id"])]
                variants.extend((payload, altered, meta) for altered, meta in _geometry_variants(payload, item))
        count = 0
        for payload, altered, meta in variants:
            with torch.no_grad():
                output = _forward(model, altered, device)
            row = _metric_row(payload, altered, native_by_id[str(payload["sample_id"])], output, fold_id, meta)
            handle.write(json.dumps(row, default=_json_default) + "\n")
            count += 1
            if count % 1000 == 0:
                handle.flush()
        handle.flush()
        _write_json(out / f"{experiment}.done.json", {"fold": fold_id, "experiment": experiment, "rows": count, "test_read": False})
    handle.close()
    _write_json(out / "fold.done.json", {"fold": fold_id, "experiments": selected, "test_read": False})


def _load_manifest_rows(manifests: Path) -> dict[str, dict]:
    import pandas as pd
    from pr_pilot.runtime.gemmi_adapter import parse_chain_list
    frames = [pd.read_csv(manifests / f"complex_{split}.tsv", sep="\t") for split in ("train", "val")]
    frame = pd.concat(frames, ignore_index=True)
    return {str(row["sample_id"]): {"protein_chains": parse_chain_list(row.get("protein_chains")), "rna_chains": parse_chain_list(row.get("rna_chains"))} for row in frame.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifests, cache_root = Path(args.manifests), Path(args.cache_root)
    if "test" in str(manifests).lower() or "holdout" in str(manifests).lower():
        raise ValueError("A2 evidence suite accepts development manifests only")
    folds = build_grouped_folds(manifests, 3, args.seed)
    if args.folds:
        wanted = set(args.folds)
        folds = [fold for fold in folds if int(fold["fold"]) in wanted]
    by_id = _cache_index(cache_root)
    expected = {sample_id for fold in folds for sample_id in fold["train_sample_ids"] + fold["val_sample_ids"]}
    if not expected.issubset(set(by_id)):
        raise ValueError("cache does not cover development folds")
    manifest_rows = _load_manifest_rows(manifests)
    protocol = {"name": "a2_evidence_suite", "seed": args.seed, "test_read": False, "model": "A2", "geometry": "G2", "radius": RADIUS, "r2p_k": 8, "p2r_k": 12, "experiments": list(EXPERIMENTS), "shuffle_repeats": 20, "coordinate_noise_angstrom": list(NOISE_LEVELS), "rigid_translation_angstrom": list(TRANSLATIONS), "rigid_rotation_degree": list(ROTATIONS), "edge_dropout_probability": list(DROPOUT_LEVELS), "edge_rewire_repeats": REWIRE_REPEATS, "checkpoint_root": str(args.checkpoint_root)}
    _write_json(Path(args.out) / "protocol.json", protocol)
    for fold in folds:
        fold_id = int(fold["fold"])
        checkpoint = Path(args.checkpoint_root) / f"fold{fold_id}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        _run_fold(args, fold, checkpoint, manifest_rows, by_id)
        print(json.dumps({"event": "fold_complete", "fold": fold_id, "test_read": False}), flush=True)
    _write_json(Path(args.out) / "complete.json", {"folds": [int(fold["fold"]) for fold in folds], "test_read": False})


if __name__ == "__main__":
    main()
