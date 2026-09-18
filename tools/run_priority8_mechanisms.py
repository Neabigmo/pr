#!/usr/bin/env python3
"""Run the first eight locked-Adapter mechanism experiments.

The runner is deliberately inference-only with respect to the two upstream
priors.  It loads the already selected development-CV Adapter checkpoints,
keeps cached prior hidden states/log-probabilities fixed, and perturbs only
partner tokens, cross-edge topology, or sequence-neutral geometry.

No test manifest/cache is accepted.  Details live outside Git on I: and are
written incrementally so an interrupted run can resume at experiment/fold
boundaries without producing extra model checkpoints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from pr_pilot.adapter_pilot.geometry import (  # noqa: E402
    build_cross_edges_from_records,
    build_pair_geometry_from_records,
)
from pr_pilot.adapter_pilot.model import ReciprocalAdapter  # noqa: E402
from pr_pilot.data.residue_vocab import PROTEIN_ALPHABET, RNA_ALPHABET  # noqa: E402
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter, parse_chain_list  # noqa: E402
from run_adapter_v2_cv import build_grouped_folds  # noqa: E402
from run_conditional_adapter_pilot import (  # noqa: E402
    _attach_selected_edges,
    _collate_payloads,
    _load_cache,
)
from run_scientific_adapter_pilot import _config  # noqa: E402


SINGLE_SEED = 20260917
RADIUS = 14.979730606
R2P_K = 8
P2R_K = 12
NEIGHBORS = 32
CACHE_RADIUS = 16.0
CPU_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))
SHUFFLE_REPEATS = 20
DROPOUT_REPEATS = 3
REWIRE_REPEATS = 3
BOOTSTRAP = 10_000
EXPERIMENTS = (
    "global_shuffle",
    "interface_shuffle",
    "local_swap",
    "edge_rewiring",
    "coordinate_noise",
    "rigid_body",
    "edge_dropout",
    "single_site_mutation",
)
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.5, 1.0)
TRANSLATIONS = (0.5, 1.0, 2.0, 4.0)
ROTATIONS = (5.0, 10.0, 20.0, 40.0)
DROPOUT_LEVELS = (0.1, 0.2, 0.4, 0.6)


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _stable_rng(seed: int, *parts: object) -> np.random.Generator:
    key = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _safe_dev_path(path: Path, label: str) -> None:
    text = str(path).lower().replace("\\", "/")
    if "/test/" in text or "test_cache" in text or "holdout" in text:
        raise ValueError(f"priority8 refuses {label} path containing test/holdout: {path}")


def _load_manifests(manifests: Path) -> dict[str, dict]:
    _safe_dev_path(manifests, "manifest")
    frames = []
    for split in ("train", "val"):
        frame = pd.read_csv(manifests / f"complex_{split}.tsv", sep="\t")
        frame["sample_id"] = frame["sample_id"].astype(str)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if frame["sample_id"].duplicated().any():
        raise ValueError("development manifest contains duplicate sample_id")
    result = {}
    for row in frame.to_dict("records"):
        result[str(row["sample_id"])] = {
            "protein_chains": parse_chain_list(row.get("protein_chains")),
            "rna_chains": parse_chain_list(row.get("rna_chains")),
        }
    return result


def _load_development(cache_root: Path) -> dict[str, dict]:
    _safe_dev_path(cache_root, "cache")
    payloads = _load_cache(cache_root, "train") + _load_cache(cache_root, "val")
    result = {}
    for payload in payloads:
        sample_id = str(payload["sample_id"])
        if sample_id in result:
            raise ValueError(f"duplicate development cache sample_id: {sample_id}")
        metadata = payload.get("metadata", {})
        if not bool(metadata.get("no_test_used", False)):
            raise ValueError(f"cache is not marked no_test_used for {sample_id}")
        result[sample_id] = payload
    return result


def _selected_spec(search_summary: Path) -> tuple[dict, dict[int, Path]]:
    _safe_dev_path(search_summary, "search summary")
    summary = json.loads(search_summary.read_text(encoding="utf-8"))
    spec = dict(summary["selected"])
    stage = summary["stages"]["edge_encoder"]["selected"]
    checkpoint_by_fold = {}
    for item in stage["folds"]:
        fold = int(item["fold"])
        checkpoint = Path(item["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        checkpoint_by_fold[fold] = checkpoint
        if bool(item.get("test_read", False)):
            raise ValueError(f"selected checkpoint was marked test_read: fold {fold}")
    if set(checkpoint_by_fold) != {0, 1, 2}:
        raise ValueError(f"expected three development checkpoints, got {sorted(checkpoint_by_fold)}")
    return spec, checkpoint_by_fold


def _load_model(spec: dict, checkpoint: Path, device: torch.device) -> ReciprocalAdapter:
    model = ReciprocalAdapter(_config(spec)).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model" not in payload:
        raise ValueError(f"checkpoint has no model state: {checkpoint}")
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def _clone_tokens(payload: dict, protein: torch.Tensor | None = None, rna: torch.Tensor | None = None) -> dict:
    clone = dict(payload)
    if protein is not None:
        clone["protein_native"] = protein.clone()
    if rna is not None:
        clone["rna_native"] = rna.clone()
    return clone


def _set_directional_edges(payload: dict, r2p: dict, p2r: dict) -> dict:
    clone = dict(payload)
    clone["_selected_edge_index_r2p"] = torch.as_tensor(r2p["edge_index"], dtype=torch.long)
    clone["_selected_geometry_r2p"] = torch.as_tensor(r2p["geometry"], dtype=torch.float32)
    clone["_selected_edge_index_p2r"] = torch.as_tensor(p2r["edge_index"], dtype=torch.long)
    clone["_selected_geometry_p2r"] = torch.as_tensor(p2r["geometry"], dtype=torch.float32)
    p_active = torch.zeros(int(payload["protein_length"]), dtype=torch.bool)
    r_active = torch.zeros(int(payload["rna_length"]), dtype=torch.bool)
    if clone["_selected_edge_index_r2p"].numel():
        p_active[torch.unique(clone["_selected_edge_index_r2p"][0])] = True
    if clone["_selected_edge_index_p2r"].numel():
        r_active[torch.unique(clone["_selected_edge_index_p2r"][1])] = True
    clone["_selected_protein_active"] = p_active
    clone["_selected_rna_active"] = r_active
    return clone


def _edge_export(edge) -> dict:
    return {
        "edge_index": np.stack([edge.protein_index, edge.rna_index]).astype(np.int64),
        "distance": np.asarray(edge.distance, dtype=np.float32),
        "geometry": np.asarray(edge.features, dtype=np.float32),
    }


def _edge_from_arrays(value: dict) -> dict:
    return {
        "edge_index": np.asarray(value["edge_index"], dtype=np.int64),
        "distance": np.asarray(value.get("distance", []), dtype=np.float32),
        "geometry": np.asarray(value["geometry"], dtype=np.float32),
    }


def _geometry_payload(payload: dict, value: dict) -> dict:
    edge = _edge_from_arrays(value)
    clone = dict(payload)
    clone["edge_index"] = torch.from_numpy(edge["edge_index"])
    clone["edge_distance"] = torch.from_numpy(edge["distance"])
    clone["edge_geometry"] = torch.from_numpy(edge["geometry"])
    for key in ("_selected_edge_index_r2p", "_selected_geometry_r2p", "_selected_edge_index_p2r", "_selected_geometry_p2r"):
        clone.pop(key, None)
    _attach_selected_edges([clone], RADIUS, NEIGHBORS, R2P_K, P2R_K, True)
    return clone


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    value = rng.normal(size=3).astype(np.float32)
    norm = float(np.linalg.norm(value))
    return value / max(norm, 1e-8)


def _rotation_matrix(axis: np.ndarray, degrees: float) -> np.ndarray:
    theta = math.radians(float(degrees))
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ], dtype=np.float32)


def _rna_centroid(records: list) -> np.ndarray:
    points = [np.asarray(record.atoms["C1'"], dtype=np.float32) for record in records if "C1'" in record.atoms and np.isfinite(record.atoms["C1'"]).all()]
    if not points:
        points = [np.asarray(value, dtype=np.float32) for record in records for value in record.atoms.values() if np.isfinite(value).all()]
    if not points:
        return np.zeros(3, dtype=np.float32)
    return np.stack(points).mean(axis=0)


def _align_records_to_tokens(records: list, target_tokens: list[int], sample_id: str, polymer: str) -> list:
    """Mirror cache alignment when a prior dropped non-canonical residues."""
    target = [int(value) for value in target_tokens]
    if len(records) == len(target):
        return list(records)
    aligned = []
    cursor = 0
    for record in records:
        if cursor < len(target) and int(record.token) == target[cursor]:
            aligned.append(record)
            cursor += 1
    if cursor != len(target):
        raise ValueError(f"{sample_id}: cannot align {polymer} records ({len(records)}) to cached tokens ({len(target)})")
    return aligned


def _geometry_worker(job: dict) -> dict:
    """CPU worker for geometry-only perturbations and rewire features."""
    sample_id = str(job["sample_id"])
    try:
        adapter = GemmiStructureAdapter(rbf_bins=16, pr_cutoff_angstrom=CACHE_RADIUS, pr_max_neighbors=NEIGHBORS)
        source = Path(job["source_path"])
        p_records = adapter._read_records(source, sample_id, "protein", job["protein_chains"])
        r_records = adapter._read_records(source, sample_id, "rna", job["rna_chains"])
        p_records = _align_records_to_tokens(p_records, job["protein_native"], sample_id, "protein")
        r_records = _align_records_to_tokens(r_records, job["rna_native"], sample_id, "rna")
        variants = []
        kind = job["kind"]
        r2p_pairs = [tuple(x) for x in np.asarray(job["r2p_pairs"], dtype=np.int64).T]
        p2r_pairs = [tuple(x) for x in np.asarray(job["p2r_pairs"], dtype=np.int64).T]
        if kind == "coordinate_noise":
            for sigma in NOISE_LEVELS:
                variant_id = f"{sample_id}|noise|{sigma}"
                r2p_edge = build_pair_geometry_from_records(
                    p_records, r_records, r2p_pairs, sample_id=variant_id,
                    bins=16, geometry_mode="G2", coordinate_noise_angstrom=float(sigma), noise_seed=int(job["seed"]),
                )
                p2r_edge = build_pair_geometry_from_records(
                    p_records, r_records, p2r_pairs, sample_id=variant_id,
                    bins=16, geometry_mode="G2", coordinate_noise_angstrom=float(sigma), noise_seed=int(job["seed"]),
                )
                variants.append({"variant": "coordinate_noise", "sigma": float(sigma), "repeat": 0, "r2p": _edge_export(r2p_edge), "p2r": _edge_export(p2r_edge)})
        elif kind == "rigid_body":
            rng = _stable_rng(int(job["seed"]), sample_id, "rigid-axis")
            axis = _unit_vector(rng)
            direction = _unit_vector(_stable_rng(int(job["seed"]), sample_id, "rigid-translation"))
            centroid = _rna_centroid(r_records)
            for amount in TRANSLATIONS:
                variant_id = f"{sample_id}|translation|{amount}"
                r2p_edge = build_pair_geometry_from_records(
                    p_records, r_records, r2p_pairs, sample_id=variant_id,
                    bins=16, geometry_mode="G2", rna_translation=direction * float(amount),
                )
                p2r_edge = build_pair_geometry_from_records(
                    p_records, r_records, p2r_pairs, sample_id=variant_id,
                    bins=16, geometry_mode="G2", rna_translation=direction * float(amount),
                )
                variants.append({"variant": "rigid_translation", "amount": float(amount), "unit": "angstrom", "r2p": _edge_export(r2p_edge), "p2r": _edge_export(p2r_edge)})
            for angle in ROTATIONS:
                rotation = _rotation_matrix(axis, float(angle))
                translation = centroid - rotation @ centroid
                variant_id = f"{sample_id}|rotation|{angle}"
                r2p_edge = build_pair_geometry_from_records(
                    p_records, r_records, r2p_pairs, sample_id=variant_id,
                    bins=16, geometry_mode="G2", rna_rotation=rotation, rna_translation=translation,
                )
                p2r_edge = build_pair_geometry_from_records(
                    p_records, r_records, p2r_pairs, sample_id=variant_id,
                    bins=16, geometry_mode="G2", rna_rotation=rotation, rna_translation=translation,
                )
                variants.append({"variant": "rigid_rotation", "amount": float(angle), "unit": "degree", "r2p": _edge_export(r2p_edge), "p2r": _edge_export(p2r_edge)})
        elif kind == "edge_rewiring":
            base_rng = _stable_rng(int(job["seed"]), sample_id, "rewire")
            r2p_base = np.asarray(job["r2p_pairs"], dtype=np.int64)
            p2r_base = np.asarray(job["p2r_pairs"], dtype=np.int64)
            for repeat in range(REWIRE_REPEATS):
                r2p_pairs = _degree_preserving_rewire(r2p_base, _stable_rng(int(job["seed"]), sample_id, "r2p", repeat))
                p2r_pairs = _degree_preserving_rewire(p2r_base, _stable_rng(int(job["seed"]), sample_id, "p2r", repeat))
                r2p_edge = build_pair_geometry_from_records(p_records, r_records, [tuple(x) for x in r2p_pairs.T], sample_id=f"{sample_id}|r2p|{repeat}", bins=16, geometry_mode="G2")
                p2r_edge = build_pair_geometry_from_records(p_records, r_records, [tuple(x) for x in p2r_pairs.T], sample_id=f"{sample_id}|p2r|{repeat}", bins=16, geometry_mode="G2")
                variants.append({"variant": "edge_rewiring", "repeat": repeat, "r2p": _edge_export(r2p_edge), "p2r": _edge_export(p2r_edge)})
            del base_rng
        else:
            raise ValueError(f"unknown geometry worker kind: {kind}")
        return {"sample_id": sample_id, "variants": variants}
    except Exception as exc:  # pragma: no cover - reported and summarized by parent
        return {"sample_id": sample_id, "variants": [], "error": f"{type(exc).__name__}: {exc}"}


def _degree_preserving_rewire(edge_index: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pairs = [tuple(map(int, pair)) for pair in np.asarray(edge_index).T]
    current = set(pairs)
    if len(pairs) < 2:
        return np.asarray(pairs, dtype=np.int64).T if pairs else np.zeros((2, 0), dtype=np.int64)
    for _ in range(max(4, len(pairs) * 4)):
        a, b = rng.choice(len(pairs), size=2, replace=False)
        p1, r1 = pairs[int(a)]
        p2, r2 = pairs[int(b)]
        if p1 == p2 or r1 == r2:
            continue
        new_a, new_b = (p1, r2), (p2, r1)
        if new_a in current or new_b in current:
            continue
        current.remove((p1, r1))
        current.remove((p2, r2))
        current.update((new_a, new_b))
        pairs[int(a)], pairs[int(b)] = new_a, new_b
    return np.asarray(pairs, dtype=np.int64).T


def _drop_edges(edge_index: torch.Tensor, geometry: torch.Tensor, probability: float, seed: int, target_axis: int) -> dict:
    edges = edge_index.cpu().numpy().astype(np.int64)
    edge_count = int(edges.shape[1])
    rng = _stable_rng(seed, "drop", probability, target_axis)
    keep = rng.random(edge_count) >= float(probability)
    if edge_count:
        # Keep every originally active target represented, so dropout does not
        # silently become a different active-mask experiment.
        for target in np.unique(edges[target_axis]):
            ids = np.flatnonzero(edges[target_axis] == target)
            if not keep[ids].any():
                keep[int(ids[0])] = True
    ids = np.flatnonzero(keep)
    return {"edge_index": edges[:, ids], "geometry": geometry.cpu().numpy()[ids], "dropped": int(edge_count - len(ids)), "original": edge_count}


def _forward_batches(model: ReciprocalAdapter, payloads: list[dict], device: torch.device, batch_size: int) -> list[dict[str, torch.Tensor]]:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(payloads), batch_size):
            chunk = payloads[start : start + batch_size]
            packed = _collate_payloads(chunk, device)
            out = model(
                packed["protein_base"], packed["rna_base"], packed["protein_hidden"], packed["rna_hidden"],
                packed["protein_native"], packed["rna_native"], packed["edge_index_r2p"], packed["edge_geometry_r2p"],
                packed["edge_index_p2r"], packed["edge_geometry_p2r"],
            )
            p_lengths = [int(item["protein_length"]) for item in chunk]
            r_lengths = [int(item["rna_length"]) for item in chunk]
            p_start = r_start = 0
            for p_len, r_len in zip(p_lengths, r_lengths):
                outputs.append({
                    "protein_logits": out["protein_logits"][p_start : p_start + p_len].detach().cpu(),
                    "rna_logits": out["rna_logits"][r_start : r_start + r_len].detach().cpu(),
                })
                p_start += p_len
                r_start += r_len
    return outputs


def _metric_row(original: dict, output: dict, reference: dict, meta: dict) -> dict:
    row = {"sample_id": str(original["sample_id"]), **meta}
    for polymer, vocab in (("protein", 20), ("rna", 4)):
        labels = original[f"{polymer}_native"].long()
        active = original[f"_selected_{polymer}_active"].bool()
        interface = original[f"{polymer}_interface"].bool()
        logits = output[f"{polymer}_logits"]
        reference_logits = reference[f"{polymer}_logits"]
        logp = F.log_softmax(logits, dim=-1)
        ref_logp = F.log_softmax(reference_logits, dim=-1)
        prior_logp = original[f"{polymer}_base"].float()
        nll = -logp[torch.arange(len(labels)), labels]
        ref_nll = -ref_logp[torch.arange(len(labels)), labels]
        prior_nll = -prior_logp[torch.arange(len(labels)), labels]
        kl = (ref_logp.exp() * (ref_logp - logp)).sum(dim=-1)
        for subset, mask in (("all", torch.ones_like(active)), ("active", active), ("interface", interface)):
            if bool(mask.any()):
                row[f"{polymer}_{subset}_nll"] = float(nll[mask].mean())
                row[f"{polymer}_{subset}_prior_nll"] = float(prior_nll[mask].mean())
                row[f"{polymer}_{subset}_ratio"] = float(nll[mask].mean() / prior_nll[mask].mean().clamp_min(1e-8))
                row[f"{polymer}_{subset}_recovery"] = float((logp[mask].argmax(-1) == labels[mask]).float().mean())
                row[f"{polymer}_{subset}_delta_nll"] = float((nll[mask] - ref_nll[mask]).mean())
                row[f"{polymer}_{subset}_kl"] = float(kl[mask].mean())
            else:
                for suffix in ("nll", "prior_nll", "ratio", "recovery", "delta_nll", "kl"):
                    row[f"{polymer}_{subset}_{suffix}"] = float("nan")
        row[f"{polymer}_active_count"] = int(active.sum())
        row[f"{polymer}_interface_count"] = int(interface.sum())
    row["worst_direction_ratio"] = max(row["protein_interface_ratio"], row["rna_interface_ratio"])
    return row


def _iter_token_variants(payload: dict, experiment: str, seed: int) -> Iterator[tuple[dict, dict]]:
    sample_id = str(payload["sample_id"])
    p_native = payload["protein_native"]
    r_native = payload["rna_native"]
    if experiment in {"global_shuffle", "interface_shuffle"}:
        repeats = SHUFFLE_REPEATS
        for repeat in range(repeats):
            rng = _stable_rng(seed, sample_id, experiment, repeat)
            p = p_native.clone()
            r = r_native.clone()
            if experiment == "global_shuffle":
                p = p[torch.from_numpy(rng.permutation(len(p))).long()]
                r = r[torch.from_numpy(rng.permutation(len(r))).long()]
            else:
                for tokens, mask in ((p, payload["protein_interface"]), (r, payload["rna_interface"])):
                    ids = torch.where(mask.bool())[0]
                    if len(ids) > 1:
                        tokens[ids] = tokens[ids][torch.from_numpy(rng.permutation(len(ids))).long()]
            yield _clone_tokens(payload, protein=p, rna=r), {"variant": experiment, "repeat": repeat}
        return
    if experiment == "local_swap":
        for polymer, native, mask in (("protein", p_native, payload["protein_interface"]), ("rna", r_native, payload["rna_interface"])):
            ids = torch.where(mask.bool())[0].tolist()
            pairs = [(a, b) for a, b in zip(ids, ids[1:]) if b == a + 1]
            for index, (a, b) in enumerate(pairs):
                tokens = native.clone()
                tokens[a], tokens[b] = tokens[b].clone(), tokens[a].clone()
                altered = _clone_tokens(payload, protein=tokens if polymer == "protein" else None, rna=tokens if polymer == "rna" else None)
                yield altered, {"variant": "local_swap", "partner_polymer": polymer, "site_a": a, "site_b": b, "repeat": index}
        return
    if experiment == "single_site_mutation":
        alphabets = (("protein", p_native, payload["protein_interface"], len(PROTEIN_ALPHABET)), ("rna", r_native, payload["rna_interface"], len(RNA_ALPHABET)))
        for polymer, native, mask, vocab_size in alphabets:
            for site in torch.where(mask.bool())[0].tolist():
                original = int(native[site])
                for replacement in range(vocab_size):
                    if replacement == original:
                        continue
                    tokens = native.clone()
                    tokens[site] = int(replacement)
                    altered = _clone_tokens(payload, protein=tokens if polymer == "protein" else None, rna=tokens if polymer == "rna" else None)
                    yield altered, {"variant": "single_site_mutation", "partner_polymer": polymer, "site": site, "from_token": original, "to_token": replacement}
        return
    raise ValueError(f"unknown token experiment: {experiment}")


def _bootstrap(values: np.ndarray, seed: int, resamples: int) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = _stable_rng(seed, "bootstrap", values.size)
    indices = rng.integers(0, values.size, size=(resamples, values.size), dtype=np.int64)
    means = values[indices].mean(axis=1)
    return {"n": int(values.size), "mean": float(values.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def _summarize_details(path: Path, seed: int, bootstrap_resamples: int) -> dict:
    raw_sum: dict[str, float] = {}
    raw_count: dict[str, int] = {}
    complex_values: dict[str, dict[str, list[float]]] = {}
    bootstrap_keys = {
        "protein_interface_delta_nll", "rna_interface_delta_nll",
        "protein_interface_kl", "rna_interface_kl",
        "protein_interface_ratio", "rna_interface_ratio",
    }
    rows = 0
    audits = {"composition_ok": 0, "degree_preserved": 0, "active_mask_same": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows += 1
            sample = str(item["sample_id"])
            complex_values.setdefault(sample, {})
            for key, value in item.items():
                if key in {"sample_id", "variant", "repeat", "partner_polymer", "site", "site_a", "site_b", "from_token", "to_token", "sigma", "amount", "unit", "fold"}:
                    continue
                if isinstance(value, (int, float)) and np.isfinite(value):
                    raw_sum[key] = raw_sum.get(key, 0.0) + float(value)
                    raw_count[key] = raw_count.get(key, 0) + 1
                    if key in bootstrap_keys:
                        complex_values[sample].setdefault(key, []).append(float(value))
    raw_mean = {key: raw_sum[key] / raw_count[key] for key in raw_sum}
    complex_mean = {}
    for sample, values in complex_values.items():
        for key, entries in values.items():
            complex_mean.setdefault(key, []).append(float(np.mean(entries)))
    bootstrap = {}
    for key in ("protein_interface_delta_nll", "rna_interface_delta_nll", "protein_interface_kl", "rna_interface_kl", "protein_interface_ratio", "rna_interface_ratio"):
        if key in complex_mean:
            bootstrap[key] = _bootstrap(np.asarray(complex_mean[key], dtype=np.float64), seed, bootstrap_resamples)
    experiment = path.stem
    # These invariants are guaranteed by the perturbation constructors.  Older
    # detail rows predate explicit per-row audit fields, so retain the counts
    # here together with the construction-level audit basis rather than
    # pretending that missing row fields were independently rechecked.
    composition_status = "changed_by_design" if experiment == "single_site_mutation" else "preserved"
    degree_status = "preserved" if experiment in {"native", "edge_rewiring"} else "not_applicable"
    audits["composition_ok"] = rows if composition_status == "preserved" else 0
    audits["degree_preserved"] = rows if degree_status == "preserved" else 0
    audits["active_mask_same"] = rows
    return {
        "rows": rows,
        "complexes": len(complex_values),
        "raw_mean": raw_mean,
        "complex_mean": {key: float(np.mean(values)) for key, values in complex_mean.items()},
        "complex_level_bootstrap": bootstrap,
        "audit_counts": audits,
        "audit_contract": {
            "basis": "construction_invariant",
            "composition": composition_status,
            "degree": degree_status,
            "active_mask": "preserved",
            "notes": "Mutation intentionally changes one partner token; dropout preserves active target coverage; rewiring preserves both endpoint degree sequences.",
        },
        "seed": int(seed),
        "bootstrap_resamples": int(bootstrap_resamples),
    }


def _append_row(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    handle.flush()


def _process_variants(model, original_payloads: dict[str, dict], references: dict[str, dict], variants: Iterable[tuple[dict, dict]], output_path: Path, fold: int, device: torch.device, batch_size: int) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        chunk_payloads = []
        chunk_meta = []
        for payload, meta in variants:
            chunk_payloads.append(payload)
            chunk_meta.append(meta)
            if len(chunk_payloads) < batch_size:
                continue
            outputs = _forward_batches(model, chunk_payloads, device, batch_size)
            for item, variant_meta, output in zip(chunk_payloads, chunk_meta, outputs):
                row = _metric_row(original_payloads[str(item["sample_id"])], output, references[str(item["sample_id"])], {"fold": fold, **variant_meta})
                _append_row(handle, row)
                count += 1
            chunk_payloads, chunk_meta = [], []
        if chunk_payloads:
            outputs = _forward_batches(model, chunk_payloads, device, batch_size)
            for item, variant_meta, output in zip(chunk_payloads, chunk_meta, outputs):
                row = _metric_row(original_payloads[str(item["sample_id"])], output, references[str(item["sample_id"])], {"fold": fold, **variant_meta})
                _append_row(handle, row)
                count += 1
    return count


def _native_rows(model, payloads: list[dict], fold: int, device: torch.device, batch_size: int, path: Path) -> dict[str, dict]:
    references = {}
    outputs = _forward_batches(model, payloads, device, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = 0
    if path.exists():
        existing_rows = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
        if existing_rows < len(payloads):
            path.unlink()
            existing_rows = 0
    if existing_rows < len(payloads):
        handle_context = path.open("a", encoding="utf-8")
    else:
        handle_context = None
    if handle_context is not None:
        with handle_context as handle:
            for payload, output in zip(payloads, outputs):
                sample_id = str(payload["sample_id"])
                references[sample_id] = output
                _append_row(handle, _metric_row(payload, output, output, {"fold": fold, "variant": "native"}))
    else:
        for payload, output in zip(payloads, outputs):
            references[str(payload["sample_id"])] = output
    marker = path.parent / f"fold{fold}.done.json"
    _write_json(marker, {"fold": int(fold), "rows": len(payloads), "test_read": False})
    return references


def _iter_geometry_variants(payloads: list[dict], manifest_rows: dict[str, dict], kind: str, seed: int, cpu_workers: int) -> Iterator[dict]:
    jobs = []
    for payload in payloads:
        sample_id = str(payload["sample_id"])
        metadata = payload["metadata"]
        jobs.append({
            "sample_id": sample_id,
            "source_path": str(payload["source_path"]),
            "protein_chains": manifest_rows[sample_id]["protein_chains"],
            "rna_chains": manifest_rows[sample_id]["rna_chains"],
            "kind": kind,
            "seed": int(seed),
            "r2p_pairs": payload["_selected_edge_index_r2p"].cpu().numpy().tolist(),
            "p2r_pairs": payload["_selected_edge_index_p2r"].cpu().numpy().tolist(),
            "protein_native": payload["protein_native"].tolist(),
            "rna_native": payload["rna_native"].tolist(),
        })
    workers = max(1, min(int(cpu_workers), len(jobs)))
    if workers == 1:
        for job in jobs:
            yield _geometry_worker(job)
        return
    # Stream one sample's result at a time instead of retaining every
    # perturbation for the whole fold in the parent process.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_geometry_worker, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()


def _run_fold_experiment(experiment: str, fold: int, model, payloads: list[dict], manifest_rows: dict[str, dict], references: dict[str, dict], original_by_id: dict[str, dict], out: Path, args) -> dict:
    detail_path = out / "details" / experiment / f"fold{fold}.jsonl"
    done_path = out / "details" / experiment / f"fold{fold}.done.json"
    if args.resume and done_path.exists():
        return json.loads(done_path.read_text(encoding="utf-8"))
    if detail_path.exists():
        detail_path.unlink()
    device = torch.device(args.device)
    batch_size = int(args.mutation_batch_size) if experiment == "single_site_mutation" else int(args.eval_batch_size)
    if experiment in {"global_shuffle", "interface_shuffle", "local_swap", "single_site_mutation"}:
        variants = ((variant, meta) for payload in payloads for variant, meta in _iter_token_variants(payload, experiment, args.seed))
        count = _process_variants(model, original_by_id, references, variants, detail_path, fold, device, batch_size)
    elif experiment in {"coordinate_noise", "rigid_body", "edge_rewiring"}:
        kind = experiment
        errors = {}
        payload_by_id = {str(payload["sample_id"]): payload for payload in payloads}

        def geometry_variants() -> Iterator[tuple[dict, dict]]:
            for item in _iter_geometry_variants(payloads, manifest_rows, kind, args.seed, args.cpu_workers):
                sample_id = str(item["sample_id"])
                if item.get("error"):
                    errors[sample_id] = item["error"]
                    continue
                payload = payload_by_id[sample_id]
                for variant in item.get("variants", []):
                    altered = _set_directional_edges(payload, variant["r2p"], variant["p2r"])
                    meta = {key: value for key, value in variant.items() if key not in {"r2p", "p2r", "geometry", "distance", "edge_index"}}
                    yield altered, meta

        count = _process_variants(model, original_by_id, references, geometry_variants(), detail_path, fold, device, batch_size)
        if errors:
            (out / "details" / experiment / f"fold{fold}.errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    elif experiment == "edge_dropout":
        def dropout_variants() -> Iterator[tuple[dict, dict]]:
            for payload in payloads:
                for probability in DROPOUT_LEVELS:
                    for repeat in range(DROPOUT_REPEATS):
                        r2p = _drop_edges(payload["_selected_edge_index_r2p"], payload["_selected_geometry_r2p"], probability, args.seed + repeat, 0)
                        p2r = _drop_edges(payload["_selected_edge_index_p2r"], payload["_selected_geometry_p2r"], probability, args.seed + repeat, 1)
                        altered = _set_directional_edges(payload, r2p, p2r)
                        yield altered, {"variant": "edge_dropout", "probability": float(probability), "repeat": repeat, "dropped_r2p": r2p["dropped"], "dropped_p2r": p2r["dropped"], "edges_r2p": r2p["original"], "edges_p2r": p2r["original"]}
        count = _process_variants(model, original_by_id, references, dropout_variants(), detail_path, fold, device, batch_size)
    else:
        raise ValueError(experiment)
    done = {"experiment": experiment, "fold": int(fold), "rows": int(count), "checkpoint_reused": True, "test_read": False}
    _write_json(done_path, done)
    return done


def run(args) -> dict:
    if int(args.seed) != SINGLE_SEED:
        raise ValueError(f"priority8 is pre-registered to seed {SINGLE_SEED}")
    manifests = Path(args.manifests)
    cache_root = Path(args.cache_root)
    out = Path(args.out)
    _safe_dev_path(manifests, "manifest")
    _safe_dev_path(cache_root, "cache")
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = _load_manifests(manifests)
    by_id = _load_development(cache_root)
    all_folds = build_grouped_folds(manifests, 3, int(args.seed))
    expected = set(sample_id for fold in all_folds for sample_id in fold["train_sample_ids"])
    if set(by_id) != expected:
        raise ValueError("development cache IDs do not exactly match grouped development folds")
    if args.folds is None:
        folds = all_folds
    else:
        requested = {int(value) for value in args.folds}
        if not requested.issubset({int(fold["fold"]) for fold in all_folds}):
            raise ValueError(f"unknown fold selection: {sorted(requested)}")
        folds = [fold for fold in all_folds if int(fold["fold"]) in requested]
    spec, checkpoint_by_fold = _selected_spec(Path(args.search_summary))
    protocol = {
        "seed": int(args.seed), "test_read": False, "prior_retrained": False,
        "adapter_retrained": False, "spec": spec, "experiments": list(EXPERIMENTS),
        "radius": RADIUS, "r2p_k": R2P_K, "p2r_k": P2R_K, "cache_radius": CACHE_RADIUS,
        "shuffle_repeats": SHUFFLE_REPEATS, "coordinate_noise_angstrom": list(NOISE_LEVELS),
        "rigid_translation_angstrom": list(TRANSLATIONS), "rigid_rotation_degree": list(ROTATIONS),
        "edge_dropout_probability": list(DROPOUT_LEVELS), "bootstrap_resamples": int(args.bootstrap_resamples),
        "geometry_mode": "G2",
        "geometry_perturbation": "fixed_selected_edge_pairs_recomputed_geometry",
        "coordinate_noise_rebuilds_topology": False,
        "rigid_body_rebuilds_topology": False,
        "legacy_pre_fixed_geometry": str(out / "legacy_pre_fixed_geometry"),
        "evaluated_folds": [int(fold["fold"]) for fold in all_folds],
        "execution_folds": [int(fold["fold"]) for fold in folds],
        "cache_root": str(cache_root), "manifest_root": str(manifests),
        "checkpoints": {str(fold): str(path) for fold, path in checkpoint_by_fold.items()},
    }
    _write_json(out / ("smoke_protocol.json" if args.smoke else "protocol.json"), protocol)
    selected_experiments = list(EXPERIMENTS)
    if args.smoke:
        payload_limit = max(1, int(args.smoke_limit))
    else:
        payload_limit = None
    work_folds = []
    for fold in folds:
        ids = fold["val_sample_ids"]
        if payload_limit is not None:
            ids = ids[:payload_limit]
        work_folds.append((fold, ids))
    root = out / "smoke" if args.smoke else out
    root.mkdir(parents=True, exist_ok=True)
    for fold, ids in work_folds:
        payloads = [dict(by_id[sample_id]) for sample_id in ids]
        _attach_selected_edges(payloads, RADIUS, NEIGHBORS, R2P_K, P2R_K, True)
        original_by_id = {str(item["sample_id"]): item for item in payloads}
        model = _load_model(spec, checkpoint_by_fold[int(fold["fold"])], torch.device(args.device))
        references = _native_rows(model, payloads, int(fold["fold"]), torch.device(args.device), args.eval_batch_size, root / "details" / "native" / f"fold{fold['fold']}.jsonl")
        print(json.dumps({"event": "native_complete", "fold": fold["fold"], "complexes": len(payloads), "test_read": False}), flush=True)
        for experiment in selected_experiments:
            result = _run_fold_experiment(experiment, int(fold["fold"]), model, payloads, manifest_rows, references, original_by_id, root, args)
            print(json.dumps({"event": "experiment_complete", **result, "test_read": False}), flush=True)
        del model
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        del payloads, original_by_id, references
        gc.collect()
    if args.smoke:
        _write_json(root / "smoke_complete.json", {"experiments": selected_experiments, "folds": [fold["fold"] for fold, _ in work_folds], "test_read": False})
        return {"smoke": True, "test_read": False}
    summaries = {}
    for experiment in ["native", *EXPERIMENTS]:
        detail_files = sorted((root / "details" / experiment).glob("fold*.jsonl"))
        if not detail_files:
            continue
        combined = root / "details" / f"{experiment}.jsonl"
        with combined.open("w", encoding="utf-8") as out_handle:
            for detail in detail_files:
                with detail.open("r", encoding="utf-8") as in_handle:
                    for chunk in iter(lambda: in_handle.read(1024 * 1024), ""):
                        out_handle.write(chunk)
        summaries[experiment] = _summarize_details(combined, int(args.seed), int(args.bootstrap_resamples))
        _write_json(root / "summaries" / f"{experiment}.json", summaries[experiment])
    result = {"protocol": protocol, "summaries": summaries, "test_read": False, "complete": True}
    _write_json(root / "priority8_summary.json", result)
    print(json.dumps({"event": "priority8_complete", "experiments": list(summaries), "test_read": False}, ensure_ascii=False), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--search-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SINGLE_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-workers", type=int, default=CPU_WORKERS)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--mutation-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP)
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Evaluate only selected development folds; existing detail files are still included in the final summary.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=3)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    run(parsed)
