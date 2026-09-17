#!/usr/bin/env python3
"""Run the frozen-prior conditional Adapter pilot.

The tool has three deliberately separated phases:

``audit``      full-heavy-atom geometry audit on the 900-complex train split;
``prepare``    run the pinned priors once and cache frozen hidden states/logits;
``screen``     train P1--P4 single-seed Adapter candidates on train/dev only;
``evaluate``   evaluate a selected dev configuration on the frozen 86-complex test.

No test manifest is read by ``audit``, ``prepare`` or ``screen``.  Caches are
stored outside the repository by default so the code workspace stays readable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from pr_pilot.adapter_pilot.geometry import build_cross_edges, heavy_contact_audit
from pr_pilot.adapter_pilot.model import AdapterConfig, ReciprocalAdapter
from pr_pilot.adapter_pilot.priors import NAMPrior, ProteinMPNNPrior, ensure_pdb_view, source_chain_order, view_chain_ids
from pr_pilot.data.residue_vocab import PROTEIN_ALPHABET, RNA_ALPHABET
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter, parse_chain_list
from pr_pilot.runtime.manifest_dataset import canonical_interface_ids


DEFAULT_ROOT = Path(r"F:\111临时\PR PILOT\remote_return_20260911")
DEFAULT_OUT = Path(r"F:\111临时\PR PILOT\pilot_conditional_adapter_20260916")
DEFAULT_P_CHECKPOINT = Path(r"F:\111临时\PR PILOT\prior_benchmark_local_20260906\03_compute\official_baselines\seed20260905\development\ProteinMPNN\model_weights\epoch61_step2806.pt")
DEFAULT_R_CHECKPOINT = Path(r"F:\111临时\PR PILOT\prior_benchmark_local_20260906\03_compute\official_baselines\seed20260905\development\NA-MPNN\s_996.pt")
DEFAULT_CHECKOUTS = Path(r"F:\111临时\PR PILOT\third_party_checkouts_local_20260907")


def _safe(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def resolve_path(path: object, data_root: Path) -> Path:
    value = Path(str(path))
    if value.exists():
        return value
    text = str(value).replace("\\", "/")
    marker = "/data/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        candidate = data_root / "data" / suffix
        if candidate.exists():
            return candidate
    candidate = data_root / "data" / "raw_shards" / "complex" / value.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"cannot resolve structure path {path}")


def rows(manifest_dir: Path, split: str) -> pd.DataFrame:
    path = manifest_dir / f"complex_{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t")
    if frame.sample_id.astype(str).duplicated().any():
        raise ValueError(f"duplicate sample_id in {path}")
    return frame


def _chains(row: pd.Series, key: str) -> list[str]:
    value = parse_chain_list(row.get(key))
    if not value:
        raise ValueError(f"{row.sample_id}: missing {key}")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cache_name(row: pd.Series) -> str:
    return f"{_safe(str(row.sample_id))}_{str(row.sample_id).replace('/', '_')}.pt"


def _native_tokens(records: list, alphabet: str) -> torch.Tensor:
    # GemmiStructureAdapter stores canonical project-vocabulary ids, not
    # one-letter strings.  Keeping this conversion explicit prevents a silent
    # alphabet re-indexing during cache creation.
    return torch.tensor([int(record.token) for record in records], dtype=torch.long)


def _check_alignment(name: str, result: dict, records: list, alphabet: str) -> None:
    if int(result["length"]) != len(records):
        raise ValueError(f"{name} length mismatch: prior={result['length']} runtime={len(records)}")
    expected = _native_tokens(records, alphabet)
    if name == "ProteinMPNN":
        # Upstream alphabet is ARND... while this project reports ACDE... .
        upstream_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        observed = torch.tensor([upstream_alphabet.index(c) for c in str(result["sequence"])], dtype=torch.long)
        observed = torch.tensor([alphabet.index(upstream_alphabet[int(x)]) for x in observed], dtype=torch.long)
    else:
        # NA-MPNN shared tokens use DA/DC/DG/DT slots; convert to AUGC.
        reverse = {21: "A", 22: "C", 23: "G", 24: "U"}
        observed = torch.tensor([RNA_ALPHABET.index(reverse.get(int(x), "X")) for x in result["tokens"]], dtype=torch.long)
    if not torch.equal(expected, observed):
        raise ValueError(f"{name} residue order/token alignment mismatch")


def _runtime_residue_key(residue_id: str) -> str:
    """Convert runtime ``chain:numicode:name`` ids to NA-MPNN key format."""
    match = re.match(r"^(.*?):(-?\d+)([^:]*):[^:]+$", str(residue_id))
    if not match:
        raise ValueError(f"invalid runtime residue id: {residue_id}")
    return f"{match.group(1)}:{match.group(2)}:{match.group(3)}"


def _prior_objects(args, device: torch.device):
    p_checkout = Path(args.checkouts) / "ProteinMPNN"
    r_checkout = Path(args.checkouts) / "NA-MPNN"
    return (
        ProteinMPNNPrior(p_checkout, Path(args.protein_checkpoint), device),
        NAMPrior(r_checkout, Path(args.rna_checkpoint), device),
    )


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _audit_chunk(items: list[tuple[int, dict]], data_root: str) -> list[tuple[int, dict]]:
    """Audit one contiguous, non-overlapping source slice in one worker."""
    root = Path(data_root)
    adapter = GemmiStructureAdapter(rbf_bins=16, pr_cutoff_angstrom=8.0, pr_max_neighbors=12)
    audit_radii = tuple(float(value) for value in range(4, 21))
    output = []
    for index, raw in items:
        row = pd.Series(raw)
        path = resolve_path(row.structure_path, root)
        pchains, rchains = _chains(row, "protein_chains"), _chains(row, "rna_chains")
        p = adapter._read_records(path, str(row.sample_id), "protein", pchains)
        r = adapter._read_records(path, str(row.sample_id), "rna", rchains)
        pair_distances: list[float] = []
        anchor_distances: list[float] = []
        # Vectorize over all RNA atoms for each protein residue.  This is
        # algebraically identical to the residue-pair minimum, but avoids
        # hundreds of thousands of tiny NumPy allocations per complex.
        r_xyz_blocks = [np.stack([xyz for name, xyz in rr.atoms.items() if not name.startswith("V") and np.isfinite(xyz).all()]) for rr in r]
        r_offsets = np.cumsum([0] + [len(block) for block in r_xyz_blocks[:-1]], dtype=np.int64)
        r_xyz = np.concatenate(r_xyz_blocks, axis=0)
        r_c1 = np.asarray([rr.atoms.get("C1'", np.full(3, np.nan, dtype=np.float32)) for rr in r], dtype=np.float32)
        r_c1_valid = np.isfinite(r_c1).all(axis=1)
        p_ca = np.asarray([pp.atoms.get("CA", np.full(3, np.nan, dtype=np.float32)) for pp in p], dtype=np.float32)
        p_ca_valid = np.isfinite(p_ca).all(axis=1)
        protein_contact_partner_counts = [0 for _ in p]
        rna_contact_partner_counts = [0 for _ in r]
        protein_neighbor_counts = {str(radius): [] for radius in audit_radii}
        rna_neighbor_counts = {str(radius): [] for radius in audit_radii}
        for i in range(len(p)):
            if p_ca_valid[i] and bool(r_c1_valid.any()):
                distances = np.linalg.norm(r_c1[r_c1_valid] - p_ca[i][None, :], axis=1)
            else:
                distances = np.asarray([], dtype=np.float32)
            for radius in audit_radii:
                protein_neighbor_counts[str(radius)].append(int(np.count_nonzero(distances <= radius)))
        for j in range(len(r)):
            if r_c1_valid[j] and bool(p_ca_valid.any()):
                distances = np.linalg.norm(p_ca[p_ca_valid] - r_c1[j][None, :], axis=1)
            else:
                distances = np.asarray([], dtype=np.float32)
            for radius in audit_radii:
                rna_neighbor_counts[str(radius)].append(int(np.count_nonzero(distances <= radius)))
        for i, pr in enumerate(p):
            p_xyz = np.stack([xyz for name, xyz in pr.atoms.items() if not name.startswith("V") and np.isfinite(xyz).all()])
            atomwise = np.linalg.norm(p_xyz[:, None, :] - r_xyz[None, :, :], axis=-1).min(axis=0)
            residuewise = np.minimum.reduceat(atomwise, r_offsets)
            contact_mask = residuewise < 5.0
            protein_contact_partner_counts[i] = int(contact_mask.sum())
            for partner_index in np.flatnonzero(contact_mask):
                rna_contact_partner_counts[int(partner_index)] += 1
            for j, value in enumerate(residuewise):
                if value < 5.0:
                    pair_distances.append(float(value))
                    if p_ca_valid[i] and r_c1_valid[j]:
                        anchor_distances.append(float(np.linalg.norm(p_ca[i] - r_c1[j])))
        p_interface_ids, r_interface_ids = canonical_interface_ids(
            path, pchains, rchains, float(row.get("canonical_interface_cutoff_angstrom", 6.0))
        )
        output.append((index, {
            "sample_id": str(row.sample_id),
            "protein_length": len(p),
            "rna_length": len(r),
            "protein_interface_nodes": int(sum(record.residue_id in p_interface_ids for record in p)),
            "rna_interface_nodes": int(sum(record.residue_id in r_interface_ids for record in r)),
            "protein_contact_active_nodes": int(sum(value > 0 for value in protein_contact_partner_counts)),
            "rna_contact_active_nodes": int(sum(value > 0 for value in rna_contact_partner_counts)),
            "contact_pairs_lt5": len(pair_distances),
            "contact_distances_lt5": pair_distances,
            "anchor_distances_ca_c1_for_contacts": anchor_distances,
            "protein_contact_partner_counts": protein_contact_partner_counts,
            "rna_contact_partner_counts": rna_contact_partner_counts,
            "anchor_neighbor_counts_by_radius": {
                "protein": protein_neighbor_counts,
                "rna": rna_neighbor_counts,
            },
            "protein_neighbor_counts": protein_neighbor_counts,
            "rna_neighbor_counts": rna_neighbor_counts,
        }))
    return output


def run_audit(args) -> dict:
    manifest_dir = Path(args.manifests)
    data_root = Path(args.data_root)
    train = rows(manifest_dir, "train")
    adapter = GemmiStructureAdapter(rbf_bins=16, pr_cutoff_angstrom=8.0, pr_max_neighbors=12)
    all_contact_distances: list[float] = []
    rows_out: list[dict] = []
    audit_path = Path(args.out) / "audit" / "train_geometry.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.exists() and not args.force:
        cached = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        all_contact_distances = [float(item) for row in cached for item in row.get("contact_distances_lt5", [])]
        rows_out = cached
    else:
        audit_path.unlink(missing_ok=True)
        raw_rows = [(index, item._asdict()) for index, item in enumerate(train.itertuples(index=False))]
        workers = max(1, min(int(args.workers), len(raw_rows)))
        # Small contiguous chunks let fast shards return while a large
        # structure is still being processed, keeping the progress log live.
        step = max(1, math.ceil(len(raw_rows) / (workers * 4)))
        chunks = [raw_rows[start : start + step] for start in range(0, len(raw_rows), step)]
        completed: dict[int, dict] = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_audit_chunk, chunk, str(data_root)) for chunk in chunks]
            with tqdm(total=len(raw_rows), desc=f"geometry audit ({workers} source shards)", unit="complex") as progress:
                for future in as_completed(futures):
                    for index, result in future.result():
                        completed[index] = result
                        progress.update(1)
        if set(completed) != set(range(len(raw_rows))):
            raise RuntimeError("geometry audit source shards have overlap or omission")
        for index in sorted(completed):
            result = completed[index]
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            rows_out.append(result)
            all_contact_distances.extend(result["contact_distances_lt5"])
    if not all_contact_distances:
        raise ValueError("geometry audit found no heavy-atom contacts <5 A")
    values = np.asarray(all_contact_distances, dtype=np.float64)
    all_anchor_distances = np.asarray([float(item) for row in rows_out for item in row.get("anchor_distances_ca_c1_for_contacts", [])], dtype=np.float64)
    if not all_anchor_distances.size:
        raise ValueError("geometry audit found no complete C-alpha/C1-prime anchors among true contacts")
    anchor_quantiles = {str(q): float(np.quantile(all_anchor_distances, q)) for q in (0.95, 0.98, 0.99)}
    neighbor_quantiles: dict[str, dict[str, dict[str, float]]] = {}
    for radius in range(4, 21):
        key = str(float(radius))
        neighbor_quantiles[key] = {}
        for polymer, field in (("protein", "protein_neighbor_counts"), ("rna", "rna_neighbor_counts")):
            counts = np.asarray([value for row in rows_out for value in row.get(field, {}).get(key, [])], dtype=np.float64)
            neighbor_quantiles[key][polymer] = {str(q): float(np.quantile(counts, q)) for q in (0.95, 0.99)}
    contact_partner_quantiles = {}
    for polymer, field in (("protein", "protein_contact_partner_counts"), ("rna", "rna_contact_partner_counts")):
        counts = np.asarray([value for row in rows_out for value in row.get(field, [])], dtype=np.float64)
        active_counts = counts[counts > 0]
        contact_partner_quantiles[polymer] = {
            "all_targets": {
                "0.50": float(np.quantile(counts, 0.50)),
                "0.95": float(np.quantile(counts, 0.95)),
                "0.99": float(np.quantile(counts, 0.99)),
                "max": float(np.max(counts)),
            },
            "active_targets_only": {
                "0.50": float(np.quantile(active_counts, 0.50)),
                "0.95": float(np.quantile(active_counts, 0.95)),
                "0.99": float(np.quantile(active_counts, 0.99)),
                "max": float(np.max(active_counts)),
            },
            "K95contact": int(math.ceil(float(np.quantile(active_counts, 0.95)))),
            "K99contact": int(math.ceil(float(np.quantile(active_counts, 0.99)))),
        }
    summary = {
        "split": "complex_train",
        "complexes": len(rows_out),
        "true_contact_pairs_lt5": int(values.size),
        "anchor_radius_quantiles_ca_c1": anchor_quantiles,
        "true_contacts_with_complete_ca_c1": int(all_anchor_distances.size),
        "neighbor_count_quantiles_by_integer_radius": neighbor_quantiles,
        "contact_partner_count_quantiles": contact_partner_quantiles,
        "contact_distance_min": float(values.min()),
        "contact_distance_median": float(np.median(values)),
        "contact_distance_max": float(values.max()),
        "definition": "true contact means full heavy-atom residue-pair minimum distance <5 A; radius quantiles are C-alpha/C1-prime distances for those contacts",
        "note": "The legacy radius table counts valid C-alpha/C1-prime anchor neighbors. K^contact is defined separately by contact_partner_count_quantiles: per-target partners whose full heavy-atom residue-pair minimum distance is <5 A. Test is not read.",
    }
    (Path(args.out) / "audit" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _audit_split_rows(manifest_dir: Path, data_root: Path, split: str, workers: int) -> list[dict]:
    frame = rows(manifest_dir, split)
    raw_rows = [(index, item._asdict()) for index, item in enumerate(frame.itertuples(index=False))]
    if not raw_rows:
        return []
    workers = max(1, min(int(workers), len(raw_rows)))
    step = max(1, math.ceil(len(raw_rows) / (workers * 4)))
    chunks = [raw_rows[start : start + step] for start in range(0, len(raw_rows), step)]
    completed: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_audit_chunk, chunk, str(data_root)) for chunk in chunks]
        with tqdm(total=len(raw_rows), desc=f"balance audit {split} ({workers} source shards)", unit="complex") as progress:
            for future in as_completed(futures):
                for index, result in future.result():
                    completed[index] = result
                    progress.update(1)
    if set(completed) != set(range(len(raw_rows))):
        raise RuntimeError(f"balance audit source shards have overlap or omission in {split}")
    return [completed[index] for index in sorted(completed)]


def _quantile_summary(values: list[int | float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def run_balance_report(args) -> dict:
    """Write per-complex length, mask, contact, and directional-edge balance data."""
    out = Path(args.out)
    audit_dir = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    train_audit_path = audit_dir / "train_geometry.jsonl"
    if train_audit_path.exists() and not args.force:
        train_records = [json.loads(line) for line in train_audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        train_records = _audit_split_rows(Path(args.manifests), Path(args.data_root), "train", int(args.workers))
    val_records = _audit_split_rows(Path(args.manifests), Path(args.data_root), "val", int(args.workers))
    cache_by_split = {
        split: {item["sample_id"]: item for item in _load_cache(Path(args.cache), split)}
        for split in ("train", "val")
    }
    report_rows: list[dict] = []
    for split, records in (("train", train_records), ("val", val_records)):
        for record in records:
            payload = cache_by_split[split].get(str(record["sample_id"]))
            if payload is None:
                raise KeyError(f"missing cached payload for {split}/{record['sample_id']}")
            shared = _select_edges(payload, float(args.radius), int(args.shared_k), target="union")
            r2p = _select_edges(payload, float(args.radius), int(args.r2p_k), target="protein")
            p2r = _select_edges(payload, float(args.radius), int(args.p2r_k), target="rna")
            report_rows.append({
                "split": split,
                "sample_id": str(record["sample_id"]),
                "protein_length": int(record["protein_length"]),
                "rna_length": int(record["rna_length"]),
                "protein_active_nodes_contact": int(record["protein_contact_active_nodes"]),
                "rna_active_nodes_contact": int(record["rna_contact_active_nodes"]),
                "protein_interface_nodes": int(record["protein_interface_nodes"]),
                "rna_interface_nodes": int(record["rna_interface_nodes"]),
                "contact_pairs_lt5": int(record["contact_pairs_lt5"]),
                "protein_contact_partner_counts": record["protein_contact_partner_counts"],
                "rna_contact_partner_counts": record["rna_contact_partner_counts"],
                "shared_k_edges": int(shared.shape[1]),
                "r2p_k_edges": int(r2p.shape[1]),
                "p2r_k_edges": int(p2r.shape[1]),
                "r2p_target_nodes": int(torch.unique(r2p[0]).numel()) if r2p.numel() else 0,
                "p2r_target_nodes": int(torch.unique(p2r[1]).numel()) if p2r.numel() else 0,
            })
    balance_path = audit_dir / "data_balance.jsonl"
    with balance_path.open("w", encoding="utf-8") as handle:
        for record in report_rows:
            handle.write(json.dumps(record) + "\n")
    summary: dict[str, object] = {
        "train_complexes": len(train_records),
        "val_complexes": len(val_records),
        "radius": float(args.radius),
        "shared_k": int(args.shared_k),
        "r2p_k": int(args.r2p_k),
        "p2r_k": int(args.p2r_k),
        "definition": "contact partner counts are per-target counts of residue partners with minimum full heavy-atom distance <5 A; selected graph edges use the cached anchor-radius graph",
        "splits": {},
    }
    for split in ("train", "val"):
        rows_for_split = [row for row in report_rows if row["split"] == split]
        p_counts = [value for row in rows_for_split for value in row["protein_contact_partner_counts"]]
        r_counts = [value for row in rows_for_split for value in row["rna_contact_partner_counts"]]
        p2r_edges = sum(int(row["p2r_k_edges"]) for row in rows_for_split)
        r2p_targets = sum(int(row["r2p_target_nodes"]) for row in rows_for_split)
        summary["splits"][split] = {
            "complexes": len(rows_for_split),
            "protein_length": _quantile_summary([row["protein_length"] for row in rows_for_split]),
            "rna_length": _quantile_summary([row["rna_length"] for row in rows_for_split]),
            "protein_active_nodes_contact": _quantile_summary([row["protein_active_nodes_contact"] for row in rows_for_split]),
            "rna_active_nodes_contact": _quantile_summary([row["rna_active_nodes_contact"] for row in rows_for_split]),
            "protein_interface_nodes": _quantile_summary([row["protein_interface_nodes"] for row in rows_for_split]),
            "rna_interface_nodes": _quantile_summary([row["rna_interface_nodes"] for row in rows_for_split]),
            "contact_partner_count_quantiles": {
                "protein_target": {
                    "all_targets": _quantile_summary(p_counts),
                    "active_targets_only": _quantile_summary([value for value in p_counts if value > 0]),
                },
                "rna_target": {
                    "all_targets": _quantile_summary(r_counts),
                    "active_targets_only": _quantile_summary([value for value in r_counts if value > 0]),
                },
            },
            "edge_totals": {
                "shared_k": sum(int(row["shared_k_edges"]) for row in rows_for_split),
                "r2p_k": sum(int(row["r2p_k_edges"]) for row in rows_for_split),
                "p2r_k": p2r_edges,
                "r2p_target_nodes": r2p_targets,
                "p2r_target_nodes": sum(int(row["p2r_target_nodes"]) for row in rows_for_split),
                "p2r_edges_over_r2p_target_nodes": float(p2r_edges / max(1, r2p_targets)),
            },
        }
    (audit_dir / "data_balance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _prepare_one(row: pd.Series, args, p_prior, r_prior, device: torch.device, view_root: Path, cache_dir: Path, split_name: str = "train") -> dict:
    data_root = Path(args.data_root)
    source = resolve_path(row.structure_path, data_root)
    pchains, rchains = _chains(row, "protein_chains"), _chains(row, "rna_chains")
    p_view = ensure_pdb_view(source, str(row.sample_id), pchains, "protein", view_root)
    r_view = ensure_pdb_view(source, str(row.sample_id), rchains, "rna", view_root)
    p_view_chains = view_chain_ids(source_chain_order(source, pchains))
    r_view_chains = view_chain_ids(source_chain_order(source, rchains))
    edge, p_records, r_records = build_cross_edges(
        source, str(row.sample_id), pchains, rchains,
        radius_angstrom=float(args.cache_radius), max_neighbors=int(args.cache_neighbors), bins=16, geometry_mode="G2",
        candidate_neighbors=int(args.cache_neighbors),
        coordinate_noise_angstrom=float(args.noise), noise_seed=int(args.seed),
    )
    p_result = p_prior.encode_backbone(p_view, p_view_chains)
    r_result = r_prior.encode_backbone(r_view, r_view_chains)
    _check_alignment("ProteinMPNN", p_result, p_records, PROTEIN_ALPHABET)
    # NA-MPNN may mask incomplete residues, while the project vocabulary may
    # reject a modified residue that the upstream parser accepts (for example
    # 5MC/YG).  Align by residue key in both directions and keep only the
    # intersection; never infer a missing token from positional offsets.
    r_source_order = source_chain_order(source, rchains)
    view_to_source = dict(zip(r_view_chains, r_source_order))
    prior_keys = [
        f"{view_to_source.get(str(key).split(':', 1)[0], str(key).split(':', 1)[0])}:{str(key).split(':', 1)[1]}"
        for key in r_result["residue_keys"]
    ]
    runtime_keys = [_runtime_residue_key(record.residue_id) for record in r_records]
    runtime_by_key: dict[str, list[int]] = {}
    for index, key in enumerate(runtime_keys):
        runtime_by_key.setdefault(key, []).append(index)
    used_runtime: set[int] = set()
    prior_keep: list[int] = []
    common_runtime: list[int] = []
    na_to_project = {21: 0, 22: 3, 23: 2, 24: 1}
    for prior_index, key in enumerate(prior_keys):
        candidates = [index for index in runtime_by_key.get(key, []) if index not in used_runtime]
        if not candidates:
            continue
        expected = na_to_project.get(int(r_result["tokens"][prior_index].item()))
        matching = [index for index in candidates if expected is not None and int(r_records[index].token) == expected]
        runtime_index = matching[0] if matching else candidates[0]
        used_runtime.add(runtime_index)
        prior_keep.append(prior_index)
        common_runtime.append(runtime_index)
    if not common_runtime:
        raise ValueError(f"NA-MPNN/project residue-key intersection is empty for {row.sample_id}")
    if common_runtime != list(range(len(r_records))):
        remap = {old: new for new, old in enumerate(common_runtime)}
        edge_keep = np.asarray([k for k, value in enumerate(edge.rna_index) if int(value) in remap], dtype=np.int64)
        edge = type(edge)(
            edge.protein_index[edge_keep],
            np.asarray([remap[int(value)] for value in edge.rna_index[edge_keep]], dtype=np.int64),
            edge.distance[edge_keep], edge.features[edge_keep], edge.feature_dim, edge.radius_angstrom, edge.max_neighbors,
        )
        r_records = [r_records[index] for index in common_runtime]
    prior_indices = torch.tensor(prior_keep, dtype=torch.long, device=r_result["hidden"].device)
    r_result = {
        **r_result,
        "hidden": r_result["hidden"][prior_indices],
        "logits": r_result["logits"][prior_indices],
        "log_probs": r_result["log_probs"][prior_indices],
        "tokens": r_result["tokens"][prior_indices],
        "residue_keys": [r_result["residue_keys"][index] for index in prior_keep],
        "length": len(prior_keep),
    }
    _check_alignment("NA-MPNN", r_result, r_records, RNA_ALPHABET)
    p_native = _native_tokens(p_records, PROTEIN_ALPHABET)
    r_native = _native_tokens(r_records, RNA_ALPHABET)
    p_ids, r_ids = canonical_interface_ids(source, pchains, rchains, float(row.get("canonical_interface_cutoff_angstrom", 6.0)))
    p_interface = torch.tensor([record.residue_id in p_ids for record in p_records], dtype=torch.bool)
    r_interface = torch.tensor([record.residue_id in r_ids for record in r_records], dtype=torch.bool)
    p_active = torch.zeros(len(p_records), dtype=torch.bool)
    r_active = torch.zeros(len(r_records), dtype=torch.bool)
    p_active[torch.from_numpy(edge.protein_index)] = True
    r_active[torch.from_numpy(edge.rna_index)] = True
    payload = {
        "sample_id": str(row.sample_id),
        "source_path": str(source),
        "protein_length": len(p_records),
        "rna_length": len(r_records),
        "protein_hidden": p_result["hidden"].detach().cpu().float(),
        "rna_hidden": r_result["hidden"].detach().cpu().float(),
        "protein_base": p_result["log_probs"].detach().cpu().float(),
        "rna_base": r_result["log_probs"].detach().cpu().float(),
        "protein_native": p_native,
        "rna_native": r_native,
        "protein_interface": p_interface,
        "rna_interface": r_interface,
        "protein_active": p_active,
        "rna_active": r_active,
        "edge_index": torch.from_numpy(np.stack([edge.protein_index, edge.rna_index])).long(),
        "edge_distance": torch.from_numpy(edge.distance).float(),
        "edge_geometry": torch.from_numpy(edge.features).float(),
        "metadata": {
            "protein_view": str(p_view), "rna_view": str(r_view), "protein_view_chains": p_view_chains, "rna_view_chains": r_view_chains, "cache_radius": float(args.cache_radius),
            "cache_neighbors": int(args.cache_neighbors), "coordinate_noise_angstrom": float(args.noise), "prior_checkpoint_protein": str(args.protein_checkpoint),
            "prior_checkpoint_rna": str(args.rna_checkpoint), "no_test_used": split_name != "test", "split": split_name,
        },
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / _cache_name(row)
    torch.save(payload, out)
    return {"sample_id": str(row.sample_id), "cache": str(out), "protein_length": len(p_records), "rna_length": len(r_records), "edges": int(len(edge.distance))}


def run_prepare(args) -> dict:
    _seed_everything(int(args.seed))
    device = torch.device(args.device)
    p_prior, r_prior = _prior_objects(args, device)
    out = Path(args.out)
    cache_root = Path(args.cache_dir) if args.cache_dir is not None else out / "cache" / f"noise{str(args.noise).replace('.', 'p')}"
    view_root = out / "views"
    manifest_dir = Path(args.manifests)
    index_path = cache_root / "index.jsonl"
    cache_root.mkdir(parents=True, exist_ok=True)
    logs = cache_root / "progress.jsonl"
    if args.force:
        logs.unlink(missing_ok=True)
    results = []
    for split in tuple(args.splits):
        target = cache_root / split
        target.mkdir(parents=True, exist_ok=True)
        frame = rows(manifest_dir, split)
        for row in tqdm((pd.Series(item._asdict()) for item in frame.itertuples(index=False)), total=len(frame), desc=f"prepare {split}", unit="complex"):
            out_file = target / _cache_name(row)
            if out_file.exists() and not args.force:
                result = {"sample_id": str(row.sample_id), "cache": str(out_file), "reused": True}
            else:
                result = _prepare_one(row, args, p_prior, r_prior, device, view_root, target, split)
                result["reused"] = False
            with logs.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "completed", "split": split, **result}) + "\n")
            results.append({"split": split, **result})
    metadata = {
        "counts": {split: int(sum(item["split"] == split for item in results)) for split in args.splits},
        "cache_root": str(cache_root), "test_read": "test" in args.splits, "device": str(device), "seed": int(args.seed),
        "progress_log": str(logs),
    }
    (cache_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _load_cache(cache_root: Path, split: str) -> list[dict]:
    files = sorted((cache_root / split).glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no caches in {cache_root / split}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in files]


def _select_edges(payload: dict, radius: float, neighbors: int, target: str = "union") -> torch.Tensor:
    """Select deterministic edges, optionally capped per directional target.

    ``union`` reproduces the original symmetric graph: top-K neighbors for
    every protein and every RNA node are unioned.  ``protein`` selects only
    top-K neighbors per protein target (R->P); ``rna`` selects only top-K per
    RNA target (P->R).  The latter is required for direction-specific K.
    """
    if target not in {"union", "protein", "rna"}:
        raise ValueError(f"unknown edge target mode: {target}")
    p = payload["edge_index"][0].numpy()
    r = payload["edge_index"][1].numpy()
    distance = payload["edge_distance"].numpy()
    candidates = np.where(distance <= float(radius))[0]
    if len(candidates) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    keep: set[int] = set()
    groups = (p, r) if target == "union" else (p if target == "protein" else r,)
    for values in groups:
        for node in np.unique(values[candidates]):
            group = candidates[values[candidates] == node]
            group = group[np.argsort(distance[group])[: int(neighbors)]]
            keep.update(int(x) for x in group)
    selected = np.asarray(sorted(keep), dtype=np.int64)
    return payload["edge_index"][:, torch.from_numpy(selected)]


def _selected_geometry(payload: dict, edge_index: torch.Tensor) -> torch.Tensor:
    full_index = payload["edge_index"]
    if edge_index.shape[1] == 0:
        return payload["edge_geometry"][:0]
    # Map selected pairs back to the cache's stable edge ordering.
    lookup = {(int(full_index[0, k]), int(full_index[1, k])): k for k in range(full_index.shape[1])}
    ids = [lookup[(int(edge_index[0, k]), int(edge_index[1, k]))] for k in range(edge_index.shape[1])]
    return payload["edge_geometry"][torch.tensor(ids, dtype=torch.long)]


def _attach_selected_edges(
    data: list[dict],
    radius: float,
    neighbors: int,
    r2p_neighbors: int | None = None,
    p2r_neighbors: int | None = None,
    direction_specific: bool = False,
) -> list[dict]:
    """Cache shared or independent directional edge subgraphs."""
    if direction_specific and (r2p_neighbors is None or p2r_neighbors is None):
        raise ValueError("direction-specific edges require r2p_neighbors and p2r_neighbors")
    for payload in data:
        if direction_specific:
            r2p = _select_edges(payload, radius, int(r2p_neighbors), target="protein")
            p2r = _select_edges(payload, radius, int(p2r_neighbors), target="rna")
        else:
            shared = _select_edges(payload, radius, int(neighbors), target="union")
            r2p = shared
            p2r = shared
        payload["_selected_edge_index_r2p"] = r2p
        payload["_selected_geometry_r2p"] = _selected_geometry(payload, r2p)
        payload["_selected_edge_index_p2r"] = p2r
        payload["_selected_geometry_p2r"] = _selected_geometry(payload, p2r)
        p_active = torch.zeros(int(payload["protein_length"]), dtype=torch.bool)
        r_active = torch.zeros(int(payload["rna_length"]), dtype=torch.bool)
        if r2p.numel():
            p_active[torch.unique(r2p[0])] = True
        if p2r.numel():
            r_active[torch.unique(p2r[1])] = True
        payload["_selected_protein_active"] = p_active
        payload["_selected_rna_active"] = r_active
    return data


def _forward_payload(model: ReciprocalAdapter, payload: dict, device: torch.device, radius: float, neighbors: int, token_off: bool = False) -> dict[str, torch.Tensor]:
    edge_index_r2p = payload.get("_selected_edge_index_r2p")
    geometry_r2p = payload.get("_selected_geometry_r2p")
    edge_index_p2r = payload.get("_selected_edge_index_p2r")
    geometry_p2r = payload.get("_selected_geometry_p2r")
    if edge_index_r2p is None or geometry_r2p is None or edge_index_p2r is None or geometry_p2r is None:
        shared = _select_edges(payload, radius, neighbors, target="union")
        edge_index_r2p = shared
        geometry_r2p = _selected_geometry(payload, shared)
        edge_index_p2r = shared
        geometry_p2r = geometry_r2p
    return model(
        payload["protein_base"].to(device), payload["rna_base"].to(device), payload["protein_hidden"].to(device), payload["rna_hidden"].to(device),
        payload["protein_native"].to(device), payload["rna_native"].to(device),
        edge_index_r2p.to(device), geometry_r2p.to(device), edge_index_p2r.to(device), geometry_p2r.to(device), token_off=token_off,
    )


def _collate_payloads(payloads: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Pack variable-size complexes into one disconnected graph batch."""
    p_lengths = [int(item["protein_hidden"].shape[0]) for item in payloads]
    r_lengths = [int(item["rna_hidden"].shape[0]) for item in payloads]
    p_offsets = np.cumsum([0, *p_lengths[:-1]], dtype=np.int64)
    r_offsets = np.cumsum([0, *r_lengths[:-1]], dtype=np.int64)
    edges_r2p: list[torch.Tensor] = []
    geometries_r2p: list[torch.Tensor] = []
    edges_p2r: list[torch.Tensor] = []
    geometries_p2r: list[torch.Tensor] = []
    for payload, p_offset, r_offset in zip(payloads, p_offsets, r_offsets):
        edge_r2p = payload["_selected_edge_index_r2p"].clone()
        edge_r2p[0] += int(p_offset)
        edge_r2p[1] += int(r_offset)
        edges_r2p.append(edge_r2p)
        geometries_r2p.append(payload["_selected_geometry_r2p"])
        edge_p2r = payload["_selected_edge_index_p2r"].clone()
        edge_p2r[0] += int(p_offset)
        edge_p2r[1] += int(r_offset)
        edges_p2r.append(edge_p2r)
        geometries_p2r.append(payload["_selected_geometry_p2r"])
    p_sample = torch.cat([torch.full((length,), index, dtype=torch.long) for index, length in enumerate(p_lengths)])
    r_sample = torch.cat([torch.full((length,), index, dtype=torch.long) for index, length in enumerate(r_lengths)])
    edge_index_r2p = torch.cat(edges_r2p, dim=1) if edges_r2p else torch.zeros((2, 0), dtype=torch.long)
    edge_geometry_r2p = torch.cat(geometries_r2p, dim=0) if geometries_r2p else torch.zeros((0, 114), dtype=torch.float32)
    edge_index_p2r = torch.cat(edges_p2r, dim=1) if edges_p2r else torch.zeros((2, 0), dtype=torch.long)
    edge_geometry_p2r = torch.cat(geometries_p2r, dim=0) if geometries_p2r else torch.zeros((0, 114), dtype=torch.float32)
    def cat(name: str) -> torch.Tensor:
        return torch.cat([item[name] for item in payloads], dim=0)
    protein_active = torch.cat([item["_selected_protein_active"] for item in payloads], dim=0)
    rna_active = torch.cat([item["_selected_rna_active"] for item in payloads], dim=0)
    return {
        "protein_base": cat("protein_base").to(device), "rna_base": cat("rna_base").to(device),
        "protein_hidden": cat("protein_hidden").to(device), "rna_hidden": cat("rna_hidden").to(device),
        "protein_native": cat("protein_native").to(device), "rna_native": cat("rna_native").to(device),
        "protein_active": protein_active.to(device), "rna_active": rna_active.to(device),
        "protein_sample": p_sample.to(device), "rna_sample": r_sample.to(device),
        "edge_index_r2p": edge_index_r2p.to(device), "edge_geometry_r2p": edge_geometry_r2p.to(device),
        "edge_index_p2r": edge_index_p2r.to(device), "edge_geometry_p2r": edge_geometry_p2r.to(device),
        "batch_size": len(payloads),
    }


def _batched_loss(
    logits: torch.Tensor,
    native: torch.Tensor,
    active: torch.Tensor,
    sample: torch.Tensor,
    batch_size: int,
    log_probs: bool = False,
) -> torch.Tensor:
    """Average active-token loss per complex using one exact mask."""
    mask = active.bool()
    if not bool(mask.any()):
        return logits.sum() * 0.0
    values = F.nll_loss(logits[mask], native[mask], reduction="none") if log_probs else F.cross_entropy(logits[mask], native[mask], reduction="none")
    sums = logits.new_zeros((batch_size,))
    counts = logits.new_zeros((batch_size,))
    sums.index_add_(0, sample[mask], values)
    counts.index_add_(0, sample[mask], torch.ones_like(values))
    valid = counts > 0
    return (sums[valid] / counts[valid]).mean()


def _metrics(model: ReciprocalAdapter | None, data: list[dict], device: torch.device, radius: float, neighbors: int, token_off: bool = False) -> dict:
    accum: dict[str, list[float]] = {key: [] for key in ("protein_all_nll", "rna_all_nll", "protein_active_nll", "rna_active_nll", "protein_interface_nll", "rna_interface_nll", "protein_all_recovery", "rna_all_recovery", "protein_active_recovery", "rna_active_recovery", "protein_interface_recovery", "rna_interface_recovery", "protein_interface_base_nll", "rna_interface_base_nll")}
    for payload in data:
        if model is None:
            p_log, r_log = payload["protein_base"].to(device), payload["rna_base"].to(device)
        else:
            out = _forward_payload(model, payload, device, radius, neighbors, token_off)
            p_log, r_log = F.log_softmax(out["protein_logits"], -1), F.log_softmax(out["rna_logits"], -1)
        p_active = payload["_selected_protein_active"]
        r_active = payload["_selected_rna_active"]
        for name, logp, native, active, interface, base in (("protein", p_log, payload["protein_native"], p_active, payload["protein_interface"], payload["protein_base"]), ("rna", r_log, payload["rna_native"], r_active, payload["rna_interface"], payload["rna_base"])):
            native, active, interface = native.to(device), active.to(device), interface.to(device)
            nll = -logp[torch.arange(len(native), device=device), native]
            pred = logp.argmax(-1)
            for suffix, mask in (("all", torch.ones_like(active)), ("active", active), ("interface", interface)):
                if bool(mask.any()):
                    accum[f"{name}_{suffix}_nll"].append(float(nll[mask].mean().cpu()))
                    accum[f"{name}_{suffix}_recovery"].append(float((pred[mask] == native[mask]).float().mean().cpu()))
            base_nll = -base.to(device)[torch.arange(len(native), device=device), native]
            if bool(interface.any()):
                accum[f"{name}_interface_base_nll"].append(float(base_nll[interface].mean().cpu()))
    result = {key: float(np.mean(value)) if value else float("nan") for key, value in accum.items()}
    result["protein_interface_ratio"] = result["protein_interface_nll"] / result["protein_interface_base_nll"]
    result["rna_interface_ratio"] = result["rna_interface_nll"] / result["rna_interface_base_nll"]
    result["relative_interface_nll"] = 0.5 * (result["protein_interface_ratio"] + result["rna_interface_ratio"])
    result["worst_direction_ratio"] = max(result["protein_interface_ratio"], result["rna_interface_ratio"])
    return result


def _config_from_args(args, geometry: str, aggregation: str) -> AdapterConfig:
    return AdapterConfig(geometry=geometry, aggregation=aggregation, hidden_dim=128, edge_dim=64, token_dim=32, message_dim=128, layers=1, dropout=0.1, rbf_bins=16)


def train_candidate(
    cache_root: Path,
    out_dir: Path,
    config: AdapterConfig,
    radius: float,
    neighbors: int,
    lr: float,
    noise: float,
    seed: int,
    device: torch.device,
    epochs: int = 60,
    patience: int = 18,
    batch_size: int = 16,
    loss_mode: str = "absolute",
    direction_specific: bool = False,
    r2p_neighbors: int | None = None,
    p2r_neighbors: int | None = None,
) -> dict:
    if loss_mode not in {"absolute", "prior_normalized"}:
        raise ValueError(f"unknown loss mode: {loss_mode}")
    _seed_everything(seed)
    train = _load_cache(cache_root, "train")
    val = _load_cache(cache_root, "val")
    _attach_selected_edges(train, radius, neighbors, r2p_neighbors, p2r_neighbors, direction_specific)
    _attach_selected_edges(val, radius, neighbors, r2p_neighbors, p2r_neighbors, direction_specific)
    model = ReciprocalAdapter(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-3)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    best_metric = float("inf")
    bad = 0
    history = []
    train_rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train)))
        train_rng.shuffle(order)
        losses = []
        for start in range(0, len(order), int(batch_size)):
            payloads = [train[index] for index in order[start : start + int(batch_size)]]
            packed = _collate_payloads(payloads, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                packed["protein_base"], packed["rna_base"], packed["protein_hidden"], packed["rna_hidden"],
                packed["protein_native"], packed["rna_native"],
                packed["edge_index_r2p"], packed["edge_geometry_r2p"], packed["edge_index_p2r"], packed["edge_geometry_p2r"],
            )
            p_loss = _batched_loss(out["protein_logits"], packed["protein_native"], packed["protein_active"], packed["protein_sample"], packed["batch_size"])
            r_loss = _batched_loss(out["rna_logits"], packed["rna_native"], packed["rna_active"], packed["rna_sample"], packed["batch_size"])
            if loss_mode == "prior_normalized":
                p_prior = _batched_loss(packed["protein_base"].detach(), packed["protein_native"], packed["protein_active"], packed["protein_sample"], packed["batch_size"], log_probs=True)
                r_prior = _batched_loss(packed["rna_base"].detach(), packed["rna_native"], packed["rna_active"], packed["rna_sample"], packed["batch_size"], log_probs=True)
                loss = 0.5 * (p_loss / p_prior.clamp_min(1e-8) + r_loss / r_prior.clamp_min(1e-8))
            else:
                p_prior, r_prior = None, None
                loss = 0.5 * (p_loss + r_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_metrics = _metrics(model, val, device, radius, neighbors)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
        history.append(record)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        torch.save({"model": model.state_dict(), "config": config.__dict__, "epoch": epoch, "seed": seed, "radius": radius, "neighbors": neighbors, "lr": lr, "noise": noise, "validation_used": True}, last_path)
        if val_metrics["worst_direction_ratio"] < best_metric - 1e-7:
            best_metric = val_metrics["worst_direction_ratio"]
            bad = 0
            torch.save({"model": model.state_dict(), "config": config.__dict__, "epoch": epoch, "seed": seed, "radius": radius, "neighbors": neighbors, "r2p_neighbors": r2p_neighbors, "p2r_neighbors": p2r_neighbors, "direction_specific": direction_specific, "loss_mode": loss_mode, "lr": lr, "noise": noise, "validation_used": True, "selection_metric": "worst_direction_ratio", "selection_value": best_metric}, best_path)
        else:
            bad += 1
        if bad >= patience:
            break
    summary = {"out": str(out_dir), "best": str(best_path), "epochs_run": len(history), "best_worst_direction_ratio": best_metric, "best_epoch": int(torch.load(best_path, map_location="cpu", weights_only=False)["epoch"]), "config": config.__dict__, "radius": radius, "neighbors": neighbors, "r2p_neighbors": r2p_neighbors, "p2r_neighbors": p2r_neighbors, "direction_specific": direction_specific, "loss_mode": loss_mode, "lr": lr, "noise": noise, "seed": seed, "batch_size": int(batch_size)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _audit_neighborhood_configs(out: Path) -> dict[str, dict[str, float | int]]:
    summary_path = out / "audit" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"run the 900-complex geometry audit first: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    quantiles = summary["anchor_radius_quantiles_ca_c1"]
    neighbor_table = summary["neighbor_count_quantiles_by_integer_radius"]

    def choice(radius_quantile: str, k_quantile: str) -> dict[str, float | int]:
        radius = float(quantiles[radius_quantile])
        lookup_radius = str(float(math.ceil(radius)))
        counts = neighbor_table[lookup_radius]
        neighbors = int(math.ceil(max(float(counts["protein"][k_quantile]), float(counts["rna"][k_quantile]))))
        return {"radius": radius, "neighbors": max(1, neighbors), "lookup_radius": float(lookup_radius)}

    return {
        "N1": choice("0.95", "0.95"),
        "N2": choice("0.98", "0.95"),
        "N3": choice("0.98", "0.99"),
        "N4": choice("0.99", "0.99"),
    }


def _screen_configs(neighborhoods: dict[str, dict[str, float | int]]) -> list[dict]:
    return [
        {"name": "P1_G0", "geometry": "G0", "aggregation": "A2"},
        {"name": "P1_G1", "geometry": "G1", "aggregation": "A2"},
        {"name": "P1_G2", "geometry": "G2", "aggregation": "A2"},
        {"name": "P2_A0", "geometry": "G2", "aggregation": "A0"},
        {"name": "P2_A1", "geometry": "G2", "aggregation": "A1"},
        {"name": "P2_A2", "geometry": "G2", "aggregation": "A2"},
        {"name": "P3_N1", "geometry": "G2", "aggregation": "A2", **neighborhoods["N1"]},
        {"name": "P3_N2", "geometry": "G2", "aggregation": "A2", **neighborhoods["N2"]},
        {"name": "P3_N3", "geometry": "G2", "aggregation": "A2", **neighborhoods["N3"]},
        {"name": "P3_N4", "geometry": "G2", "aggregation": "A2", **neighborhoods["N4"]},
        {"name": "P4_LR1e-4", "geometry": "G2", "aggregation": "A2", "lr": 1e-4},
        {"name": "P4_LR3e-4", "geometry": "G2", "aggregation": "A2", "lr": 3e-4},
        {"name": "P4_LR1e-3", "geometry": "G2", "aggregation": "A2", "lr": 1e-3},
    ]


def run_screen(args) -> dict:
    cache_root = Path(args.cache)
    device = torch.device(args.device)
    out = Path(args.out) / "screening"
    neighborhoods = _audit_neighborhood_configs(Path(args.out))
    default_radius = float(neighborhoods["N3"]["radius"])
    default_neighbors = int(neighborhoods["N3"]["neighbors"])
    results = []
    for spec in _screen_configs(neighborhoods):
        target = out / spec["name"]
        if (target / "summary.json").exists() and not args.force:
            results.append(json.loads((target / "summary.json").read_text(encoding="utf-8")))
            continue
        radius = float(spec.get("radius", default_radius if args.radius is None else args.radius))
        neighbors = int(spec.get("neighbors", default_neighbors if args.neighbors is None else args.neighbors))
        config = _config_from_args(args, spec["geometry"], spec["aggregation"])
        results.append(train_candidate(cache_root, target, config, radius, neighbors, float(spec.get("lr", args.lr)), float(spec.get("noise", 0.0)), int(args.seed), device, int(args.epochs), int(args.patience), int(args.batch_size)))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    results = sorted(results, key=lambda item: item.get("best_worst_direction_ratio", item.get("best_relative_interface_nll", float("inf"))))
    summary = {"screening": results, "selection_rule": "lowest validation worst-direction interface ratio; test not read", "seed": int(args.seed)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_balance(args) -> dict:
    """Run the four controlled B0--B3 train/validation experiments."""
    out = Path(args.out) / "balance"
    radius = float(args.radius)
    specs = [
        {"name": "B0_absolute_sharedK51_sharedEdge", "loss_mode": "absolute", "direction_specific": False, "separate_edge_encoders": False},
        {"name": "B1_priorNormalized_sharedK51_sharedEdge", "loss_mode": "prior_normalized", "direction_specific": False, "separate_edge_encoders": False},
        {"name": "B2_priorNormalized_K8_12_sharedEdge", "loss_mode": "prior_normalized", "direction_specific": True, "separate_edge_encoders": False},
        {"name": "B3_priorNormalized_K8_12_separateEdge", "loss_mode": "prior_normalized", "direction_specific": True, "separate_edge_encoders": True},
    ]
    results = []
    for spec in specs:
        target = out / spec["name"]
        if (target / "summary.json").exists() and not args.force:
            result = json.loads((target / "summary.json").read_text(encoding="utf-8"))
        else:
            config = AdapterConfig(
                geometry="G2", aggregation="A2", hidden_dim=128, edge_dim=64,
                token_dim=32, message_dim=128, layers=1, dropout=0.1,
                rbf_bins=16, separate_edge_encoders=bool(spec["separate_edge_encoders"]),
            )
            result = train_candidate(
                Path(args.cache), target, config, radius, int(args.shared_k), float(args.lr), 0.0,
                int(args.seed), torch.device(args.device), int(args.epochs), int(args.patience), int(args.batch_size),
                loss_mode=str(spec["loss_mode"]), direction_specific=bool(spec["direction_specific"]),
                r2p_neighbors=int(args.r2p_k), p2r_neighbors=int(args.p2r_k),
            )
        result["experiment"] = spec["name"]
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "experiments": results,
        "selection_rule": "minimum validation worst_direction_ratio=max(protein_interface_ratio,rna_interface_ratio); test not read",
        "seed": int(args.seed),
        "radius": radius,
        "shared_k": int(args.shared_k),
        "r2p_k": int(args.r2p_k),
        "p2r_k": int(args.p2r_k),
        "batch_size": int(args.batch_size),
    }
    summary["ranking"] = [item["experiment"] for item in sorted(results, key=lambda item: item["best_worst_direction_ratio"])]
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _paired_bootstrap(rows: pd.DataFrame, value_a: str, value_b: str, repeats: int, seed: int) -> dict:
    grouped = rows.groupby("sample_id")[[value_a, value_b]].mean().dropna()
    delta = grouped[value_a].to_numpy() - grouped[value_b].to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    return {"n_complex": int(len(delta)), "mean_delta_a_minus_b": float(delta.mean()), "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "bootstrap_repeats": repeats}


def _evaluate_one(payload: dict, model: ReciprocalAdapter | None, device: torch.device, radius: float, neighbors: int, token_off: bool, common_rna_positions: set[tuple[str, int]] | None = None) -> list[dict]:
    if model is None:
        p_log, r_log = payload["protein_base"].to(device), payload["rna_base"].to(device)
    else:
        out = _forward_payload(model, payload, device, radius, neighbors, token_off)
        p_log, r_log = out["protein_logits"], out["rna_logits"]
    result = []
    p_active = payload.get("_selected_protein_active", payload["protein_active"])
    r_active = payload.get("_selected_rna_active", payload["rna_active"])
    for polymer, logp, native, interface, active in (("protein", p_log, payload["protein_native"], payload["protein_interface"], p_active), ("rna", r_log, payload["rna_native"], payload["rna_interface"], r_active)):
        native, interface, active = native.to(device), interface.to(device), active.to(device)
        mask = torch.ones(len(native), dtype=torch.bool, device=device)
        if polymer == "rna" and common_rna_positions is not None:
            mask = torch.tensor([(str(payload["sample_id"]), i) in common_rna_positions for i in range(len(native))], device=device)
        logp = F.log_softmax(logp, -1)
        nll = -logp[torch.arange(len(native), device=device), native]
        pred = logp.argmax(-1)
        for subset, subset_mask in (("all", mask), ("active", mask & active), ("interface", mask & interface)):
            if bool(subset_mask.any()):
                result.append({"sample_id": payload["sample_id"], "polymer": polymer, "subset": subset, "nll": float(nll[subset_mask].mean().cpu()), "recovery": float((pred[subset_mask] == native[subset_mask]).float().mean().cpu()), "token_off": token_off})
    return result


def run_evaluate(args) -> dict:
    cache_root = Path(args.cache)
    device = torch.device(args.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    selected = Path(args.checkpoint)
    payload = torch.load(selected, map_location="cpu", weights_only=False)
    config = AdapterConfig(**payload["config"])
    model = ReciprocalAdapter(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    test_cache = Path(args.test_cache)
    test = _load_cache(test_cache, "test")
    direction_specific = bool(payload.get("direction_specific", False))
    _attach_selected_edges(
        test,
        float(payload["radius"]),
        int(payload["neighbors"]),
        payload.get("r2p_neighbors"),
        payload.get("p2r_neighbors"),
        direction_specific,
    )
    common: set[tuple[str, int]] | None = None
    if args.common_tokens:
        df = pd.read_csv(args.common_tokens, sep="\t")
        df = df[df.polymer == "rna"]
        common = {(str(row.sample_id), int(row.position)) for row in df.itertuples(index=False)}
    rows_out = []
    for item in tqdm(test, desc="evaluate frozen test", unit="complex"):
        rows_out.extend(_evaluate_one(item, model, device, float(payload["radius"]), int(payload["neighbors"]), False, common))
        rows_out.extend(_evaluate_one(item, model, device, float(payload["radius"]), int(payload["neighbors"]), True, common))
    df = pd.DataFrame(rows_out)
    df.to_csv(Path(args.out) / "test_token_metrics.tsv", sep="\t", index=False)
    prior_rows = []
    for item in test:
        prior_rows.extend(_evaluate_one(item, None, device, float(payload["radius"]), int(payload["neighbors"]), False, common))
    prior = pd.DataFrame(prior_rows)
    prior.to_csv(Path(args.out) / "test_prior_token_metrics.tsv", sep="\t", index=False)
    merged = df.merge(prior, on=["sample_id", "polymer", "subset", "token_off"], suffixes=("_adapter", "_prior"))
    merged["delta_nll_prior_minus_adapter"] = merged.nll_prior - merged.nll_adapter
    merged["delta_recovery_adapter_minus_prior"] = merged.recovery_adapter - merged.recovery_prior
    merged.to_csv(Path(args.out) / "test_paired_metrics.tsv", sep="\t", index=False)
    native = merged[merged.token_off == False]
    token_off_rows = df[df.token_off == True].rename(columns={"nll": "nll_token_off", "recovery": "recovery_token_off"})
    native_vs_off = native.merge(token_off_rows, on=["sample_id", "polymer", "subset"], suffixes=("_native", "_token_off"))
    native_vs_off["delta_nll_token_off_minus_native"] = native_vs_off.nll_token_off - native_vs_off.nll_adapter
    bootstrap: dict[str, dict] = {}
    for polymer in ("protein", "rna"):
        for subset in ("all", "active", "interface"):
            key = f"{polymer}_{subset}"
            subset_rows = native[(native.polymer == polymer) & (native.subset == subset)]
            if not subset_rows.empty:
                bootstrap[f"{key}_nll"] = _paired_bootstrap(subset_rows, "nll_adapter", "nll_prior", 10000, 20260917)
                bootstrap[f"{key}_recovery"] = _paired_bootstrap(subset_rows, "recovery_adapter", "recovery_prior", 10000, 20260917)
    common_count = int(sum(len(item["rna_native"]) for item in test))
    summary = {"checkpoint": str(selected), "test_complexes": len(test), "common_rna_positions": common_count if common is None else len(common), "metrics": native.groupby(["polymer", "subset"])[["nll_adapter", "nll_prior", "recovery_adapter", "recovery_prior", "delta_nll_prior_minus_adapter", "delta_recovery_adapter_minus_prior"]].mean().reset_index().to_dict(orient="records"), "native_vs_token_off": native_vs_off.groupby(["polymer", "subset"])[["nll_adapter", "nll_token_off", "delta_nll_token_off_minus_native"]].mean().reset_index().to_dict(orient="records"), "paired_bootstrap": bootstrap, "test_used_only_after_selection": True}
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    def common(p):
        p.add_argument("--manifests", type=Path, default=DEFAULT_ROOT / "manifests" / "round_20260905_exception_v2")
        p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
        p.add_argument("--out", type=Path, default=DEFAULT_OUT)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--force", action="store_true")
    audit = sub.add_parser("audit"); common(audit); audit.add_argument("--workers", type=int, default=12); audit.set_defaults(func=run_audit)
    balance = sub.add_parser("balance"); common(balance); balance.add_argument("--cache", type=Path, required=True); balance.add_argument("--workers", type=int, default=12); balance.add_argument("--radius", type=float, default=14.357456359863283); balance.add_argument("--shared-k", type=int, default=51); balance.add_argument("--r2p-k", type=int, default=8); balance.add_argument("--p2r-k", type=int, default=12); balance.set_defaults(func=run_balance_report)
    prep = sub.add_parser("prepare"); common(prep); prep.add_argument("--checkouts", type=Path, default=DEFAULT_CHECKOUTS); prep.add_argument("--protein-checkpoint", type=Path, default=DEFAULT_P_CHECKPOINT); prep.add_argument("--rna-checkpoint", type=Path, default=DEFAULT_R_CHECKPOINT); prep.add_argument("--cache-radius", type=float, default=16.0); prep.add_argument("--cache-neighbors", type=int, default=32); prep.add_argument("--noise", type=float, default=0.0); prep.add_argument("--cache-dir", type=Path); prep.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val")); prep.set_defaults(func=run_prepare)
    screen = sub.add_parser("screen"); common(screen); screen.add_argument("--cache", type=Path, required=True); screen.add_argument("--radius", type=float); screen.add_argument("--neighbors", type=int); screen.add_argument("--lr", type=float, default=3e-4); screen.add_argument("--epochs", type=int, default=60); screen.add_argument("--patience", type=int, default=18); screen.add_argument("--batch-size", type=int, default=16); screen.set_defaults(func=run_screen)
    balance_train = sub.add_parser("balance-train"); common(balance_train); balance_train.add_argument("--cache", type=Path, required=True); balance_train.add_argument("--radius", type=float, default=14.357456359863283); balance_train.add_argument("--shared-k", type=int, default=51); balance_train.add_argument("--r2p-k", type=int, default=8); balance_train.add_argument("--p2r-k", type=int, default=12); balance_train.add_argument("--lr", type=float, default=3e-4); balance_train.add_argument("--epochs", type=int, default=60); balance_train.add_argument("--patience", type=int, default=18); balance_train.add_argument("--batch-size", type=int, default=16); balance_train.set_defaults(func=run_balance)
    ev = sub.add_parser("evaluate"); common(ev); ev.add_argument("--cache", type=Path, required=True); ev.add_argument("--test-cache", type=Path, required=True); ev.add_argument("--checkpoint", type=Path, required=True); ev.add_argument("--common-tokens", type=Path); ev.set_defaults(func=run_evaluate)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = arguments.func(arguments)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_jsonable), flush=True)
