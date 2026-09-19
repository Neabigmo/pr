#!/usr/bin/env python3
"""Audit that cached prior hidden states are encoder-side and sequence-free."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = path.read_text(encoding="utf-8").splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"function {name!r} not found in {path}")


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--na-checkout", type=Path, default=Path(r"F:\111临时\PR PILOT\third_party_checkouts_local_20260907\NA-MPNN"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    priors = repo / "src" / "pr_pilot" / "adapter_pilot" / "priors.py"
    na_utils = args.na_checkout / "inference" / "model_utils.py"
    sample_path = next(iter(sorted((args.cache_root / "train").glob("*.pt"))), None)
    if sample_path is None:
        raise FileNotFoundError(f"no cache sample in {args.cache_root / 'train'}")
    payload = torch.load(sample_path, map_location="cpu", weights_only=False, mmap=True)
    protein_encode = _function_source(priors, "_protein_encoded")
    na_encode = _function_source(na_utils, "encode")
    na_features = _function_source(na_utils, "forward")
    evidence = {
        "protein_encoder_calls_features_without_sequence": "model.features(X, mask, tensors[12], chain_encoding)" in protein_encode,
        "protein_structure_only_helper_exists": "def _protein_structure_encoded" in priors.read_text(encoding="utf-8"),
        "protein_encoder_decoder_not_used": "decoder_layers" not in _function_source(priors, "_protein_structure_encoded"),
        "na_encoder_calls_geometry_features": "V, E, E_idx = self.features(feature_dict)" in na_encode,
        "na_feature_forward_does_not_read_sequence": 'feature_dict["S"]' not in na_features and "feature_dict['S']" not in na_features,
        "cache_hidden_dim_protein": list(payload["protein_hidden"].shape[-1:]) == [128],
        "cache_hidden_dim_rna": list(payload["rna_hidden"].shape[-1:]) == [128],
        "cache_metadata_hidden_contract": payload.get("metadata", {}).get("hidden_contract") == "sequence_free_encoder_side",
        "cache_metadata_no_test_used": bool(payload.get("metadata", {}).get("no_test_used", False)),
    }
    result = {
        "contract": "sequence_free_encoder_side",
        "passed": bool(all(evidence.values())),
        "evidence": evidence,
        "sample_id": str(payload["sample_id"]),
        "sample_cache": str(sample_path),
        "source_hashes": {str(priors): _sha256(priors), str(na_utils): _sha256(na_utils)},
        "test_read": False,
        "note": "The cache stores encoder hidden states produced by the pinned prior wrappers; A2 never receives decoder or teacher-forced hidden states.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(2)
    return result


if __name__ == "__main__":
    main()
