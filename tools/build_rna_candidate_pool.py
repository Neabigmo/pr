#!/usr/bin/env python3
"""Build the RNA structural-prior candidate table before joint clustering.

The primary source is standalone experimental RNA. If that pool is too small,
we add RNA-chain views extracted from already screened experimental Protein-RNA
complexes. The protein partner is *not* exposed during RNA-prior training; only
the selected RNA chain is loaded by the structure adapter.

Important ordering:
1. standalone RNA and experimental complexes are screened independently;
2. this script creates a combined RNA candidate table;
3. `pr-pilot annotate` jointly clusters this combined table together with all
   complex chains, so R80/Rfam identifiers are directly comparable;
4. final 100 complex test is frozen;
5. all final-test RNA R80/Rfam neighbours are purged before the 1,000 RNA pool
   is sampled.

Thus using an RNA chain view from a complex does not leak the final holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_pool(standalone_path: Path, complexes_path: Path, out_path: Path) -> pd.DataFrame:
    standalone = pd.read_csv(standalone_path, sep=None, engine="python").copy()
    complexes = pd.read_csv(complexes_path, sep=None, engine="python").copy()
    required_standalone = {"sample_id", "structure_path", "sequence", "sequence_hash"}
    required_complex = {"sample_id", "structure_path", "rna_chain_sequences", "experimental"}
    missing = required_standalone - set(standalone.columns)
    if missing:
        raise ValueError(f"Standalone RNA table missing: {sorted(missing)}")
    missing = required_complex - set(complexes.columns)
    if missing:
        raise ValueError(f"Complex table missing: {sorted(missing)}")
    if not complexes["experimental"].astype(bool).all():
        raise ValueError("RNA prior fallback may use experimental complexes only")

    standalone["source_view"] = "standalone_rna"
    if "rna_chains" not in standalone:
        standalone["rna_chains"] = standalone.get("chain_id", "")

    extracted = []
    for row in complexes.itertuples(index=False):
        chain_map = json.loads(str(row.rna_chain_sequences))
        if not isinstance(chain_map, dict) or not chain_map:
            raise ValueError(f"Invalid rna_chain_sequences for {row.sample_id}")
        for chain, sequence in sorted(chain_map.items()):
            sequence = str(sequence).upper().replace("T", "U")
            if not sequence or set(sequence) - set("AUGC"):
                raise ValueError(f"Non-canonical extracted RNA sequence: {row.sample_id}:{chain}")
            extracted.append(
                {
                    "sample_id": f"{row.sample_id}::RNA::{chain}",
                    "structure_path": str(row.structure_path),
                    "sequence": sequence,
                    "sequence_hash": _sha(sequence),
                    "length": len(sequence),
                    "rna_chains": str(chain),
                    "chain_id": str(chain),
                    "source_view": "protein_rna_complex_chain_extracted",
                    "source_complex_sample_id": str(row.sample_id),
                    "pdb_id": getattr(row, "pdb_id", ""),
                    "experimental": True,
                }
            )
    extracted_df = pd.DataFrame(extracted)

    keep_columns = sorted(set(standalone.columns) | set(extracted_df.columns))
    combined = pd.concat(
        [standalone.reindex(columns=keep_columns), extracted_df.reindex(columns=keep_columns)],
        ignore_index=True,
    )
    # Exact sequence duplication across structures is allowed at this candidate
    # stage because structural conformers may differ. Stable sample IDs must not.
    if combined["sample_id"].astype(str).duplicated().any():
        duplicated = combined.loc[combined["sample_id"].astype(str).duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate RNA sample IDs: {duplicated[:5]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, sep="\t", index=False)
    summary = {
        "standalone_candidates": int(len(standalone)),
        "complex_chain_views": int(len(extracted_df)),
        "combined_candidates": int(len(combined)),
        "unique_sequences": int(combined["sequence_hash"].nunique()),
        "policy": "standalone preferred conceptually; extracted experimental-complex RNA chains are partner-hidden structural-prior views and are jointly clustered before strict-test purge",
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", type=Path, required=True)
    parser.add_argument("--complexes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = build_pool(args.standalone, args.complexes, args.out)
    print(f"RNA structural-prior candidates: {len(frame)} -> {args.out}")


if __name__ == "__main__":
    main()
