#!/usr/bin/env python3
"""Prepare the exact frozen 900/100 pools for official ProteinMPNN and NA-MPNN.

This converter deliberately shares the project's canonical residue vocabulary.
A modified residue accepted by screening (for example a CCD monomer mapping to a
canonical amino acid/base) must therefore map to the same supervised token for
DM-ICF and the official baseline. Unsupported target-polymer monomers fail loudly
instead of silently shortening one method's sequence.

Primary baseline policy:
- random initialization;
- identical frozen 900/100 sample IDs;
- 0.10 A coordinate-noise scale;
- approximately 150 dataset passes;
- published checkpoints reported only as a separate reference track.
"""
from __future__ import annotations

import argparse
import csv
import json
import string
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import torch

from pr_pilot.data.residue_vocab import classify_residue
from pr_pilot.runtime.gemmi_adapter import parse_chain_list

PROTEIN_ATOM_ORDER = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2"
]


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def _residue_atom_dict(residue: gemmi.Residue) -> dict[str, tuple[np.ndarray, float, float]]:
    """Return highest-occupancy coordinates per atom name."""
    chosen: dict[str, tuple[np.ndarray, float, float]] = {}
    for atom in residue:
        name = atom.name.strip()
        occupancy = float(atom.occ)
        xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
        bfactor = float(atom.b_iso)
        if not np.isfinite(xyz).all():
            continue
        if name not in chosen or occupancy > chosen[name][1]:
            chosen[name] = (xyz, occupancy, bfactor)
    return chosen


def extract_matching_chain(
    path: Path,
    expected_sequence: str,
    polymer: str,
    explicit_chains: list[str] | None = None,
):
    """Find the unique selected chain whose canonicalized sequence equals manifest.

    An unsupported polymer residue inside an explicitly selected candidate chain is
    a hard error: dropping it would make the baseline receive a different structure
    from DM-ICF while still claiming identical IDs.
    """
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"No model in {path}")
    candidates = []
    selected = None if explicit_chains is None else set(explicit_chains)
    for chain in structure[0]:
        chain_name = str(chain.name)
        if selected is not None and chain_name not in selected:
            continue
        sequence = []
        residues = []
        saw_target_polymer = False
        for residue in chain:
            cls = classify_residue(residue.name)
            if cls.polymer == polymer:
                saw_target_polymer = True
                if cls.token is None:
                    raise ValueError(
                        f"Unsupported {polymer} residue {residue.name} in selected chain "
                        f"{chain_name} of {path}"
                    )
                sequence.append(cls.token)
                residues.append(residue)
            elif cls.polymer == "unsupported_polymer" and saw_target_polymer:
                raise ValueError(
                    f"Unsupported polymer residue {residue.name} inside candidate chain {chain_name} of {path}"
                )
        if sequence:
            candidates.append((chain_name, "".join(sequence), residues))

    exact = [item for item in candidates if item[1] == expected_sequence]
    if len(exact) != 1:
        description = [(chain, seq[:30], len(seq)) for chain, seq, _ in candidates]
        raise ValueError(
            f"Expected exactly one {polymer} chain matching frozen manifest sequence in {path}; "
            f"expected_length={len(expected_sequence)} candidates={description}"
        )
    return exact[0]


def synth_pdb_id(index: int) -> str:
    alphabet = string.digits + string.ascii_lowercase
    value = index
    chars = []
    for _ in range(4):
        chars.append(alphabet[value % 36])
        value //= 36
    return "".join(reversed(chars))


def prepare_proteinmpnn(train_path: Path, val_path: Path, out: Path) -> dict:
    """Convert exactly frozen Protein structures to the official training layout."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    mapping = []
    counter = 0
    val_clusters = []

    for split, dataframe in [("train", read_manifest(train_path)), ("val", read_manifest(val_path))]:
        for _, row in dataframe.iterrows():
            sequence = str(row["sequence"]).strip().upper()
            chains = parse_chain_list(row.get("protein_chains", row.get("chain_id", row.get("chains"))))
            chain_name, obtained, residues = extract_matching_chain(
                Path(str(row["structure_path"])), sequence, "protein", chains
            )
            if obtained != sequence:
                raise AssertionError("Canonical baseline sequence drift")

            pdb_id = synth_pdb_id(counter + 1)
            synthetic_chain = f"{pdb_id}_A"
            cluster = counter + 1 if split == "train" else 100000 + counter + 1
            prefix = out / "pdb" / pdb_id[1:3] / pdb_id
            prefix.parent.mkdir(parents=True, exist_ok=True)

            length = len(residues)
            xyz = np.zeros((length, 14, 3), dtype=np.float32)
            mask = np.zeros((length, 14), dtype=np.float32)
            bfac = np.zeros((length, 14), dtype=np.float32)
            occ = np.zeros((length, 14), dtype=np.float32)
            for residue_index, residue in enumerate(residues):
                atoms = _residue_atom_dict(residue)
                for atom_index, name in enumerate(PROTEIN_ATOM_ORDER):
                    if name in atoms:
                        xyz[residue_index, atom_index] = atoms[name][0]
                        mask[residue_index, atom_index] = 1.0
                        occ[residue_index, atom_index] = atoms[name][1]
                        bfac[residue_index, atom_index] = atoms[name][2]
                if not bool(mask[residue_index, :4].all()):
                    raise ValueError(
                        f"ProteinMPNN requires N/CA/C/O: {row['sample_id']} residue {residue_index}"
                    )

            torch.save(
                {
                    "seq": sequence,
                    "xyz": torch.from_numpy(xyz),
                    "mask": torch.from_numpy(mask),
                    "bfac": torch.from_numpy(bfac),
                    "occ": torch.from_numpy(occ),
                },
                str(prefix) + "_A.pt",
            )
            torch.save(
                {
                    "method": "pilot",
                    "date": "2000-01-01",
                    "resolution": 1.0,
                    "chains": ["A"],
                    "tm": np.array([[[1.0, 1.0, 0.0]]], dtype=np.float32).reshape(1, 1, 3),
                    "asmb_ids": [],
                    "asmb_details": [],
                    "asmb_method": [],
                    "asmb_chains": [],
                },
                str(prefix) + ".pt",
            )
            rows.append([synthetic_chain, "2000-01-01", "1.0", f"hash{counter}", str(cluster), sequence])
            if split == "val":
                val_clusters.append(cluster)
            mapping.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "synthetic_chain": synthetic_chain,
                    "source_chain": chain_name,
                    "split": split,
                    "cluster": cluster,
                    "sequence_hash": str(row.get("sequence_hash", "")),
                }
            )
            counter += 1

    with (out / "list.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["CHAINID", "DEPOSITION", "RESOLUTION", "HASH", "CLUSTER", "SEQUENCE"])
        writer.writerows(rows)
    (out / "valid_clusters.txt").write_text("\n".join(map(str, val_clusters)) + "\n", encoding="utf-8")
    (out / "test_clusters.txt").write_text("", encoding="utf-8")
    pd.DataFrame(mapping).to_csv(out / "sample_mapping.tsv", sep="\t", index=False)
    return {
        "train": sum(item["split"] == "train" for item in mapping),
        "val": sum(item["split"] == "val" for item in mapping),
        "conversion_failures": 0,
    }


def write_rna_only_pdb(path: Path, residues, sequence: str) -> None:
    """Write canonicalized RNA PDB while preserving coordinates from source residues."""
    if len(residues) != len(sequence):
        raise ValueError("RNA residue/sequence length mismatch")
    lines = []
    serial = 1
    for residue_index, (residue, base) in enumerate(zip(residues, sequence), start=1):
        atoms = _residue_atom_dict(residue)
        for atom_name, (xyz, occupancy, bfactor) in atoms.items():
            element = "".join(char for char in atom_name if char.isalpha())[:1].upper() or "C"
            x, y, z = map(float, xyz)
            lines.append(
                f"ATOM  {serial:5d} {atom_name:>4s} {base:>3s} A{residue_index:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}"
            )
            serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rna_preprocessed_paths(root: Path, stem: str, length: int) -> dict:
    preprocessed = root / "preprocessed"
    preprocessed.mkdir(parents=True, exist_ok=True)
    key = "1"
    zeros = np.zeros(length, dtype=np.int32)
    zeros64 = np.zeros(length, dtype=np.int64)
    values = {
        "asmb_lengths_path": {key: (length, 0, 0, length)},
        "asmb_interface_masks_path": {key: zeros},
        "asmb_side_chain_interface_masks_path": {key: zeros},
        "asmb_nearest_protein_side_chain_index_path": {key: zeros64},
        "asmb_base_pair_masks_path": {key: zeros},
        "asmb_base_pair_index_path": {key: zeros64},
        "asmb_canonical_base_pair_masks_path": {key: zeros},
        "asmb_canonical_base_pair_index_path": {key: zeros64},
    }
    result = {}
    for name, obj in values.items():
        path = preprocessed / f"{stem}.{name}.npy"
        np.save(path, obj)
        result[name] = str(path.resolve())
    return result


def _estimate_na_steps(lengths: list[int], batch_tokens: int, passes: int) -> int:
    """Approximate official StructureLoader batches for equal dataset-pass budget."""
    batch = []
    clusters = 0
    for size in sorted(lengths):
        if size > batch_tokens:
            raise ValueError(
                f"RNA length {size} exceeds baseline BATCH_TOKENS={batch_tokens}; do not silently drop samples"
            )
        if size * (len(batch) + 1) <= batch_tokens:
            batch.append(size)
        else:
            if batch:
                clusters += 1
            batch = [size]
    if batch:
        clusters += 1
    return max(1, clusters * passes)


def prepare_nampnn(
    train_path: Path,
    val_path: Path,
    out: Path,
    passes: int = 150,
    batch_tokens: int = 6000,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    pdb_dir = out / "rna_pdb"
    pdb_dir.mkdir(exist_ok=True)
    outputs = {}
    train_lengths = []

    for split, path in [("train", train_path), ("valid", val_path)]:
        dataframe = read_manifest(path)
        rows = []
        for index, row in dataframe.iterrows():
            sequence = str(row["sequence"]).strip().upper().replace("T", "U")
            chains = parse_chain_list(row.get("rna_chains", row.get("chain_id", row.get("chains"))))
            chain_name, obtained, residues = extract_matching_chain(
                Path(str(row["structure_path"])), sequence, "rna", chains
            )
            if obtained != sequence:
                raise AssertionError("Canonical RNA baseline sequence drift")
            stem = f"{split}_{index:04d}_{row['sample_id']}"
            pdb_path = (pdb_dir / f"{stem}.pdb").resolve()
            write_rna_only_pdb(pdb_path, residues, sequence)
            record = {
                "id": str(row["sample_id"]),
                "structure_path": str(pdb_path),
                "date": "2000-01-01",
                "sampling_probability": 1.0,
                "ppm_paths": "[]",
                "source_chain": chain_name,
            }
            record.update(_rna_preprocessed_paths(out, stem, len(sequence)))
            rows.append(record)
            if split == "train":
                train_lengths.append(len(sequence))
        out_csv = out / f"{split}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        outputs[split] = str(out_csv.resolve())

    total_steps = _estimate_na_steps(train_lengths, batch_tokens, passes)
    config = {
        "VOCAB_SIZE": 33,
        "NUM_LETTERS": 33,
        "PARSE_PROTEIN": 0,
        "PARSE_DNA": 0,
        "PARSE_RNA": 1,
        "PARSE_RNA_AS_DNA": 0,
        "NA_SHARED_TOKENS": 1,
        "NA_REF_ATOM": "C1'",
        "INCLUDE_PRED_NA_N": 1,
        "PROTEIN_BACKBONE_OCC_CUTOFF": 0.8,
        "PROTEIN_SIDE_CHAIN_OCC_CUTOFF": 0.5,
        "DNA_BACKBONE_OCC_CUTOFF": 0.8,
        "DNA_SIDE_CHAIN_OCC_CUTOFF": 0.5,
        "RNA_BACKBONE_OCC_CUTOFF": 0.8,
        "RNA_SIDE_CHAIN_OCC_CUTOFF": 0.5,
        "EXCLUDED_ELEMENTS": [1],
        "DATE_CUTOFF": "2030-01-01",
        "MAX_NUMBER_OF_PDBS_TRAIN": len(read_manifest(train_path)),
        "MAX_NUMBER_OF_PDBS_VALID": len(read_manifest(val_path)),
        "BATCH_TOKENS": batch_tokens,
        "LOSS_TOKENS": batch_tokens,
        # Preserve the official NA-MPNN training default. This external reference
        # is not the same-data causal control for DM-ICF, so we do not silently
        # alter the upstream objective to imitate our internal label smoothing.
        "LABEL_SMOOTHING": 0.1,
        "EXCLUDE_RES": ["HOH", "NA", "CL", "K", "BR"],
        "MIN_PROTEIN_LENGTH_CUTOFF": 1,
        "NUM_WORKERS": 4,
        "TOTAL_STEPS": total_steps,
        "RANDOMIZE_NMR_MODEL": 0,
        "CROP_LARGE_STRUCTURES": 0,
        "MIN_OVERLAP_LENGTH": 5,
        "DF_PATH_TRAIN": outputs["train"],
        "DF_PATH_VALID": outputs["valid"],
        "BASE_FOLDER": str((out / "model").resolve()),
        "PREV_CHECKPOINT": "",
        "HIDDEN_DIM": 128,
        "NUM_ENCODER_LAYERS": 3,
        "NUM_DECODER_LAYERS": 3,
        "NUM_NEIGHBORS": 32,
        "DROPOUT": 0.1,
        "DECODE_PROTEIN_FIRST": 0,
        "PROTEIN_BACKBONE_NOISE": 0.1,
        "DNA_BACKBONE_NOISE": 0.1,
        "RNA_BACKBONE_NOISE": 0.1,
        "PARSE_PPMS": 0,
        "NA_ONLY_AS_UNIFORM_PPM": 0,
        "DROP_PROTEIN_PROBABILITY": 0,
        "PROTEIN_INTERFACE_RESIDUE_MUTATION_PROBABILITY": 0,
        "MUTATE_BASE_PAIR_TOGETHER": 0,
        "MUTATE_ENTIRE_SIDE_CHAIN_INTERFACE_PROBABILITY": 0,
        "NA_NON_INTERFACE_AS_UNIFORM_PPM": 0,
        "GRADIENT_NORM": 1.0,
        "MIXED_PRECISION": 1,
        "SAVE_EVERY_N_STEPS": max(1, total_steps // 20),
        "ATOMS_TO_LOAD": "backbone",
        "METRICS_TO_COMPUTE": "basic",
    }
    cfg_path = out / "na_mpnn_from_scratch.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {
        "train": len(read_manifest(train_path)),
        "val": len(read_manifest(val_path)),
        "passes": passes,
        "estimated_total_steps": total_steps,
        "config": str(cfg_path),
        "conversion_failures": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-train", type=Path, required=True)
    parser.add_argument("--protein-val", type=Path, required=True)
    parser.add_argument("--rna-train", type=Path, required=True)
    parser.add_argument("--rna-val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=150)
    args = parser.parse_args()
    result = {
        "ProteinMPNN": prepare_proteinmpnn(
            args.protein_train,
            args.protein_val,
            args.out / "proteinmpnn",
        ),
        "NA-MPNN": prepare_nampnn(
            args.rna_train,
            args.rna_val,
            args.out / "na_mpnn",
            args.passes,
        ),
    }
    (args.out / "baseline_preparation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
