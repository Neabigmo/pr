"""Independent heavy-atom amino-acid/nucleotide contact statistics.

This analysis is deliberately independent of the model PR graph: no neighbour
cap, no model edge cutoff, no learned representations. It parses the frozen
experimental structures and counts unique residue-nucleotide pairs whose heavy
atoms approach within the pre-registered cutoff. Contacts are additionally
stratified by the contacted RNA moiety (base, sugar, phosphate).

The resulting PMI is post-hoc validation only and must never initialize C.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import gemmi
import numpy as np
import pandas as pd

from pr_pilot.data.residue_vocab import PROTEIN_ALPHABET, RNA_ALPHABET, classify_residue
from pr_pilot.evaluation.battery import empirical_pmi, matrix_correlations
from pr_pilot.model.dmicf import double_center
from pr_pilot.runtime.gemmi_adapter import parse_chain_list

P_INDEX = {aa: i for i, aa in enumerate(PROTEIN_ALPHABET)}
R_INDEX = {b: i for i, b in enumerate(RNA_ALPHABET)}
PHOSPHATE_ATOMS = {"P", "OP1", "OP2", "O5'", "O3'"}
SUGAR_ATOMS = {"C1'", "C2'", "O2'", "C3'", "C4'", "O4'", "C5'"}
BACKBONE_ATOMS = PHOSPHATE_ATOMS | SUGAR_ATOMS


def _heavy_atoms(residue: gemmi.Residue) -> dict[str, np.ndarray]:
    chosen: dict[str, tuple[float, np.ndarray]] = {}
    for atom in residue:
        if atom.element.name == "H":
            continue
        xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
        if not np.isfinite(xyz).all():
            continue
        name = atom.name.strip()
        occ = float(atom.occ)
        if name not in chosen or occ > chosen[name][0]:
            chosen[name] = (occ, xyz)
    return {k: v[1] for k, v in chosen.items()}


def _selected_residues(path: Path, protein_chains: list[str] | None, rna_chains: list[str] | None):
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"No model in {path}")
    pkeep = None if protein_chains is None else set(protein_chains)
    rkeep = None if rna_chains is None else set(rna_chains)
    proteins = []
    rnas = []
    for chain in structure[0]:
        cname = str(chain.name)
        for residue in chain:
            cls = classify_residue(residue.name)
            if cls.token is None:
                continue
            atoms = _heavy_atoms(residue)
            if cls.polymer == "protein" and (pkeep is None or cname in pkeep):
                proteins.append((cname, str(residue.seqid), P_INDEX[cls.token], atoms))
            elif cls.polymer == "rna" and (rkeep is None or cname in rkeep):
                rnas.append((cname, str(residue.seqid), R_INDEX[cls.token], atoms))
    return proteins, rnas


def _minimum_by_moiety(protein_atoms: dict[str, np.ndarray], rna_atoms: dict[str, np.ndarray]) -> dict[str, float]:
    if not protein_atoms or not rna_atoms:
        return {"any": float("inf"), "base": float("inf"), "sugar": float("inf"), "phosphate": float("inf")}
    pxyz = np.stack(list(protein_atoms.values()))
    groups = {
        "base": [xyz for name, xyz in rna_atoms.items() if name not in BACKBONE_ATOMS],
        "sugar": [xyz for name, xyz in rna_atoms.items() if name in SUGAR_ATOMS],
        "phosphate": [xyz for name, xyz in rna_atoms.items() if name in PHOSPHATE_ATOMS],
    }
    result = {}
    all_r = np.stack(list(rna_atoms.values()))
    result["any"] = float(np.linalg.norm(pxyz[:, None, :] - all_r[None, :, :], axis=-1).min())
    for name, coords in groups.items():
        result[name] = (
            float(np.linalg.norm(pxyz[:, None, :] - np.stack(coords)[None, :, :], axis=-1).min())
            if coords
            else float("inf")
        )
    return result


def empirical_contact_tables(manifest_path: Path, cutoff: float = 5.0) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    manifest = pd.read_csv(manifest_path, sep=None, engine="python")
    required = {"sample_id", "structure_path", "protein_chains", "rna_chains", "experimental"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if not manifest["experimental"].astype(bool).all():
        raise ValueError("Empirical PMI must use experimental structures only")

    counts = {name: np.zeros((20, 4), dtype=np.int64) for name in ["any", "base", "sugar", "phosphate"]}
    rows = []
    for row in manifest.itertuples(index=False):
        proteins, rnas = _selected_residues(
            Path(row.structure_path),
            parse_chain_list(row.protein_chains),
            parse_chain_list(row.rna_chains),
        )
        for pchain, presid, aa, patoms in proteins:
            for rchain, rresid, base, ratoms in rnas:
                distances = _minimum_by_moiety(patoms, ratoms)
                contacted = []
                for stratum in ["any", "base", "sugar", "phosphate"]:
                    if distances[stratum] <= cutoff:
                        counts[stratum][aa, base] += 1
                        contacted.append(stratum)
                if "any" in contacted:
                    rows.append(
                        {
                            "sample_id": str(row.sample_id),
                            "protein_chain": pchain,
                            "protein_residue": presid,
                            "rna_chain": rchain,
                            "rna_residue": rresid,
                            "aa": aa,
                            "base": base,
                            "min_distance": distances["any"],
                            "base_distance": distances["base"],
                            "sugar_distance": distances["sugar"],
                            "phosphate_distance": distances["phosphate"],
                            "base_contact": "base" in contacted,
                            "sugar_contact": "sugar" in contacted,
                            "phosphate_contact": "phosphate" in contacted,
                        }
                    )
    return pd.DataFrame(rows), counts


def analyze_empirical_pmi(
    manifest_path: Path,
    c_path: Path,
    out_dir: Path,
    *,
    cutoff: float = 5.0,
    pseudocount: float = 0.5,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    contacts, counts = empirical_contact_tables(manifest_path, cutoff=cutoff)
    contacts.to_csv(out_dir / "heavy_atom_contacts.tsv", sep="\t", index=False)
    c_full = np.load(c_path)
    if c_full.shape != (20, 4):
        raise ValueError(f"C must be 20x4, got {c_full.shape}")
    c_interaction = double_center(torch_from_numpy(c_full)).numpy()
    results = {}
    for stratum, matrix in counts.items():
        pmi = empirical_pmi(matrix, pseudocount=pseudocount)
        pmi_interaction = double_center(torch_from_numpy(pmi)).numpy()
        np.save(out_dir / f"counts_{stratum}_20x4.npy", matrix)
        np.save(out_dir / f"pmi_{stratum}.npy", pmi)
        np.save(out_dir / f"pmi_{stratum}_interaction_only.npy", pmi_interaction)
        results[stratum] = {
            "n_unique_residue_pairs": int(matrix.sum()),
            "full_C_vs_PMI": matrix_correlations(c_full, pmi),
            "interaction_only_C_vs_centered_PMI": matrix_correlations(c_interaction, pmi_interaction),
        }
    summary = {
        "contact_cutoff_angstrom": float(cutoff),
        "pseudocount": float(pseudocount),
        "source": "full experimental heavy-atom residue pairs; independent of model PR graph",
        "strata": results,
    }
    (out_dir / "empirical_pmi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def torch_from_numpy(array: np.ndarray):
    # Local import keeps this module's top-level dependencies easy to inspect.
    import torch

    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True, help="C_global_centered.npy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--pseudocount", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(analyze_empirical_pmi(args.manifest, args.c, args.out, cutoff=args.cutoff, pseudocount=args.pseudocount), indent=2))


if __name__ == "__main__":
    main()
