#!/usr/bin/env python3
"""Prepare exact one-sided views of the frozen 100 Protein-RNA test complexes.

ProteinMPNN receives only the selected Protein chains; NA-MPNN receives only the
selected RNA chains. The script writes canonicalized PDB files plus a position map
from each baseline output index back to the original complex residue ID and the
DM-ICF interface label. No partner atoms are retained in either baseline view.
"""
from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import yaml

from pr_pilot.data.residue_vocab import classify_residue
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter, parse_chain_list
from pr_pilot.runtime.manifest_dataset import ManifestRow, load_complex_row


CHAIN_IDS = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)
AA1_TO_3 = {
    "A":"ALA", "R":"ARG", "N":"ASN", "D":"ASP", "C":"CYS", "Q":"GLN", "E":"GLU", "G":"GLY",
    "H":"HIS", "I":"ILE", "L":"LEU", "K":"LYS", "M":"MET", "F":"PHE", "P":"PRO", "S":"SER",
    "T":"THR", "W":"TRP", "Y":"TYR", "V":"VAL",
}


def _safe_name(sample_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)


def _best_atoms(residue: gemmi.Residue):
    chosen = {}
    for atom in residue:
        if atom.element.name == "H":
            continue
        xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
        if not np.isfinite(xyz).all():
            continue
        name = atom.name.strip()
        occ = float(atom.occ)
        if name not in chosen or occ > chosen[name][0]:
            chosen[name] = (occ, atom)
    return [item[1] for item in chosen.values()]


def _original_residue_id(chain_name: str, residue: gemmi.Residue) -> str:
    name = residue.name.strip().upper()
    return f"{chain_name}:{residue.seqid.num}{residue.seqid.icode.strip()}:{name}"


def _write_polymer_view(
    structure_path: Path,
    output_path: Path,
    polymer: str,
    selected_chains: list[str],
    interface_by_residue: dict[str, bool],
    expected_tokens: list[str],
) -> list[dict]:
    structure = gemmi.read_structure(str(structure_path))
    if len(structure) == 0:
        raise ValueError(f"No model in {structure_path}")
    selected = set(selected_chains)
    source_chains = [str(chain.name) for chain in structure[0] if str(chain.name) in selected]
    if set(source_chains) != selected:
        raise ValueError(f"Selected chains {selected_chains} not all found in {structure_path}")
    if len(source_chains) > len(CHAIN_IDS):
        raise ValueError("Too many chains for one-character PDB chain remapping")
    chain_remap = {source: CHAIN_IDS[i] for i, source in enumerate(source_chains)}

    lines = []
    mapping = []
    serial = 1
    baseline_position = 0
    observed_tokens = []
    for chain in structure[0]:
        source_chain = str(chain.name)
        if source_chain not in selected:
            continue
        target_chain = chain_remap[source_chain]
        target_resnum = 1
        for residue in chain:
            cls = classify_residue(residue.name)
            if cls.polymer != polymer or cls.token is None:
                if cls.polymer == "unsupported_polymer":
                    raise ValueError(
                        f"Unsupported polymer residue {residue.name} in selected {polymer} chain {source_chain}"
                    )
                continue
            canonical_resname = AA1_TO_3[cls.token] if polymer == "protein" else cls.token
            original_id = _original_residue_id(source_chain, residue)
            atoms = _best_atoms(residue)
            if not atoms:
                raise ValueError(f"No atoms for {original_id}")
            for atom in atoms:
                atom_name = atom.name.strip()
                x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                occ = float(atom.occ)
                b = float(atom.b_iso)
                element = atom.element.name.strip() or atom_name[:1]
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:>4s} {canonical_resname:>3s} {target_chain:1s}{target_resnum:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          {element:>2s}"
                )
                serial += 1
            mapping.append(
                {
                    "polymer": polymer,
                    "baseline_position": baseline_position,
                    "baseline_chain": target_chain,
                    "baseline_resnum": target_resnum,
                    "source_chain": source_chain,
                    "original_residue_id": original_id,
                    "token": cls.token,
                    "is_interface": bool(interface_by_residue.get(original_id, False)),
                }
            )
            observed_tokens.append(cls.token)
            baseline_position += 1
            target_resnum += 1
        lines.append("TER")
    lines.append("END")
    if observed_tokens != expected_tokens:
        raise ValueError(
            f"Baseline view sequence drift for {structure_path}: observed length {len(observed_tokens)} "
            f"!= expected length {len(expected_tokens)} or token order differs"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return mapping


def prepare_holdout(config_path: Path, manifest_path: Path, out_dir: Path) -> pd.DataFrame:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    g = cfg["geometry"]
    adapter = GemmiStructureAdapter(
        int(g["rbf_bins"]),
        int(g["intra_max_neighbors"]),
        float(g["pr_cutoff_angstrom"]),
        int(g["pr_max_neighbors"]),
        0.0,
        int(cfg["experiment"]["pilot_seed"]),
        bool(g["rich_pr_geometry"]),
    )
    manifest = pd.read_csv(manifest_path, sep=None, engine="python")
    out_dir.mkdir(parents=True, exist_ok=True)
    map_rows = []
    sample_rows = []
    for _, raw in manifest.iterrows():
        sample_id = str(raw.sample_id)
        row = ManifestRow(sample_id, Path(str(raw.structure_path)), raw.to_dict())
        complex_sample = load_complex_row(adapter, row)
        p_interface = dict(zip(complex_sample.protein.residue_ids, complex_sample.protein.interface.tolist()))
        r_interface = dict(zip(complex_sample.rna.residue_ids, complex_sample.rna.interface.tolist()))
        p_expected = ["ACDEFGHIKLMNPQRSTVWY"[int(x)] for x in complex_sample.protein.sequence]
        r_expected = ["AUGC"[int(x)] for x in complex_sample.rna.sequence]
        p_chains = parse_chain_list(raw.get("protein_chains")) or []
        r_chains = parse_chain_list(raw.get("rna_chains")) or []
        safe = _safe_name(sample_id)
        p_path = out_dir / "protein_pdb" / f"{safe}.pdb"
        r_path = out_dir / "rna_pdb" / f"{safe}.pdb"
        p_map = _write_polymer_view(row.structure_path, p_path, "protein", p_chains, p_interface, p_expected)
        r_map = _write_polymer_view(row.structure_path, r_path, "rna", r_chains, r_interface, r_expected)
        for item in p_map + r_map:
            map_rows.append({"sample_id": sample_id, **item})
        sample_rows.append(
            {
                "sample_id": sample_id,
                "protein_pdb": str(p_path.resolve()),
                "rna_pdb": str(r_path.resolve()),
                "protein_chain_ids": " ".join(sorted({x["baseline_chain"] for x in p_map})),
                "rna_chain_ids": "".join(sorted({x["baseline_chain"] for x in r_map})),
                "protein_length": len(p_map),
                "rna_length": len(r_map),
            }
        )
    mapping = pd.DataFrame(map_rows)
    samples = pd.DataFrame(sample_rows)
    mapping.to_csv(out_dir / "position_mapping.tsv", sep="\t", index=False)
    samples.to_csv(out_dir / "samples.tsv", sep="\t", index=False)
    summary = {
        "complexes": int(len(samples)),
        "protein_positions": int((mapping.polymer == "protein").sum()),
        "rna_positions": int((mapping.polymer == "rna").sum()),
        "partner_atoms_in_baseline_views": False,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frame = prepare_holdout(args.config, args.manifest, args.out)
    print(f"Prepared {len(frame)} one-sided holdout views -> {args.out}")


if __name__ == "__main__":
    main()
