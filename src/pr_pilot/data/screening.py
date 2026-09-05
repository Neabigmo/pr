"""Coordinate-level screening for the mini-pilot.

This module turns downloaded RCSB mmCIF files into auditable eligible tables.
Every rejection has an explicit reason. We intentionally screen locally rather
than assuming an RCSB metadata query proves that a usable interface exists.

Pilot policy:
- protein-only and RNA-only pools use one representative clean polymer chain per
  PDB entry so "1000 structures" means 1000 distinct PDB entries;
- complex samples use biological assembly 1 and retain all protein/RNA chains
  that participate in the Protein-RNA contact graph;
- DNA/NA-hybrid candidates are already excluded at discovery and any remaining
  canonical DNA residues trigger rejection;
- ribosome/spliceosome filtering is applied to complex samples;
- interface existence and interface backbone completeness are checked from
  coordinates, not titles alone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
from typing import Literal

import gemmi
import numpy as np
import pandas as pd


AA3_TO_1 = {
    "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C",
    "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I",
    "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P",
    "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V",
}
RNA_TO_1 = {"A":"A", "C":"C", "G":"G", "U":"U"}
DNA_NAMES = {"DA", "DC", "DG", "DT", "DU"}
PROTEIN_CORE = {"N", "CA", "C", "O"}
RNA_SUGAR_CORE = {"C1'", "C2'", "C3'", "C4'", "O4'"}
EXCLUDE_COMPLEX_KEYWORDS = (
    "ribosome", "ribosomal", "spliceosome", "spliceosomal", "pre-spliceosome",
)


@dataclass(frozen=True)
class ScreenConfig:
    protein_min_length: int = 30
    protein_max_length: int = 1000
    rna_min_length: int = 5
    rna_max_length: int = 500
    max_total_tokens: int = 1000
    max_resolution_angstrom: float = 4.0
    allow_nmr_without_resolution: bool = True
    interface_contact_angstrom: float = 6.0
    min_interfacial_residue_pairs: int = 3
    max_interface_missing_fraction: float = 0.10
    exclude_large_rnp_keywords: bool = True


@dataclass
class ResidueRecord:
    chain: str
    index: int
    name: str
    token: str
    atoms: set[str]
    heavy_xyz: np.ndarray


@dataclass
class ChainRecord:
    chain: str
    polymer: Literal["protein", "rna"]
    residues: list[ResidueRecord]

    @property
    def sequence(self) -> str:
        return "".join(r.token for r in self.residues)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _block_value(block: gemmi.cif.Block, tag: str) -> str:
    value = block.find_value(tag)
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value in {"?", "."} else value


def _metadata(path: Path) -> tuple[str, str, float | None, gemmi.Structure]:
    doc = gemmi.cif.read_file(str(path))
    block = doc.sole_block()
    title = " ".join([
        _block_value(block, "_struct.title"),
        _block_value(block, "_struct_keywords.pdbx_keywords"),
        _block_value(block, "_struct_keywords.text"),
    ]).strip()
    method_values = [str(x).strip() for x in block.find_values("_exptl.method")]
    method = ";".join(x for x in method_values if x and x not in {"?", "."})
    resolution = None
    for tag in ("_refine.ls_d_res_high", "_em_3d_reconstruction.resolution", "_reflns.d_resolution_high"):
        vals = []
        for raw in block.find_values(tag):
            try:
                x = float(str(raw))
                if math.isfinite(x) and x > 0:
                    vals.append(x)
            except ValueError:
                pass
        if vals:
            resolution = min(vals)
            break
    structure = gemmi.make_structure_from_block(block)
    return title, method, resolution, structure


def _heavy_xyz(residue: gemmi.Residue) -> tuple[set[str], np.ndarray]:
    names: set[str] = set()
    coords = []
    for atom in residue:
        name = atom.name.strip()
        names.add(name)
        if atom.element.name != "H":
            coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    arr = np.asarray(coords, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((0, 3), dtype=np.float32)
    return names, arr


def _chains(structure: gemmi.Structure) -> tuple[list[ChainRecord], list[ChainRecord], bool]:
    if len(structure) == 0:
        return [], [], False
    model = structure[0]
    proteins: list[ChainRecord] = []
    rnas: list[ChainRecord] = []
    has_dna = False
    for chain in model:
        protein_res: list[ResidueRecord] = []
        rna_res: list[ResidueRecord] = []
        for idx, residue in enumerate(chain):
            name = residue.name.strip().upper()
            if name in DNA_NAMES:
                has_dna = True
                continue
            if name in AA3_TO_1:
                atoms, xyz = _heavy_xyz(residue)
                protein_res.append(ResidueRecord(chain.name, idx, name, AA3_TO_1[name], atoms, xyz))
            elif name in RNA_TO_1:
                atoms, xyz = _heavy_xyz(residue)
                rna_res.append(ResidueRecord(chain.name, idx, name, RNA_TO_1[name], atoms, xyz))
        # Mixed protein/RNA in one chain is considered malformed for this pilot.
        if protein_res and rna_res:
            continue
        if protein_res:
            proteins.append(ChainRecord(chain.name, "protein", protein_res))
        elif rna_res:
            rnas.append(ChainRecord(chain.name, "rna", rna_res))
    return proteins, rnas, has_dna


def _passes_resolution(method: str, resolution: float | None, cfg: ScreenConfig) -> bool:
    if resolution is not None:
        return resolution <= cfg.max_resolution_angstrom
    return cfg.allow_nmr_without_resolution and "NMR" in method.upper()


def _min_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    # Residues are small; the direct broadcast is faster/simpler than building a
    # global KD-tree and keeps exact residue-pair distances auditable.
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).min())


def _interface_pairs(proteins: list[ChainRecord], rnas: list[ChainRecord], cutoff: float):
    pairs: list[tuple[ResidueRecord, ResidueRecord, float]] = []
    for pc in proteins:
        for rc in rnas:
            for pr in pc.residues:
                for rr in rc.residues:
                    d = _min_distance(pr.heavy_xyz, rr.heavy_xyz)
                    if d <= cutoff:
                        pairs.append((pr, rr, d))
    return pairs


def _interface_missing_fraction(pairs: list[tuple[ResidueRecord, ResidueRecord, float]]) -> float:
    p_unique = {(x.chain, x.index): x for x, _, _ in pairs}.values()
    r_unique = {(x.chain, x.index): x for _, x, _ in pairs}.values()
    missing = total = 0
    for residue in p_unique:
        total += len(PROTEIN_CORE)
        missing += len(PROTEIN_CORE - residue.atoms)
    for residue in r_unique:
        total += len(RNA_SUGAR_CORE)
        missing += len(RNA_SUGAR_CORE - residue.atoms)
    return 1.0 if total == 0 else missing / total


def _selected_contact_chains(pairs):
    p_names = {p.chain for p, _, _ in pairs}
    r_names = {r.chain for _, r, _ in pairs}
    return p_names, r_names


def _chain_json(chains: list[ChainRecord]) -> str:
    return json.dumps({c.chain: c.sequence for c in chains}, sort_keys=True)


def screen_file(path: Path, kind: Literal["protein", "rna", "complex"], cfg: ScreenConfig) -> tuple[dict | None, str]:
    """Return (eligible_record, rejection_reason)."""
    pdb_id = path.name.split("-")[0].split(".")[0].upper()
    try:
        title, method, resolution, structure = _metadata(path)
    except Exception as exc:
        return None, f"parse_error:{type(exc).__name__}"
    if not _passes_resolution(method, resolution, cfg):
        return None, "resolution_or_method"
    proteins, rnas, has_dna = _chains(structure)
    if has_dna:
        return None, "contains_DNA"

    common = {
        "pdb_id": pdb_id,
        "structure_path": str(path.resolve()),
        "experimental": True,
        "method": method,
        "resolution": resolution,
        "title": title,
    }

    if kind == "protein":
        if rnas:
            return None, "contains_RNA"
        eligible = [c for c in proteins if cfg.protein_min_length <= len(c.residues) <= cfg.protein_max_length]
        if not eligible:
            return None, "protein_length"
        chain = max(eligible, key=lambda c: (len(c.residues), c.chain))
        seq = chain.sequence
        return {
            **common,
            "sample_id": f"{pdb_id}:{chain.chain}",
            "chain_id": chain.chain,
            "sequence": seq,
            "sequence_hash": sha256_text(seq),
            "length": len(seq),
        }, ""

    if kind == "rna":
        if proteins:
            return None, "contains_protein"
        eligible = [c for c in rnas if cfg.rna_min_length <= len(c.residues) <= cfg.rna_max_length]
        if not eligible:
            return None, "rna_length"
        chain = max(eligible, key=lambda c: (len(c.residues), c.chain))
        seq = chain.sequence
        return {
            **common,
            "sample_id": f"{pdb_id}:{chain.chain}",
            "chain_id": chain.chain,
            "sequence": seq,
            "sequence_hash": sha256_text(seq),
            "length": len(seq),
        }, ""

    if kind != "complex":
        raise ValueError(kind)
    if cfg.exclude_large_rnp_keywords and any(k in title.lower() for k in EXCLUDE_COMPLEX_KEYWORDS):
        return None, "excluded_large_RNP_keyword"
    if not proteins or not rnas:
        return None, "missing_polymer_partner"
    pairs = _interface_pairs(proteins, rnas, cfg.interface_contact_angstrom)
    if len(pairs) < cfg.min_interfacial_residue_pairs:
        return None, "insufficient_PR_contact"
    p_names, r_names = _selected_contact_chains(pairs)
    proteins = [c for c in proteins if c.chain in p_names]
    rnas = [c for c in rnas if c.chain in r_names]
    p_len = sum(len(c.residues) for c in proteins)
    r_len = sum(len(c.residues) for c in rnas)
    if p_len < cfg.protein_min_length or r_len < cfg.rna_min_length:
        return None, "complex_chain_length"
    if p_len + r_len > cfg.max_total_tokens:
        return None, "complex_too_long"
    missing_fraction = _interface_missing_fraction(pairs)
    if missing_fraction > cfg.max_interface_missing_fraction:
        return None, "interface_missing_atoms"
    pseq = "|".join(c.sequence for c in sorted(proteins, key=lambda x: x.chain))
    rseq = "|".join(c.sequence for c in sorted(rnas, key=lambda x: x.chain))
    sample_id = f"{pdb_id}-assembly1"
    return {
        **common,
        "sample_id": sample_id,
        "mother_sample_id": sample_id,
        "protein_chains": ";".join(sorted(p_names)),
        "rna_chains": ";".join(sorted(r_names)),
        "protein_chain_sequences": _chain_json(proteins),
        "rna_chain_sequences": _chain_json(rnas),
        "protein_sequence": pseq,
        "rna_sequence": rseq,
        "protein_hash": sha256_text(pseq),
        "rna_hash": sha256_text(rseq),
        "protein_length": p_len,
        "rna_length": r_len,
        "total_tokens": p_len + r_len,
        "interface_residue_pairs": len(pairs),
        "interface_min_distance": min(d for _, _, d in pairs),
        "interface_missing_fraction": missing_fraction,
    }, ""


def screen_download_manifest(
    download_manifest: Path,
    kind: Literal["protein", "rna", "complex"],
    out_dir: Path,
    cfg: ScreenConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen every downloaded file and persist eligible/rejected tables."""
    downloads = pd.read_csv(download_manifest, sep="\t")
    eligible: list[dict] = []
    rejected: list[dict] = []
    for row in downloads.itertuples(index=False):
        record, reason = screen_file(Path(row.path), kind, cfg)
        if record is None:
            rejected.append({"pdb_id": row.pdb_id, "path": row.path, "reason": reason})
        else:
            eligible.append(record)
    out_dir.mkdir(parents=True, exist_ok=True)
    elig_df = pd.DataFrame(eligible)
    rej_df = pd.DataFrame(rejected)
    elig_df.to_csv(out_dir / f"{kind}_eligible.tsv", sep="\t", index=False)
    rej_df.to_csv(out_dir / f"{kind}_rejected.tsv", sep="\t", index=False)
    summary = {
        "kind": kind,
        "config": asdict(cfg),
        "downloaded": len(downloads),
        "eligible": len(elig_df),
        "rejected": len(rej_df),
        "rejection_counts": rej_df["reason"].value_counts().to_dict() if len(rej_df) else {},
    }
    (out_dir / f"{kind}_screen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return elig_df, rej_df
