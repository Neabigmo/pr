"""Coordinate-level screening for the mini-pilot.

Every rejection is explicit and auditable. Screening and training share the same
canonical residue vocabulary, so a structure cannot pass QC with one sequence
and later be silently shortened by the tensor adapter.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
from time import perf_counter
from typing import Literal

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from pr_pilot.data.residue_vocab import classify_residue


PROTEIN_CORE = {"N", "CA", "C", "O"}
RNA_SUGAR_CORE = {"C1'", "C2'", "C3'", "C4'", "O4'"}
EXCLUDE_COMPLEX_KEYWORDS = (
    "ribosome",
    "ribosomal",
    "spliceosome",
    "spliceosomal",
    "pre-spliceosome",
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
    apply_resolution_method_filter: bool = True
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
    modified: bool
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

    @property
    def modified_fraction(self) -> float:
        return sum(r.modified for r in self.residues) / max(1, len(self.residues))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _block_value(block: gemmi.cif.Block, tag: str) -> str:
    value = block.find_value(tag)
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value in {"?", "."} else value


def _metadata_header(path: Path) -> tuple[str, str, float | None, gemmi.cif.Block]:
    doc = gemmi.cif.read_file(str(path))
    block = doc.sole_block()
    title = " ".join(
        [
            _block_value(block, "_struct.title"),
            _block_value(block, "_struct_keywords.pdbx_keywords"),
            _block_value(block, "_struct_keywords.text"),
        ]
    ).strip()
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
    return title, method, resolution, block


def _metadata(path: Path) -> tuple[str, str, float | None, gemmi.Structure]:
    """Read metadata and materialize coordinates for callers needing a structure."""
    title, method, resolution, block = _metadata_header(path)
    return title, method, resolution, gemmi.make_structure_from_block(block)


def _heavy_xyz(residue: gemmi.Residue) -> tuple[set[str], np.ndarray]:
    names: set[str] = set()
    chosen: dict[str, tuple[float, np.ndarray]] = {}
    for atom in residue:
        name = atom.name.strip()
        xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
        if not np.isfinite(xyz).all():
            continue
        occ = float(atom.occ)
        names.add(name)
        if atom.element.name != "H" and (name not in chosen or occ > chosen[name][0]):
            chosen[name] = (occ, xyz)
    arr = np.asarray([v[1] for v in chosen.values()], dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((0, 3), dtype=np.float32)
    return names, arr


def _chains(structure: gemmi.Structure) -> tuple[list[ChainRecord], list[ChainRecord], bool, bool]:
    if len(structure) == 0:
        return [], [], False, False
    model = structure[0]
    proteins: list[ChainRecord] = []
    rnas: list[ChainRecord] = []
    has_dna = False
    has_unsupported_polymer = False
    for chain in model:
        protein_res: list[ResidueRecord] = []
        rna_res: list[ResidueRecord] = []
        chain_has_unsupported = False
        for idx, residue in enumerate(chain):
            name = residue.name.strip().upper()
            cls = classify_residue(name)
            if cls.polymer == "dna":
                has_dna = True
                continue
            if cls.polymer == "unsupported_polymer":
                chain_has_unsupported = True
                continue
            if cls.polymer not in {"protein", "rna"} or cls.token is None:
                continue
            atoms, xyz = _heavy_xyz(residue)
            rec = ResidueRecord(chain.name, idx, name, cls.token, cls.modified, atoms, xyz)
            if cls.polymer == "protein":
                protein_res.append(rec)
            else:
                rna_res.append(rec)
        if chain_has_unsupported and (protein_res or rna_res):
            has_unsupported_polymer = True
        # A polymer chain classified simultaneously as protein and RNA is malformed
        # for this pilot and therefore treated as unsupported.
        if protein_res and rna_res:
            has_unsupported_polymer = True
            continue
        if protein_res:
            proteins.append(ChainRecord(chain.name, "protein", protein_res))
        elif rna_res:
            rnas.append(ChainRecord(chain.name, "rna", rna_res))
    return proteins, rnas, has_dna, has_unsupported_polymer


def _passes_resolution(method: str, resolution: float | None, cfg: ScreenConfig) -> bool:
    if resolution is not None:
        return resolution <= cfg.max_resolution_angstrom
    return cfg.allow_nmr_without_resolution and "NMR" in method.upper()


def _min_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return float(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1).min())


def _interface_pairs(proteins: list[ChainRecord], rnas: list[ChainRecord], cutoff: float):
    """Return contacting residue pairs using an exact spatial-indexed search.

    The KD-tree is only a broad phase. Every candidate residue pair is sent
    through the original ``_min_distance`` calculation, so the returned pairs
    and distances retain the previous screening semantics.
    """
    protein_residues = [residue for chain in proteins for residue in chain.residues]
    rna_residues = [residue for chain in rnas for residue in chain.residues]
    protein_points: list[np.ndarray] = []
    protein_residue_ids: list[int] = []
    rna_points: list[np.ndarray] = []
    rna_residue_ids: list[int] = []
    for residue_id, residue in enumerate(protein_residues):
        for point in residue.heavy_xyz:
            protein_points.append(point)
            protein_residue_ids.append(residue_id)
    for residue_id, residue in enumerate(rna_residues):
        for point in residue.heavy_xyz:
            rna_points.append(point)
            rna_residue_ids.append(residue_id)
    if not protein_points or not rna_points:
        return []

    protein_xyz = np.asarray(protein_points, dtype=np.float32)
    rna_xyz = np.asarray(rna_points, dtype=np.float32)
    tree = cKDTree(rna_xyz)
    # Include the next representable radius as a conservative broad phase;
    # the original float32 distance test remains the final authority.
    search_cutoff = float(np.nextafter(float(cutoff), float("inf")))
    candidate_pairs: set[tuple[int, int]] = set()
    chunk_size = 8192
    for start in range(0, len(protein_xyz), chunk_size):
        stop = min(start + chunk_size, len(protein_xyz))
        neighbours = tree.query_ball_point(protein_xyz[start:stop], search_cutoff, eps=0.0)
        for offset, rna_atom_ids in enumerate(neighbours):
            protein_residue_id = protein_residue_ids[start + offset]
            candidate_pairs.update(
                (protein_residue_id, rna_residue_ids[rna_atom_id])
                for rna_atom_id in rna_atom_ids
            )

    pairs: list[tuple[ResidueRecord, ResidueRecord, float]] = []
    for protein_residue_id, rna_residue_id in sorted(candidate_pairs):
        protein_residue = protein_residues[protein_residue_id]
        rna_residue = rna_residues[rna_residue_id]
        distance = _min_distance(protein_residue.heavy_xyz, rna_residue.heavy_xyz)
        if distance <= cutoff:
            pairs.append((protein_residue, rna_residue, distance))
    return pairs


def _missing_fraction(residues, required: set[str]) -> float:
    residues = list(residues)
    if not residues:
        return 1.0
    missing = sum(len(required - residue.atoms) for residue in residues)
    return missing / (len(residues) * len(required))


def _reference_complete(chain: ChainRecord) -> bool:
    if chain.polymer == "protein":
        # ProteinMPNN baseline preparation requires all four backbone atoms.
        return all(PROTEIN_CORE <= r.atoms for r in chain.residues)
    # RNA adapter needs a stable sugar reference and frame. P may legitimately be
    # absent at a terminal nucleotide, so it is not an all-residue hard requirement.
    return all({"C1'", "C3'", "C4'"} <= r.atoms for r in chain.residues)


def _interface_missing_fraction(pairs: list[tuple[ResidueRecord, ResidueRecord, float]]) -> float:
    p_unique = {(x.chain, x.index): x for x, _, _ in pairs}.values()
    r_unique = {(x.chain, x.index): x for _, x, _ in pairs}.values()
    p_missing = _missing_fraction(p_unique, PROTEIN_CORE)
    r_missing = _missing_fraction(r_unique, RNA_SUGAR_CORE)
    return 0.5 * (p_missing + r_missing)


def _selected_contact_chains(pairs):
    return {p.chain for p, _, _ in pairs}, {r.chain for _, r, _ in pairs}


def _chain_json(chains: list[ChainRecord]) -> str:
    return json.dumps({c.chain: c.sequence for c in chains}, sort_keys=True)


def screen_file(path: Path, kind: Literal["protein", "rna", "complex"], cfg: ScreenConfig) -> tuple[dict | None, str]:
    """Return ``(eligible_record, rejection_reason)`` for one coordinate file."""
    pdb_id = path.name.split("-")[0].split(".")[0].upper()
    try:
        title, method, resolution, block = _metadata_header(path)
    except Exception as exc:
        return None, f"parse_error:{type(exc).__name__}"
    if cfg.apply_resolution_method_filter and not _passes_resolution(method, resolution, cfg):
        return None, "resolution_or_method"
    if kind == "complex" and cfg.exclude_large_rnp_keywords and any(
        keyword in title.lower() for keyword in EXCLUDE_COMPLEX_KEYWORDS
    ):
        return None, "excluded_large_RNP_keyword"
    try:
        # Use the same file parser as the runtime adapter.  Gemmi's block
        # materializer can omit unsupported polymer residues, which would let
        # screening accept a chain that later fails during training.
        structure = gemmi.read_structure(str(path))
    except Exception as exc:
        return None, f"parse_error:{type(exc).__name__}"
    proteins, rnas, has_dna, has_unsupported = _chains(structure)
    if has_dna:
        return None, "contains_DNA"
    if has_unsupported:
        return None, "unsupported_modified_polymer"

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
        eligible = [
            c
            for c in proteins
            if cfg.protein_min_length <= len(c.residues) <= cfg.protein_max_length and _reference_complete(c)
        ]
        if not eligible:
            return None, "protein_length_or_backbone_completeness"
        chain = max(eligible, key=lambda c: (len(c.residues), c.chain))
        seq = chain.sequence
        return {
            **common,
            "sample_id": f"{pdb_id}:{chain.chain}",
            "chain_id": chain.chain,
            "protein_chains": chain.chain,
            "sequence": seq,
            "sequence_hash": sha256_text(seq),
            "length": len(seq),
            "modified_fraction": chain.modified_fraction,
        }, ""

    if kind == "rna":
        if proteins:
            return None, "contains_protein"
        eligible = [
            c
            for c in rnas
            if cfg.rna_min_length <= len(c.residues) <= cfg.rna_max_length and _reference_complete(c)
        ]
        if not eligible:
            return None, "rna_length_or_backbone_completeness"
        chain = max(eligible, key=lambda c: (len(c.residues), c.chain))
        seq = chain.sequence
        return {
            **common,
            "sample_id": f"{pdb_id}:{chain.chain}",
            "chain_id": chain.chain,
            "rna_chains": chain.chain,
            "sequence": seq,
            "sequence_hash": sha256_text(seq),
            "length": len(seq),
            "modified_fraction": chain.modified_fraction,
        }, ""

    if kind != "complex":
        raise ValueError(kind)
    if not proteins or not rnas:
        return None, "missing_polymer_partner"
    pairs = _interface_pairs(proteins, rnas, cfg.interface_contact_angstrom)
    if len(pairs) < cfg.min_interfacial_residue_pairs:
        return None, "insufficient_PR_contact"
    p_names, r_names = _selected_contact_chains(pairs)
    proteins = [c for c in proteins if c.chain in p_names]
    rnas = [c for c in rnas if c.chain in r_names]
    if not all(_reference_complete(c) for c in proteins + rnas):
        return None, "selected_chain_reference_atom_missing"
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
        "protein_modified_fraction": sum(c.modified_fraction * len(c.residues) for c in proteins) / p_len,
        "rna_modified_fraction": sum(c.modified_fraction * len(c.residues) for c in rnas) / r_len,
    }, ""


def screen_download_manifest(
    download_manifest: Path,
    kind: Literal["protein", "rna", "complex"],
    out_dir: Path,
    cfg: ScreenConfig,
    *,
    progress_log: Path | None = None,
    progress_label: str | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen every downloaded file and persist eligible/rejected tables."""
    downloads = pd.read_csv(download_manifest, sep="\t")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_log = progress_log or out_dir / f"{kind}_progress.jsonl"
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    eligible: list[dict] = []
    rejected: list[dict] = []
    rows = downloads.itertuples(index=False)
    if show_progress:
        rows = tqdm(
            rows,
            total=len(downloads),
            desc=progress_label or f"screen {kind}",
            unit="sample",
            dynamic_ncols=True,
        )
    with progress_log.open("w", encoding="utf-8") as log_handle:
        for completed_index, row in enumerate(rows, start=1):
            started = perf_counter()
            try:
                record, reason = screen_file(Path(row.path), kind, cfg)
            except Exception as exc:
                log_handle.write(
                    json.dumps(
                        {
                            "event": "record_error",
                            "kind": kind,
                            "index": completed_index,
                            "total": len(downloads),
                            "pdb_id": str(row.pdb_id),
                            "path": str(row.path),
                            "status": "error",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "elapsed_seconds": perf_counter() - started,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log_handle.flush()
                raise
            if record is None:
                rejected.append({"pdb_id": row.pdb_id, "path": row.path, "reason": reason})
                status = "rejected"
            else:
                eligible.append(record)
                status = "eligible"
            log_handle.write(
                json.dumps(
                    {
                        "event": "record_complete",
                        "kind": kind,
                        "index": completed_index,
                        "total": len(downloads),
                        "pdb_id": str(row.pdb_id),
                        "path": str(row.path),
                        "status": status,
                        "reason": reason,
                        "sample_id": record.get("sample_id") if record is not None else None,
                        "elapsed_seconds": perf_counter() - started,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log_handle.flush()
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
        "progress_log": str(progress_log),
        "progress_records": len(downloads),
    }
    (out_dir / f"{kind}_screen_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return elig_df, rej_df
