"""Canonical residue mapping shared by screening, adapters and baseline export.

Silent residue dropping is forbidden: modified residues that Gemmi can map to a
canonical parent (for example MSE->M, PSU->U) are accepted; DNA and unsupported
non-canonical nucleotides are explicitly distinguishable from ordinary ligands.
"""
from __future__ import annotations

from dataclasses import dataclass

import gemmi


PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
RNA_ALPHABET = "AUGC"

# Legacy aliases found in some PDB writers. Gemmi handles the standard CCD names.
RNA_ALIASES = {"RA": "A", "RU": "U", "RG": "G", "RC": "C", "ADE": "A", "URA": "U", "GUA": "G", "CYT": "C"}
PROTEIN_ALIASES: dict[str, str] = {}


@dataclass(frozen=True)
class ResidueClass:
    polymer: str  # protein | rna | dna | unsupported_polymer | other
    token: str | None
    modified: bool


def classify_residue(name: str) -> ResidueClass:
    """Classify a CCD/PDB residue name without guessing unsupported chemistry."""
    name = str(name).strip().upper()
    if name in RNA_ALIASES:
        return ResidueClass("rna", RNA_ALIASES[name], name not in RNA_ALPHABET)
    if name in PROTEIN_ALIASES:
        return ResidueClass("protein", PROTEIN_ALIASES[name], True)

    info = gemmi.find_tabulated_residue(name)
    code = str(info.one_letter_code).strip().upper()
    if info.kind == gemmi.ResidueKind.DNA:
        return ResidueClass("dna", code if code else None, name not in {"DA", "DC", "DG", "DT", "DU"})
    if info.kind == gemmi.ResidueKind.RNA:
        if code in RNA_ALPHABET:
            return ResidueClass("rna", code, name != code)
        # Inosine and other non-A/U/G/C monomers are real RNA, but mapping them to
        # an arbitrary canonical base would alter the supervised target. Reject.
        return ResidueClass("unsupported_polymer", None, True)
    if info.is_amino_acid():
        if code in PROTEIN_ALPHABET:
            return ResidueClass("protein", code, len(name) != 3 or name not in {
                "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            })
        return ResidueClass("unsupported_polymer", None, True)
    return ResidueClass("other", None, False)


def protein_token(name: str) -> str | None:
    c = classify_residue(name)
    return c.token if c.polymer == "protein" else None


def rna_token(name: str) -> str | None:
    c = classify_residue(name)
    return c.token if c.polymer == "rna" else None
