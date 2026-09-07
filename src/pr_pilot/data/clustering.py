"""Joint sequence clustering and Rfam annotation for leakage control.

Crucial rule: single-molecule and complex candidates are clustered together.
Otherwise a test complex could be P30/R80-related to a pretraining structure but
carry unrelated cluster labels generated in separate runs.

External tools:
- MMseqs2 for P90/P40/P30 and RNA R90/R80 clustering;
- Infernal cmscan + official Rfam.cm/Rfam.clanin for RNA family labels.

The functions fail loudly when tools/resources are unavailable. Missing Rfam
labels are represented as ``unknown`` and strict splitting falls back to R80 for
those sequences; this fallback is recorded rather than hidden.
"""
from __future__ import annotations

from pathlib import Path
import gzip
import json
import shutil
import subprocess

import pandas as pd


PROTEIN_THRESHOLDS = {"protein_cluster_p90": 0.90, "protein_cluster_p40": 0.40, "protein_cluster_p30": 0.30}
RNA_THRESHOLDS = {"rna_cluster_r90": 0.90, "rna_cluster_r80": 0.80}


def _require_executable(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise RuntimeError(f"Required executable {name!r} is not on PATH")
    return exe


def _write_fasta(records: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for key in sorted(records):
            seq = records[key].replace("|", "")
            if not seq:
                raise ValueError(f"Empty sequence for {key}")
            out.write(f">{key}\n{seq}\n")


def _run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}); inspect {log}")


def run_mmseqs_cluster(
    fasta: Path,
    out_dir: Path,
    label: str,
    min_seq_id: float,
    coverage: float = 0.8,
    threads: int | None = None,
) -> dict[str, str]:
    """Run MMseqs2 clustering and return the member-to-representative mapping.

    The explicit ``createdb``/``cluster``/``createtsv`` sequence is equivalent
    to the clustering stage of ``easy-cluster`` but avoids materializing the
    representative and all-member FASTA exports.  Those exports can become
    disproportionately large on the Windows/WSL filesystem while the TSV is
    the only artifact required by the leakage-safe splitter.
    """
    mmseqs = _require_executable("mmseqs")
    database = out_dir / f"{label}_db"
    cluster_database = out_dir / f"{label}_clusters"
    tmp = out_dir / f"tmp_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    createdb = [mmseqs, "createdb", str(fasta), str(database)]
    cluster = [
        mmseqs, "cluster", str(database), str(cluster_database), str(tmp),
        "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0",
    ]
    createtsv = [mmseqs, "createtsv", str(database), str(database), str(cluster_database)]
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be positive")
        for command in (createdb, cluster, createtsv):
            command.extend(["--threads", str(threads)])
    cluster_tsv = out_dir / f"{label}_cluster.tsv"
    _run(createdb, out_dir / f"{label}_createdb.log")
    _run(cluster, out_dir / f"{label}.log")
    _run([*createtsv, str(cluster_tsv)], out_dir / f"{label}_createtsv.log")
    if not cluster_tsv.exists():
        raise RuntimeError(f"MMseqs did not produce {cluster_tsv}")
    mapping: dict[str, str] = {}
    for line in cluster_tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        representative, member = line.split("\t")[:2]
        mapping[member] = representative
    return mapping


def prepare_rfam_database(rfam_cm_gz: Path, out_dir: Path) -> Path:
    """Decompress and cmpress Rfam.cm once."""
    cmpress = _require_executable("cmpress")
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = out_dir / "Rfam.cm"
    if not cm.exists():
        with gzip.open(rfam_cm_gz, "rb") as src, cm.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    indexes = [Path(str(cm) + suffix) for suffix in (".i1f", ".i1i", ".i1m", ".i1p")]
    if not all(p.exists() for p in indexes):
        _run([cmpress, "-F", str(cm)], out_dir / "cmpress.log")
    return cm


def run_rfam_cmscan(fasta: Path, rfam_cm: Path, clanin: Path, out_dir: Path, cpu: int = 4) -> dict[str, set[str]]:
    """Return query sequence ID -> set of Rfam accessions passing curated GA thresholds."""
    cmscan = _require_executable("cmscan")
    out_dir.mkdir(parents=True, exist_ok=True)
    tbl = out_dir / "rfam.tbl"
    command = [
        cmscan, "--cpu", str(cpu), "--cut_ga", "--rfam", "--nohmmonly",
        "--clanin", str(clanin), "--oskip", "--fmt", "2",
        "--tblout", str(tbl), str(rfam_cm), str(fasta),
    ]
    _run(command, out_dir / "cmscan.log")
    hits: dict[str, set[str]] = {}
    for line in tbl.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        # Infernal ``--fmt 2`` starts with index, target name, target
        # accession, then query name.  The query identifier must be taken from
        # column 4; using the accession column silently labels every sequence
        # as ``unknown`` downstream.
        target_name, target_accession, query_name = fields[1], fields[2], fields[3]
        family = target_accession if target_accession not in {"-", "."} else target_name
        hits.setdefault(query_name, set()).add(family)
    return hits


def _load_chain_dict(raw: str) -> dict[str, str]:
    data = json.loads(raw)
    if not isinstance(data, dict) or not data:
        raise ValueError("Chain sequence JSON must be a non-empty object")
    return {str(k): str(v) for k, v in data.items()}


def annotate_all_candidates(
    protein_eligible: Path,
    rna_eligible: Path,
    complex_eligible: Path,
    out_dir: Path,
    *,
    rfam_cm_gz: Path,
    rfam_clanin: Path,
    cmscan_cpu: int = 4,
) -> tuple[Path, Path, Path]:
    """Jointly annotate all candidate tables and return annotated TSV paths."""
    p_single = pd.read_csv(protein_eligible, sep="\t")
    r_single = pd.read_csv(rna_eligible, sep="\t")
    complexes = pd.read_csv(complex_eligible, sep="\t")
    out_dir.mkdir(parents=True, exist_ok=True)

    protein_sequences: dict[str, str] = {}
    rna_sequences: dict[str, str] = {}
    p_single_key: dict[str, str] = {}
    r_single_key: dict[str, str] = {}
    complex_p_keys: dict[str, list[str]] = {}
    complex_r_keys: dict[str, list[str]] = {}

    for row in p_single.itertuples(index=False):
        key = f"singleP::{row.sample_id}"
        protein_sequences[key] = str(row.sequence)
        p_single_key[str(row.sample_id)] = key
    for row in r_single.itertuples(index=False):
        key = f"singleR::{row.sample_id}"
        rna_sequences[key] = str(row.sequence)
        r_single_key[str(row.sample_id)] = key
    for row in complexes.itertuples(index=False):
        sid = str(row.sample_id)
        pkeys, rkeys = [], []
        for chain, seq in _load_chain_dict(row.protein_chain_sequences).items():
            key = f"complexP::{sid}::{chain}"
            protein_sequences[key] = seq
            pkeys.append(key)
        for chain, seq in _load_chain_dict(row.rna_chain_sequences).items():
            key = f"complexR::{sid}::{chain}"
            rna_sequences[key] = seq
            rkeys.append(key)
        complex_p_keys[sid] = pkeys
        complex_r_keys[sid] = rkeys

    protein_fasta = out_dir / "all_protein_sequences.fa"
    rna_fasta = out_dir / "all_rna_sequences.fa"
    _write_fasta(protein_sequences, protein_fasta)
    _write_fasta(rna_sequences, rna_fasta)

    p_maps = {
        col: run_mmseqs_cluster(protein_fasta, out_dir / "mmseqs", col, threshold, threads=cmscan_cpu)
        for col, threshold in PROTEIN_THRESHOLDS.items()
    }
    r_maps = {
        col: run_mmseqs_cluster(rna_fasta, out_dir / "mmseqs", col, threshold, threads=cmscan_cpu)
        for col, threshold in RNA_THRESHOLDS.items()
    }

    rfam_cm = prepare_rfam_database(rfam_cm_gz, out_dir / "rfam_db")
    rfam_hits = run_rfam_cmscan(rna_fasta, rfam_cm, rfam_clanin, out_dir / "rfam_scan", cpu=cmscan_cpu)

    for col, mapping in p_maps.items():
        p_single[col] = p_single["sample_id"].astype(str).map(lambda sid: mapping[p_single_key[sid]])
        complexes[col] = complexes["sample_id"].astype(str).map(
            lambda sid: ";".join(sorted({mapping[k] for k in complex_p_keys[sid]}))
        )
    for col, mapping in r_maps.items():
        r_single[col] = r_single["sample_id"].astype(str).map(lambda sid: mapping[r_single_key[sid]])
        complexes[col] = complexes["sample_id"].astype(str).map(
            lambda sid: ";".join(sorted({mapping[k] for k in complex_r_keys[sid]}))
        )

    r_single["rfam_family"] = r_single["sample_id"].astype(str).map(
        lambda sid: ";".join(sorted(rfam_hits.get(r_single_key[sid], set()))) or "unknown"
    )
    complexes["rfam_family"] = complexes["sample_id"].astype(str).map(
        lambda sid: ";".join(sorted(set().union(*(rfam_hits.get(k, set()) for k in complex_r_keys[sid])))) or "unknown"
    )

    # Freeze-manifest code expects these generic names.
    p_single["protein_cluster_p30"] = p_single["protein_cluster_p30"].astype(str)
    r_single["rna_cluster_r80"] = r_single["rna_cluster_r80"].astype(str)

    p_path = out_dir / "protein_annotated.tsv"
    r_path = out_dir / "rna_annotated.tsv"
    c_path = out_dir / "complex_annotated.tsv"
    p_single.to_csv(p_path, sep="\t", index=False)
    r_single.to_csv(r_path, sep="\t", index=False)
    complexes.to_csv(c_path, sep="\t", index=False)
    return p_path, r_path, c_path
