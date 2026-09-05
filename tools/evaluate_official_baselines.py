#!/usr/bin/env python3
"""Evaluate pinned official one-sided baselines on the frozen final 100 complexes.

ProteinMPNN sees only Protein coordinates and exports official backbone-only
unconditional probabilities. NA-MPNN sees only RNA coordinates and exports the
official specificity PPM. Both are converted into one common per-position table.

These are external *one-sided structural references*. They do not receive partner
identity and therefore are not substitutes for same-data DM-ICF controls.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pr_pilot.baselines.wrappers import ensure_lock_file, pinned_upstream


PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
RNA_ALPHABET = "AUGC"
PROTEIN_INDEX = {x: i for i, x in enumerate(PROTEIN_ALPHABET)}
RNA_INDEX = {x: i for i, x in enumerate(RNA_ALPHABET)}


def _run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _clone_locked(repo_root: Path, third_party_root: Path) -> dict[str, Path]:
    lock = ensure_lock_file(repo_root)
    third_party_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for name in ["ProteinMPNN", "NA-MPNN"]:
        spec = pinned_upstream(name, lock)
        checkout = third_party_root / (
            "ProteinMPNN" if name == "ProteinMPNN" else "NA-MPNN"
        )
        if not checkout.exists():
            _run(["git", "clone", spec.url, str(checkout)])
        _run(["git", "fetch", "--all", "--tags"], checkout)
        _run(["git", "checkout", "--detach", spec.commit], checkout)
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        if head != spec.commit:
            raise RuntimeError(
                f"Pinned checkout mismatch for {name}: {head} != {spec.commit}"
            )
        result[name] = checkout
    return result


def _logsumexp(
    x: np.ndarray, axis: int = -1, keepdims: bool = False
) -> np.ndarray:
    maximum = np.max(x, axis=axis, keepdims=True)
    value = maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True))
    return value if keepdims else np.squeeze(value, axis=axis)


def _protein_rows(
    npz_path: Path, mapping: pd.DataFrame, sample_id: str, seed: int
) -> list[dict]:
    data = np.load(npz_path, allow_pickle=True)
    log_p = np.asarray(data["log_p"])
    if log_p.ndim == 3:
        log_p = log_p.mean(axis=0)
    if log_p.ndim != 2 or log_p.shape[1] < 20:
        raise ValueError(f"Unexpected ProteinMPNN log_p shape {log_p.shape} in {npz_path}")
    canonical = log_p[:, :20]
    canonical = canonical - _logsumexp(canonical, axis=-1, keepdims=True)
    mapping = mapping.sort_values("baseline_position")
    if len(mapping) != canonical.shape[0]:
        raise ValueError(
            f"ProteinMPNN position count mismatch for {sample_id}: "
            f"{canonical.shape[0]} vs {len(mapping)}"
        )
    rows = []
    for position, item in enumerate(mapping.itertuples(index=False)):
        native = PROTEIN_INDEX[str(item.token)]
        predicted = int(canonical[position].argmax())
        rows.append(
            {
                "sample_id": sample_id,
                "polymer": "protein",
                "position": position,
                "original_residue_id": item.original_residue_id,
                "native_token": native,
                "predicted_token": predicted,
                "native_log_probability": float(canonical[position, native]),
                "max_probability": float(np.exp(canonical[position, predicted])),
                "is_interface": bool(item.is_interface),
                "model": "ProteinMPNN_full1000",
                "seed": int(seed),
                "probability_semantics": (
                    "official backbone-only unconditional log probabilities; "
                    "renormalized over 20 canonical amino acids"
                ),
            }
        )
    return rows


def _rna_columns(restype_to_int: dict) -> list[int]:
    """Return columns in this project's canonical A/U/G/C order.

    With ``NA_SHARED_TOKENS=1`` upstream maps RNA A,C,G,U onto the DA,DC,DG,DT
    token slots.  The correct canonical order is therefore DA,DT,DG,DC -- *not*
    DA,DC,DG,DT.  Prefer RNA aliases when available because they already encode
    the shared-token mapping.
    """
    if all(key in restype_to_int for key in ["A", "U", "G", "C"]):
        return [int(restype_to_int[key]) for key in ["A", "U", "G", "C"]]
    if all(key in restype_to_int for key in ["DA", "DT", "DG", "DC"]):
        return [int(restype_to_int[key]) for key in ["DA", "DT", "DG", "DC"]]
    raise ValueError(
        f"Cannot identify A/U/G/C columns from restype_to_int={restype_to_int}"
    )


def _rna_rows(
    npz_path: Path, mapping: pd.DataFrame, sample_id: str, seed: int
) -> list[dict]:
    data = np.load(npz_path, allow_pickle=True)
    ppm = np.asarray(data["predicted_ppm"], dtype=np.float64)
    if ppm.ndim == 3:
        ppm = ppm.mean(axis=0)
    restype_to_int = data["restype_to_int"].item()
    columns = _rna_columns(restype_to_int)
    four = ppm[:, columns]
    four = np.clip(four, 0.0, None)
    denom = four.sum(axis=-1, keepdims=True)
    if (denom <= 0).any():
        raise ValueError(
            f"NA-MPNN produced zero four-base probability mass in {npz_path}"
        )
    four /= denom

    rna_mask = np.asarray(
        data.get("rna_mask", np.ones(len(four), dtype=bool))
    ).astype(bool)
    if len(rna_mask) != len(four):
        raise ValueError("NA-MPNN rna_mask length mismatch")
    four = four[rna_mask]
    mapping = mapping.sort_values("baseline_position")
    if len(mapping) != four.shape[0]:
        raise ValueError(
            f"NA-MPNN position count mismatch for {sample_id}: "
            f"{four.shape[0]} vs {len(mapping)}"
        )

    rows = []
    for position, item in enumerate(mapping.itertuples(index=False)):
        native = RNA_INDEX[str(item.token)]
        predicted = int(four[position].argmax())
        rows.append(
            {
                "sample_id": sample_id,
                "polymer": "rna",
                "position": position,
                "original_residue_id": item.original_residue_id,
                "native_token": native,
                "predicted_token": predicted,
                "native_log_probability": float(
                    math.log(max(four[position, native], 1e-12))
                ),
                "max_probability": float(four[position, predicted]),
                "is_interface": bool(item.is_interface),
                "model": "NA-MPNN_full1000",
                "seed": int(seed),
                "probability_semantics": (
                    "official specificity PPM averaged over sampling trajectories; "
                    "renormalized over canonical A/U/G/C"
                ),
            }
        )
    return rows


def _protein_command(
    repo: Path,
    checkpoint: Path,
    pdb: Path,
    chains: str,
    out: Path,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        str(repo / "protein_mpnn_run.py"),
        "--path_to_model_weights",
        str(checkpoint.parent),
        "--model_name",
        checkpoint.stem,
        "--pdb_path",
        str(pdb),
        "--pdb_path_chains",
        chains,
        "--out_folder",
        str(out),
        "--unconditional_probs_only",
        "1",
        "--batch_size",
        "1",
        "--num_seq_per_target",
        "1",
        "--seed",
        str(seed),
        "--suppress_print",
        "1",
    ]


def _na_command(
    repo: Path,
    checkpoint: Path,
    pdb: Path,
    out: Path,
    seed: int,
    batch_size: int,
) -> list[str]:
    # Every flag below is present in pinned inference/run.py.  In particular,
    # there is no --rna_backbone_noise flag in that entrypoint.
    return [
        sys.executable,
        str(repo / "inference" / "run.py"),
        "--model_type",
        "na_mpnn",
        "--mode",
        "specificity",
        "--checkpoint_na_mpnn",
        str(checkpoint),
        "--pdb_path",
        str(pdb),
        "--out_folder",
        str(out),
        "--parse_na_only",
        "1",
        "--design_na_only",
        "1",
        "--load_residues_with_missing_atoms",
        "1",
        "--output_pdbs",
        "0",
        "--output_sequences",
        "0",
        "--output_specificity",
        "1",
        "--temperature",
        "1.0",
        "--batch_size",
        str(batch_size),
        "--number_of_batches",
        "1",
        "--seed",
        str(seed),
        "--catch_failed_inferences",
        "0",
    ]


def evaluate(
    repo_root: Path,
    baseline_summary: Path,
    prepared_holdout: Path,
    output: Path,
    third_party_root: Path,
    na_probability_samples: int = 64,
) -> dict:
    upstream = _clone_locked(repo_root, third_party_root)
    runs = json.loads(baseline_summary.read_text(encoding="utf-8"))
    samples = pd.read_csv(prepared_holdout / "samples.tsv", sep="\t")
    mapping = pd.read_csv(prepared_holdout / "position_mapping.tsv", sep="\t")
    if len(samples) != 100:
        raise ValueError(f"Expected frozen 100 complex views, found {len(samples)}")
    output.mkdir(parents=True, exist_ok=True)

    all_rows = []
    run_manifest = []
    for run in runs:
        seed = int(run["seed"])
        if "checkpoint" not in run.get("ProteinMPNN", {}) or "checkpoint" not in run.get("NA-MPNN", {}):
            raise ValueError(
                "Baseline summary lacks full-refit checkpoints; do not evaluate development weights"
            )
        protein_checkpoint = Path(run["ProteinMPNN"]["checkpoint"])
        rna_checkpoint = Path(run["NA-MPNN"]["checkpoint"])
        seed_out = output / f"seed{seed}"
        token_rows = []

        for sample in samples.itertuples(index=False):
            sample_id = str(sample.sample_id)
            safe = Path(sample.protein_pdb).stem
            sample_map = mapping[mapping.sample_id.astype(str) == sample_id]

            protein_out = seed_out / "raw" / "ProteinMPNN" / safe
            _run(
                _protein_command(
                    upstream["ProteinMPNN"],
                    protein_checkpoint,
                    Path(sample.protein_pdb),
                    str(sample.protein_chain_ids),
                    protein_out,
                    seed,
                ),
                cwd=upstream["ProteinMPNN"],
            )
            protein_npz = protein_out / "unconditional_probs_only" / f"{safe}.npz"
            if not protein_npz.exists():
                raise FileNotFoundError(protein_npz)
            token_rows.extend(
                _protein_rows(
                    protein_npz,
                    sample_map[sample_map.polymer == "protein"],
                    sample_id,
                    seed,
                )
            )

            rna_out = seed_out / "raw" / "NA-MPNN" / safe
            _run(
                _na_command(
                    upstream["NA-MPNN"],
                    rna_checkpoint,
                    Path(sample.rna_pdb),
                    rna_out,
                    seed,
                    na_probability_samples,
                ),
                cwd=upstream["NA-MPNN"],
            )
            rna_npz = rna_out / "specificity" / f"{safe}.npz"
            if not rna_npz.exists():
                raise FileNotFoundError(rna_npz)
            token_rows.extend(
                _rna_rows(
                    rna_npz,
                    sample_map[sample_map.polymer == "rna"],
                    sample_id,
                    seed,
                )
            )

        token_df = pd.DataFrame(token_rows)
        token_path = seed_out / "external_one_sided_tokens.tsv"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_df.to_csv(token_path, sep="\t", index=False)
        all_rows.append(token_df)
        run_manifest.append(
            {
                "model": "external_one_sided_refs",
                "seed": seed,
                "run_dir": str(seed_out.resolve()),
            }
        )

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(
        output / "all_seeds_external_one_sided_tokens.tsv", sep="\t", index=False
    )
    summary = {
        "complexes": int(len(samples)),
        "seeds": sorted(combined.seed.astype(int).unique().tolist()),
        "protein_rows": int((combined.polymer == "protein").sum()),
        "rna_rows": int((combined.polymer == "rna").sum()),
        "protein_reference": "ProteinMPNN backbone-only; no RNA partner",
        "rna_reference": "NA-MPNN RNA-only specificity PPM; no Protein partner",
        "causal_comparison_warning": (
            "Use DM-ICF internal partner-blind/component controls for cross-molecular "
            "mechanistic claims; these external baselines solve one-sided tasks."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(run_manifest).to_csv(
        output / "run_manifest.tsv", sep="\t", index=False
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--prepared-holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--third-party-root", type=Path, default=Path("third_party/checkouts")
    )
    parser.add_argument("--na-probability-samples", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.repo_root.resolve(),
                args.baseline_summary.resolve(),
                args.prepared_holdout.resolve(),
                args.out.resolve(),
                args.third_party_root.resolve(),
                args.na_probability_samples,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
