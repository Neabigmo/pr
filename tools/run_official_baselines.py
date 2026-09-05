#!/usr/bin/env python3
"""Prepare, train and full-1000-refit the two pinned official baselines.

Fairness policy
---------------
For each predeclared training seed:
1. train on frozen 900 and select best epoch on frozen 100 validation;
2. restart from random initialization;
3. train exactly that fixed epoch/pass count on all frozen 1,000 structures;
4. never use the final 100 Protein-RNA complexes for model selection.

A one-row duplicated training structure is supplied as a dummy validation set
for the upstream final-refit loaders. It is never used for optimization or model
selection; it only satisfies upstream code that expects a non-empty validation
loader.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pr_pilot.baselines.wrappers import ensure_lock_file, pinned_upstream


PROTEIN_BEST_RE = re.compile(r"epoch:\s*(\d+),\s*step:\s*\d+,.*?valid:\s*([0-9.eE+\-]+)")
NA_BEST_RE = re.compile(r"epoch:\s*(\d+),\s*step:\s*\d+,.*?valid_rna_loss:\s*([0-9.eE+\-]+)")


def _run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _git(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *command], cwd=cwd, text=True).strip()


def clone_locked(repo_root: Path, third_party_root: Path) -> dict[str, Path]:
    lock = ensure_lock_file(repo_root)
    third_party_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ["ProteinMPNN", "NA-MPNN"]:
        spec = pinned_upstream(name, lock)
        destination = third_party_root / ("ProteinMPNN" if name == "ProteinMPNN" else "NA-MPNN")
        if not destination.exists():
            _run(["git", "clone", spec.url, str(destination)])
        _git(["fetch", "--all", "--tags"], destination)
        _git(["checkout", "--detach", spec.commit], destination)
        head = _git(["rev-parse", "HEAD"], destination)
        if head != spec.commit:
            raise RuntimeError(f"{name} checkout mismatch: {head} != {spec.commit}")
        paths[name] = destination
    return paths


def _prepare(
    repo_root: Path,
    protein_train: Path,
    protein_val: Path,
    rna_train: Path,
    rna_val: Path,
    output: Path,
    na_passes: int,
) -> None:
    _run(
        [
            sys.executable,
            str(repo_root / "tools" / "prepare_official_baselines.py"),
            "--protein-train",
            str(protein_train),
            "--protein-val",
            str(protein_val),
            "--rna-train",
            str(rna_train),
            "--rna-val",
            str(rna_val),
            "--out",
            str(output),
            "--passes",
            str(na_passes),
        ],
        cwd=repo_root,
    )


def _protein_command(
    repo: Path,
    data: Path,
    output: Path,
    seed: int,
    epochs: int,
    examples_per_epoch: int,
) -> list[str]:
    return [
        sys.executable,
        str(repo / "training" / "training.py"),
        "--path_for_training_data",
        str(data / "pdb"),
        "--path_for_outputs",
        str(output),
        "--path_for_training_clusters",
        str(data / "list.csv"),
        "--path_for_valid_clusters",
        str(data / "valid_clusters.txt"),
        "--path_for_test_clusters",
        str(data / "test_clusters.txt"),
        "--num_epochs",
        str(epochs),
        "--save_model_every_n_epochs",
        "1",
        "--reload_data_every_n_epochs",
        "1",
        "--num_examples_per_epoch",
        str(examples_per_epoch),
        "--batch_size",
        "6000",
        "--max_protein_length",
        "1000",
        "--backbone_noise",
        "0.1",
        "--seed",
        str(seed),
        "--mixed_precision",
    ]


def _na_command(repo: Path, config: Path, output_root: Path, seed: int) -> list[str]:
    return [
        sys.executable,
        str(repo / "na_run.py"),
        "--path_for_outputs",
        str(output_root),
        "--model_input_json",
        str(config),
        "--seed",
        str(seed),
    ]


def _best_epoch(log_path: Path, pattern: re.Pattern[str], label: str) -> tuple[int, float]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {label} training log: {log_path}")
    matches = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            matches.append((int(match.group(1)), float(match.group(2))))
    if not matches:
        raise RuntimeError(f"Could not parse validation epochs from {label} log {log_path}")
    return min(matches, key=lambda item: item[1])


def _find_single(root: Path, pattern: str) -> Path:
    candidates = sorted(root.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one file matching {root}/{pattern}, found {candidates}")
    return candidates[0]


def _latest_checkpoint(root: Path, pattern: str) -> Path:
    candidates = list(root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints under {root} matching {pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _dummy_validation(pool_path: Path, output: Path) -> Path:
    frame = pd.read_csv(pool_path, sep="\t")
    if frame.empty:
        raise ValueError(f"Cannot make dummy validation from empty pool {pool_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[:1].to_csv(output, sep="\t", index=False)
    return output


def _verify_counts(manifest_root: Path) -> None:
    expected = {
        "protein_train.tsv": 900,
        "protein_val.tsv": 100,
        "protein_pool.tsv": 1000,
        "rna_train.tsv": 900,
        "rna_val.tsv": 100,
        "rna_pool.tsv": 1000,
    }
    for name, count in expected.items():
        path = manifest_root / name
        if not path.exists():
            raise FileNotFoundError(path)
        got = len(pd.read_csv(path, sep="\t"))
        if got != count:
            raise ValueError(f"{name}: expected {count}, got {got}")


def run_seed(
    repo_root: Path,
    manifest_root: Path,
    upstream: dict[str, Path],
    output: Path,
    seed: int,
    max_passes: int,
    prepare_only: bool,
) -> dict:
    seed_root = output / f"seed{seed}"
    development_prep = seed_root / "prepared_development"
    _prepare(
        repo_root,
        manifest_root / "protein_train.tsv",
        manifest_root / "protein_val.tsv",
        manifest_root / "rna_train.tsv",
        manifest_root / "rna_val.tsv",
        development_prep,
        max_passes,
    )
    if prepare_only:
        return {"seed": seed, "prepared_development": str(development_prep)}

    protein_dev = seed_root / "development" / "ProteinMPNN"
    _run(
        _protein_command(
            upstream["ProteinMPNN"],
            development_prep / "proteinmpnn",
            protein_dev,
            seed,
            max_passes,
            examples_per_epoch=900,
        ),
        cwd=upstream["ProteinMPNN"],
    )
    protein_best_epoch, protein_best_valid = _best_epoch(
        protein_dev / "log.txt", PROTEIN_BEST_RE, "ProteinMPNN"
    )

    na_dev_config = development_prep / "na_mpnn" / "na_mpnn_from_scratch.json"
    na_dev = seed_root / "development" / "NA-MPNN"
    _run(_na_command(upstream["NA-MPNN"], na_dev_config, na_dev, seed), cwd=upstream["NA-MPNN"])
    na_dev_log = _find_single(na_dev, "*/log.txt")
    na_best_epoch, na_best_valid = _best_epoch(na_dev_log, NA_BEST_RE, "NA-MPNN")

    dummy_root = seed_root / "refit_dummy_validation"
    protein_dummy = _dummy_validation(manifest_root / "protein_pool.tsv", dummy_root / "protein.tsv")
    rna_dummy = _dummy_validation(manifest_root / "rna_pool.tsv", dummy_root / "rna.tsv")
    refit_prep = seed_root / "prepared_refit_full1000"
    _prepare(
        repo_root,
        manifest_root / "protein_pool.tsv",
        protein_dummy,
        manifest_root / "rna_pool.tsv",
        rna_dummy,
        refit_prep,
        na_best_epoch,
    )

    protein_refit = seed_root / "final_refit_full1000" / "ProteinMPNN"
    _run(
        _protein_command(
            upstream["ProteinMPNN"],
            refit_prep / "proteinmpnn",
            protein_refit,
            seed,
            protein_best_epoch,
            examples_per_epoch=1000,
        ),
        cwd=upstream["ProteinMPNN"],
    )
    protein_final = protein_refit / "model_weights" / "epoch_last.pt"
    if not protein_final.exists():
        raise FileNotFoundError(protein_final)

    na_refit_config = refit_prep / "na_mpnn" / "na_mpnn_from_scratch.json"
    na_refit = seed_root / "final_refit_full1000" / "NA-MPNN"
    _run(_na_command(upstream["NA-MPNN"], na_refit_config, na_refit, seed), cwd=upstream["NA-MPNN"])
    na_final = _latest_checkpoint(na_refit, "*/model_weights/*.pt")

    summary = {
        "seed": seed,
        "ProteinMPNN": {
            "development_best_epoch": protein_best_epoch,
            "development_best_valid_perplexity": protein_best_valid,
            "final_refit_training_structures": 1000,
            "final_refit_epochs": protein_best_epoch,
            "checkpoint": str(protein_final.resolve()),
        },
        "NA-MPNN": {
            "development_best_epoch": na_best_epoch,
            "development_best_valid_rna_loss": na_best_valid,
            "final_refit_training_structures": 1000,
            "final_refit_dataset_passes": na_best_epoch,
            "checkpoint": str(na_final.resolve()),
        },
        "final_test_used_for_selection": False,
    }
    (seed_root / "baseline_refit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests/pilot_v1"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--third-party-root", type=Path, default=Path("third_party/checkouts"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260905, 20260906, 20260907])
    parser.add_argument("--max-passes", type=int, default=150)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest_root = args.manifest_root.resolve()
    output = args.out.resolve()
    third_party_root = args.third_party_root.resolve()
    _verify_counts(manifest_root)
    upstream = clone_locked(repo_root, third_party_root)

    summaries = [
        run_seed(
            repo_root,
            manifest_root,
            upstream,
            output,
            int(seed),
            int(args.max_passes),
            args.prepare_only,
        )
        for seed in args.seeds
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "baseline_run_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
