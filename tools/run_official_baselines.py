#!/usr/bin/env python3
"""Prepare, train and full-1000-refit the two pinned official baselines.

Fairness policy
---------------
For each predeclared seed:
1. train on the exact frozen 900 and select the best epoch on the exact frozen 100;
2. restart from random initialization;
3. train that fixed number of epochs/data-passes on all frozen 1,000 structures;
4. never use the final 100 Protein-RNA complexes for model selection.

The upstream repositories are not edited.  ``tools/run_seeded_upstream.py`` only
controls RNG initialization before the pinned upstream entrypoint is executed.
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


PROTEIN_BEST_RE = re.compile(
    r"epoch:\s*(\d+),\s*step:\s*\d+,.*?valid:\s*([0-9.eE+\-]+)"
)
NA_BEST_RE = re.compile(
    r"epoch:\s*(\d+),\s*step:\s*\d+,.*?valid_rna_loss:\s*([0-9.eE+\-]+)"
)


def _run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _git(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *command], cwd=cwd, text=True).strip()


def clone_locked(repo_root: Path, third_party_root: Path) -> dict[str, Path]:
    lock = ensure_lock_file(repo_root)
    third_party_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in ["ProteinMPNN", "NA-MPNN"]:
        spec = pinned_upstream(name, lock)
        destination = third_party_root / (
            "ProteinMPNN" if name == "ProteinMPNN" else "NA-MPNN"
        )
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
    repo_root: Path,
    upstream_repo: Path,
    prepared_root: Path,
    output: Path,
    seed: int,
    epochs: int,
    examples_per_epoch: int,
) -> list[str]:
    """Command valid for pinned ProteinMPNN training.py.

    ``path_for_training_data`` is the directory containing list.csv,
    valid_clusters.txt, test_clusters.txt and pdb/ -- not the pdb/ subfolder.
    """
    return [
        sys.executable,
        str(repo_root / "tools" / "run_seeded_upstream.py"),
        "--seed",
        str(seed),
        "--script",
        str(upstream_repo / "training" / "training.py"),
        "--deterministic-empty-numpy-seed",
        "--",
        "--path_for_training_data",
        str(prepared_root),
        "--path_for_outputs",
        str(output),
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
        "--mixed_precision",
        "True",
    ]


def _na_run_config(base_config: Path, output_root: Path) -> Path:
    params = json.loads(base_config.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    params["BASE_FOLDER"] = str(output_root.resolve())
    params["PREV_CHECKPOINT"] = ""
    path = output_root / "run_config.json"
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return path


def _na_command(
    repo_root: Path,
    upstream_repo: Path,
    config: Path,
    seed: int,
) -> list[str]:
    # Pinned na_run.py reads exactly one positional JSON path: sys.argv[1].
    return [
        sys.executable,
        str(repo_root / "tools" / "run_seeded_upstream.py"),
        "--seed",
        str(seed),
        "--script",
        str(upstream_repo / "na_run.py"),
        "--",
        str(config),
    ]


def _best_epoch(
    log_path: Path, pattern: re.Pattern[str], label: str
) -> tuple[int, float]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing {label} training log: {log_path}")
    matches: list[tuple[int, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            matches.append((int(match.group(1)), float(match.group(2))))
    if not matches:
        raise RuntimeError(
            f"Could not parse validation epochs from {label} log {log_path}"
        )
    return min(matches, key=lambda item: item[1])


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
            repo_root,
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

    na_dev_root = seed_root / "development" / "NA-MPNN"
    na_dev_cfg = _na_run_config(
        development_prep / "na_mpnn" / "na_mpnn_from_scratch.json", na_dev_root
    )
    _run(
        _na_command(repo_root, upstream["NA-MPNN"], na_dev_cfg, seed),
        cwd=upstream["NA-MPNN"],
    )
    na_best_epoch, na_best_valid = _best_epoch(
        na_dev_root / "log.txt", NA_BEST_RE, "NA-MPNN"
    )

    # Upstream loaders require a non-empty validation source even during final
    # refit.  This one-row duplicate is never used for optimization or selection.
    dummy_root = seed_root / "refit_dummy_validation"
    protein_dummy = _dummy_validation(
        manifest_root / "protein_pool.tsv", dummy_root / "protein.tsv"
    )
    rna_dummy = _dummy_validation(
        manifest_root / "rna_pool.tsv", dummy_root / "rna.tsv"
    )
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
            repo_root,
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

    na_refit_root = seed_root / "final_refit_full1000" / "NA-MPNN"
    na_refit_cfg = _na_run_config(
        refit_prep / "na_mpnn" / "na_mpnn_from_scratch.json", na_refit_root
    )
    _run(
        _na_command(repo_root, upstream["NA-MPNN"], na_refit_cfg, seed),
        cwd=upstream["NA-MPNN"],
    )
    na_final = na_refit_root / "last.pt"
    if not na_final.exists():
        raise FileNotFoundError(na_final)

    summary = {
        "seed": seed,
        "ProteinMPNN": {
            "development_best_epoch": protein_best_epoch,
            "development_best_valid_perplexity": protein_best_valid,
            "final_refit_training_structures": 1000,
            "final_refit_epochs": protein_best_epoch,
            "checkpoint": str(protein_final.resolve()),
            "upstream_rng_note": (
                "Top-level Python/NumPy/PyTorch seeds fixed; legacy no-arg NumPy "
                "worker reseeding is deterministically patched by wrapper only."
            ),
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
    (seed_root / "baseline_refit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest-root", type=Path, default=Path("manifests/pilot_v1")
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument(
        "--third-party-root", type=Path, default=Path("third_party/checkouts")
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260905, 20260906, 20260907]
    )
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
    (output / "baseline_run_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
