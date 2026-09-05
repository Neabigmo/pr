#!/usr/bin/env python3
"""CPU-only contract audit for the pinned ProteinMPNN and NA-MPNN versions.

This does not train a model. It proves that our wrapper assumptions match the exact
pinned upstream source before GPU time is spent.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

from pr_pilot.baselines.wrappers import ensure_lock_file, pinned_upstream


EXPECTED_PROTEIN_ARGS = {
    "--path_for_training_data",
    "--path_for_outputs",
    "--num_epochs",
    "--save_model_every_n_epochs",
    "--reload_data_every_n_epochs",
    "--num_examples_per_epoch",
    "--batch_size",
    "--max_protein_length",
    "--backbone_noise",
    "--mixed_precision",
}
FORBIDDEN_OLD_PROTEIN_ARGS = {
    "--path_for_training_clusters",
    "--path_for_valid_clusters",
    "--path_for_test_clusters",
}


def _clone_or_verify(repo_root: Path, third_party_root: Path) -> dict[str, Path]:
    lock = ensure_lock_file(repo_root)
    third_party_root.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name in ["ProteinMPNN", "NA-MPNN"]:
        spec = pinned_upstream(name, lock)
        checkout = third_party_root / name
        if not checkout.exists():
            subprocess.run(["git", "clone", spec.url, str(checkout)], check=True)
        subprocess.run(["git", "fetch", "--all", "--tags"], cwd=checkout, check=True)
        subprocess.run(["git", "checkout", "--detach", spec.commit], cwd=checkout, check=True)
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        if head != spec.commit:
            raise RuntimeError(f"{name} pinned checkout mismatch: {head} != {spec.commit}")
        out[name] = checkout
    return out


def _argparse_options(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                options.add(arg.value)
    return options


def audit(repo_root: Path, third_party_root: Path) -> dict:
    upstream = _clone_or_verify(repo_root, third_party_root)

    protein_source = upstream["ProteinMPNN"] / "training" / "training.py"
    protein_options = _argparse_options(protein_source)
    missing = sorted(EXPECTED_PROTEIN_ARGS - protein_options)
    if missing:
        raise AssertionError(f"Pinned ProteinMPNN missing expected args: {missing}")

    runner_core = (repo_root / "tools" / "run_official_baselines_core.py").read_text(
        encoding="utf-8"
    )
    leaked_old = sorted(
        arg for arg in FORBIDDEN_OLD_PROTEIN_ARGS if f'"{arg}"' in runner_core
    )
    if leaked_old:
        raise AssertionError(
            f"ProteinMPNN runner contains unsupported training args: {leaked_old}"
        )
    if 'development_prep / "proteinmpnn" / "pdb"' in runner_core:
        raise AssertionError(
            "ProteinMPNN path_for_training_data points at /pdb instead of prepared root"
        )

    na_train_source = (upstream["NA-MPNN"] / "na_run.py").read_text(encoding="utf-8")
    if not re.search(r"sys\.argv\s*\[\s*1\s*\]", na_train_source):
        raise AssertionError(
            "Pinned NA-MPNN na_run.py no longer appears to use one positional JSON path"
        )
    if not re.search(
        r"BASE_FOLDER[^\n]*\+\s*['\"]log\.txt['\"]", na_train_source
    ):
        raise AssertionError(
            "Pinned NA-MPNN log-path concatenation changed; review BASE_FOLDER separator contract"
        )

    runner_wrapper = (repo_root / "tools" / "run_official_baselines.py").read_text(
        encoding="utf-8"
    )
    prepare_wrapper = (repo_root / "tools" / "prepare_official_baselines.py").read_text(
        encoding="utf-8"
    )
    if '+ "/"' not in runner_wrapper or '+ "/"' not in prepare_wrapper:
        raise AssertionError("NA-MPNN BASE_FOLDER trailing-separator guard is missing")

    evaluator = (repo_root / "tools" / "evaluate_official_baselines.py").read_text(
        encoding="utf-8"
    )
    if "--rna_backbone_noise" in evaluator:
        raise AssertionError("NA-MPNN evaluator contains unsupported --rna_backbone_noise")
    if '["DA", "DT", "DG", "DC"]' not in evaluator:
        raise AssertionError(
            "NA-MPNN AUGC shared-token mapping is not explicitly DA,DT,DG,DC"
        )

    inference_options = _argparse_options(upstream["NA-MPNN"] / "inference" / "run.py")
    required_inference = {
        "--model_type",
        "--mode",
        "--checkpoint_na_mpnn",
        "--pdb_path",
        "--out_folder",
        "--design_na_only",
        "--parse_na_only",
        "--output_specificity",
        "--omit_AA",
        "--temperature",
        "--batch_size",
        "--number_of_batches",
        "--seed",
    }
    missing_inference = sorted(required_inference - inference_options)
    if missing_inference:
        raise AssertionError(
            f"Pinned NA-MPNN inference missing expected args: {missing_inference}"
        )

    return {
        "ProteinMPNN": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=upstream["ProteinMPNN"], text=True
            ).strip(),
            "training_cli_checked": True,
            "path_for_training_data_contract": "prepared root containing list.csv and pdb/",
        },
        "NA-MPNN": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=upstream["NA-MPNN"], text=True
            ).strip(),
            "training_contract": "single positional JSON path",
            "base_folder_trailing_separator_checked": True,
            "evaluation_rna_column_order": "AUGC = DA,DT,DG,DC under shared tokens",
        },
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--third-party-root", type=Path, default=Path("third_party/checkouts"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.repo_root.resolve(), args.third_party_root.resolve())
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
