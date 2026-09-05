#!/usr/bin/env python3
"""CPU-only contract preflight for the two pinned official baselines.

Run this after checkout/preparation and before spending GPU time.  The preflight
checks immutable SHAs, upstream entrypoint signatures, prepared data layout and
the exact command shapes used by ``run_official_baselines.py``.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

from pr_pilot.baselines.wrappers import ensure_lock_file, pinned_upstream

# When invoked as ``python tools/preflight_official_baselines.py`` the tools
# directory itself is on sys.path, not a Python package named ``tools``.
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from run_official_baselines import _na_command, _protein_command  # noqa: E402


def _head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def _argparse_flags(script: Path) -> set[str]:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                flags.add(arg.value)
    return flags


def _assert_protein_layout(root: Path) -> None:
    for name in ["list.csv", "valid_clusters.txt", "test_clusters.txt", "pdb"]:
        if not (root / name).exists():
            raise FileNotFoundError(root / name)


def _assert_na_config(path: Path) -> None:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ["BASE_FOLDER", "TOTAL_STEPS", "DF_PATH_TRAIN", "DF_PATH_VALID"]:
        if key not in cfg:
            raise ValueError(f"NA-MPNN config missing {key}")
    if int(cfg["TOTAL_STEPS"]) <= 0:
        raise ValueError("NA-MPNN TOTAL_STEPS must be positive")
    for key in ["DF_PATH_TRAIN", "DF_PATH_VALID"]:
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(cfg[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--third-party-root", type=Path, default=Path("third_party/checkouts")
    )
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    third = args.third_party_root.resolve()
    prepared = args.prepared.resolve()
    lock = ensure_lock_file(repo_root)

    protein_spec = pinned_upstream("ProteinMPNN", lock)
    na_spec = pinned_upstream("NA-MPNN", lock)
    protein_repo = third / "ProteinMPNN"
    na_repo = third / "NA-MPNN"
    if _head(protein_repo) != protein_spec.commit:
        raise RuntimeError("ProteinMPNN checkout does not match LOCK.json")
    if _head(na_repo) != na_spec.commit:
        raise RuntimeError("NA-MPNN checkout does not match LOCK.json")

    _assert_protein_layout(prepared / "proteinmpnn")
    na_cfg = prepared / "na_mpnn" / "na_mpnn_from_scratch.json"
    _assert_na_config(na_cfg)

    protein_script = protein_repo / "training" / "training.py"
    flags = _argparse_flags(protein_script)
    command = _protein_command(
        repo_root,
        protein_repo,
        prepared / "proteinmpnn",
        Path("/tmp/proteinmpnn_preflight_out"),
        seed=1,
        epochs=1,
        examples_per_epoch=1,
    )
    passed_flags = {
        x for x in command if isinstance(x, str) and x.startswith("--")
    }
    wrapper_flags = {
        "--seed",
        "--script",
        "--deterministic-empty-numpy-seed",
    }
    upstream_passed = passed_flags - wrapper_flags
    unsupported = sorted(upstream_passed - flags)
    if unsupported:
        raise ValueError(f"Unsupported ProteinMPNN training flags: {unsupported}")
    if str(prepared / "proteinmpnn" / "pdb") in command:
        raise AssertionError(
            "ProteinMPNN data root incorrectly points to pdb/ child"
        )

    na_script = na_repo / "na_run.py"
    source = na_script.read_text(encoding="utf-8")
    if "JSON = sys.argv[1]" not in source:
        raise RuntimeError(
            "Pinned NA-MPNN entrypoint contract changed; inspect before running"
        )
    na_command = _na_command(repo_root, na_repo, na_cfg, seed=1)
    separator = na_command.index("--")
    if na_command[separator + 1 :] != [str(na_cfg)]:
        raise AssertionError(
            "NA-MPNN training must receive exactly one positional JSON"
        )
    if "+'last.pt'" not in source and '+\"last.pt\"' not in source:
        raise RuntimeError(
            "Pinned NA-MPNN checkpoint contract changed; last.pt not found"
        )

    report = {
        "ProteinMPNN_commit": protein_spec.commit,
        "NA-MPNN_commit": na_spec.commit,
        "protein_layout": "ok",
        "protein_cli": "ok",
        "na_json_contract": "ok",
        "na_checkpoint_contract": "BASE_FOLDER/last.pt",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
