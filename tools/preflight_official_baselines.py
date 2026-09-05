#!/usr/bin/env python3
"""CPU-only preflight for immutable official ProteinMPNN / NA-MPNN baselines.

This catches wrapper/upstream drift before any GPU time is spent. It validates the
pinned SHAs, expected upstream entrypoint behavior and the project's command/
probability mapping contracts. If checkouts are absent, use ``--clone`` to obtain
the exact locked revisions.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# When invoked as ``python tools/preflight_official_baselines.py``, Python places
# tools/ rather than the repository root at sys.path[0]. Add the root explicitly
# so this script and CI can import sibling tools as a namespace package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pr_pilot.baselines.wrappers import ensure_lock_file, pinned_upstream
from tools.evaluate_official_baselines import _rna_columns
from tools.run_official_baselines import _na_command, _protein_command, clone_locked


PROTEIN_ALLOWED = {
    "--path_for_training_data", "--path_for_outputs", "--previous_checkpoint", "--num_epochs",
    "--save_model_every_n_epochs", "--reload_data_every_n_epochs", "--num_examples_per_epoch",
    "--batch_size", "--max_protein_length", "--hidden_dim", "--num_encoder_layers",
    "--num_decoder_layers", "--num_neighbors", "--dropout", "--backbone_noise", "--rescut",
    "--debug", "--gradient_norm", "--mixed_precision",
}


def _forwarded_after_separator(command: list[str]) -> list[str]:
    if "--" not in command:
        raise AssertionError("Seeded upstream command lacks '--' separator")
    return command[command.index("--") + 1 :]


def _flags(tokens: list[str]) -> set[str]:
    return {x for x in tokens if x.startswith("--")}


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def preflight(repo_root: Path, third_party_root: Path, clone: bool = False) -> dict:
    lock = ensure_lock_file(repo_root)
    expected = {
        "ProteinMPNN": "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57",
        "NA-MPNN": "9fabc2482092b725e067969fba21297a806b6fda",
    }
    for name, sha in expected.items():
        got = pinned_upstream(name, lock).commit
        if got != sha:
            raise AssertionError(f"Pinned {name} SHA changed: expected audit SHA {sha}, got {got}")

    if clone:
        checkouts = clone_locked(repo_root, third_party_root)
    else:
        checkouts = {"ProteinMPNN": third_party_root / "ProteinMPNN", "NA-MPNN": third_party_root / "NA-MPNN"}
    checkout_status = {}
    for name, path in checkouts.items():
        if path.exists():
            spec = pinned_upstream(name, lock)
            head = _git_head(path)
            if head != spec.commit:
                raise AssertionError(f"{name} checkout {head} != locked {spec.commit}")
            entry = path / str(spec.training_entrypoint)
            if not entry.exists():
                raise FileNotFoundError(entry)
            checkout_status[name] = {"present": True, "head": head, "entrypoint": str(entry)}
        else:
            checkout_status[name] = {"present": False, "note": "run with --clone before training"}

    protein_cmd = _protein_command(repo_root, checkouts["ProteinMPNN"], Path("PREPARED_PROTEIN_ROOT"), Path("OUTPUT"), 7, 3, 900)
    p_forward = _forwarded_after_separator(protein_cmd)
    flags = _flags(p_forward)
    unknown = flags - PROTEIN_ALLOWED
    if unknown:
        raise AssertionError(f"ProteinMPNN wrapper contains unsupported flags: {sorted(unknown)}")
    if "--path_for_training_clusters" in flags or "--seed" in flags:
        raise AssertionError("ProteinMPNN upstream command leaked removed wrapper-only flags")
    idx = p_forward.index("--path_for_training_data")
    if p_forward[idx + 1] != "PREPARED_PROTEIN_ROOT":
        raise AssertionError("ProteinMPNN training data must point to prepared root, not pdb/ child")
    mp = p_forward.index("--mixed_precision")
    if p_forward[mp + 1] not in {"True", "False"}:
        raise AssertionError("ProteinMPNN mixed_precision requires an explicit bool value")

    na_cmd = _na_command(repo_root, checkouts["NA-MPNN"], Path("run.json"), 7)
    na_forward = _forwarded_after_separator(na_cmd)
    if na_forward != ["run.json"]:
        raise AssertionError(f"Pinned NA-MPNN na_run.py must receive one positional JSON, got {na_forward}")

    columns = _rna_columns({"DA": 0, "DC": 1, "DG": 2, "DT": 3})
    if columns != [0, 3, 2, 1]:
        raise AssertionError(f"NA-MPNN shared-token AUGC order is wrong: {columns}")

    return {
        "lock_valid": True,
        "locked_shas": {name: pinned_upstream(name, lock).commit for name in expected},
        "checkout_status": checkout_status,
        "protein_command_contract": "PASS",
        "na_training_positional_json_contract": "PASS",
        "na_augc_probability_mapping": "PASS",
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--third-party-root", type=Path, default=Path("third_party/checkouts"))
    parser.add_argument("--clone", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = preflight(args.repo_root.resolve(), args.third_party_root.resolve(), args.clone)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
