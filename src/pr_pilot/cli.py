"""Command-line entrypoints for the pilot.

The CLI is intentionally conservative: operations that would consume real
structures are routed through explicit adapter functions. If an adapter is not
implemented for a local dataset schema, the command fails with a precise error
rather than generating placeholder results.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd
import yaml

from pr_pilot.data.manifest import FrozenCounts, freeze_single_molecule_pool, freeze_complex_pool, assert_no_test_leakage
from pr_pilot.evaluation.battery import mandatory_test_registry


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_freeze(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    seed = int(cfg["experiment"]["pilot_seed"])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    freeze_single_molecule_pool(args.proteins, out, "protein", seed)
    freeze_single_molecule_pool(args.rnas, out, "rna", seed + 101)
    freeze_complex_pool(args.complexes, out, seed + 202, require_strict_bilateral=cfg["experiment"]["strict_mode"])
    print(f"Frozen manifests written to {out}")


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def cmd_audit_data(args: argparse.Namespace) -> None:
    root = args.manifest_root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    expected = {
        "protein_pool.tsv": 1000,
        "protein_train.tsv": 900,
        "protein_val.tsv": 100,
        "rna_pool.tsv": 1000,
        "rna_train.tsv": 900,
        "rna_val.tsv": 100,
        "complex_pool.tsv": 1100,
        "complex_dev.tsv": 1000,
        "complex_train.tsv": 900,
        "complex_val.tsv": 100,
        "complex_test.tsv": 100,
    }
    report = {"counts": {}, "errors": [], "warnings": []}
    for name, n in expected.items():
        path = root / name
        if not path.exists():
            report["errors"].append(f"missing {name}")
            continue
        got = len(_read_tsv(path))
        report["counts"][name] = got
        if got != n:
            report["errors"].append(f"{name}: expected {n}, got {got}")

    if not report["errors"]:
        train = _read_tsv(root / "complex_train.tsv")
        val = _read_tsv(root / "complex_val.tsv")
        test = _read_tsv(root / "complex_test.tsv")
        try:
            assert_no_test_leakage(train, val, test, strict_cluster_check=True)
        except Exception as exc:
            report["errors"].append(str(exc))

        if not test["experimental"].astype(bool).all():
            report["errors"].append("Final 100 test set contains non-experimental structures")

    (out / "manifest_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    mandatory_test_registry().to_csv(out / "mandatory_test_registry.tsv", sep="\t", index=False)
    if report["errors"]:
        raise SystemExit("Data audit FAILED; inspect manifest_audit.json")
    print("Data audit passed. Geometry/interface audits require the local structure adapter and must run before training.")


def cmd_train(args: argparse.Namespace) -> None:
    # Training is routed to a project-local structure dataset adapter because raw
    # structure storage differs across installations. We fail rather than pretend
    # to train without a parser. The model/loss/stage code is already implemented.
    raise SystemExit(
        "Training command reached before local structure adapter registration. "
        "Implement pr_pilot.runtime.dataset_adapter.load_structure_sample for your frozen mmCIF/PDB layout, "
        "then connect it to the generic stage loop. See docs/IMPLEMENTATION_CONTRACT.md."
    )


def cmd_registry(args: argparse.Namespace) -> None:
    print(mandatory_test_registry().to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pr-pilot")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("freeze")
    f.add_argument("--config", type=Path, required=True)
    f.add_argument("--proteins", type=Path, required=True)
    f.add_argument("--rnas", type=Path, required=True)
    f.add_argument("--complexes", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    f.set_defaults(func=cmd_freeze)

    a = sub.add_parser("audit-data")
    a.add_argument("--config", type=Path, required=True)
    a.add_argument("--manifest-root", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.set_defaults(func=cmd_audit_data)

    t = sub.add_parser("train")
    t.add_argument("--stage", required=True, choices=["protein_prior", "rna_prior", "global_c", "delta_c", "alpha", "joint"])
    t.add_argument("--config", type=Path, required=True)
    t.add_argument("--manifest", type=Path, required=True)
    t.add_argument("--validation", type=Path, required=True)
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("test-registry")
    r.set_defaults(func=cmd_registry)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
