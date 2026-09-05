"""Audited CLI wrapper.

The command parser/handlers live in ``cli_legacy.py``. This wrapper extends the
mandatory data audit with v3 prior-validation and downstream-backbone reuse checks.
"""
from __future__ import annotations

import json

import pandas as pd

from pr_pilot import cli_legacy as _legacy
from pr_pilot.cli_legacy import *  # noqa: F401,F403
from pr_pilot.data.manifest import assert_prior_train_val_disjoint


_original_cmd_audit_data = _legacy.cmd_audit_data


def cmd_audit_data(args) -> None:
    _original_cmd_audit_data(args)

    root = args.manifest_root
    out = args.out
    report = {"errors": [], "status": "pass"}
    protein_train = pd.read_csv(root / "protein_train.tsv", sep="\t")
    protein_val = pd.read_csv(root / "protein_val.tsv", sep="\t")
    rna_train = pd.read_csv(root / "rna_train.tsv", sep="\t")
    rna_val = pd.read_csv(root / "rna_val.tsv", sep="\t")
    rna_pool = pd.read_csv(root / "rna_pool.tsv", sep="\t")
    complex_pool = pd.read_csv(root / "complex_pool.tsv", sep="\t")

    try:
        assert_prior_train_val_disjoint(protein_train, protein_val, rna_train, rna_val)
    except Exception as exc:
        report["errors"].append(str(exc))

    if "source_complex_sample_id" in rna_pool.columns:
        frozen_complexes = set(complex_pool["sample_id"].astype(str))
        source = rna_pool["source_complex_sample_id"].fillna("").astype(str)
        leaked = sorted(set(source[source.str.len().gt(0)]) & frozen_complexes)
        if leaked:
            report["errors"].append(
                f"RNA prior contains {len(leaked)} extracted views sourced from frozen 1,100 complexes; "
                f"examples={leaked[:5]}"
            )

    if report["errors"]:
        report["status"] = "fail"
    (out / "prior_validation_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if report["errors"]:
        raise SystemExit("Prior-validation/downstream-view audit FAILED")
    print(
        "Prior audit passed: P30/R80/Rfam validation separation and downstream RNA-view purge."
    )


_legacy.cmd_audit_data = cmd_audit_data


def main() -> None:
    _legacy.main()


if __name__ == "__main__":
    main()
