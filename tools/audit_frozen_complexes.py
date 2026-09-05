#!/usr/bin/env python3
"""Audit frozen complex manifests before GPU training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pr_pilot.data.frozen_audit import audit_frozen_complexes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = audit_frozen_complexes(
        args.manifest_root,
        float(cfg["structure_filters"]["interface_contact_angstrom"]),
        args.out,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
