#!/usr/bin/env python3
"""Run the pre-final-test DeltaC mean-drift audit on development data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pr_pilot.evaluation.delta_drift import audit_delta_c_drift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = audit_delta_c_drift(
        cfg,
        args.checkpoint,
        args.manifest,
        args.out,
        args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
