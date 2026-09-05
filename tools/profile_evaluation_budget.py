#!/usr/bin/env python3
"""Profile joint inference on development data before final100 is opened."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pr_pilot.evaluation.runtime_profile import profile_inference_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--profile-candidates", type=int, default=2)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = profile_inference_budget(
        cfg,
        args.checkpoint,
        args.manifest,
        args.out,
        args.device,
        args.profile_candidates,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
