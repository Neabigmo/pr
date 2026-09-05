#!/usr/bin/env python3
"""Run an unmodified upstream Python entrypoint under a deterministic seed.

This wrapper exists because the pinned ProteinMPNN training script does not expose
a seed argument and the pinned NA-MPNN training entrypoint accepts only a
positional JSON config. We keep upstream checkouts pristine and seed Python,
NumPy and PyTorch in-process before executing the script with ``runpy``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import runpy
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    script = ns.script.resolve()
    if not script.exists():
        raise FileNotFoundError(script)
    forwarded = list(ns.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    seed = int(ns.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
