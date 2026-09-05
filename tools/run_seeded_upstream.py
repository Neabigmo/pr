#!/usr/bin/env python3
"""Run an unmodified upstream Python script under a recorded main-process seed.

This wrapper exists because some pinned research repositories do not expose a
seed argument in their CLI. It deliberately does *not* edit the upstream checkout.
It seeds Python, NumPy and PyTorch in the current process, rewrites ``sys.argv``
to the upstream script arguments, and executes the script with ``runpy``.

Important limitation
--------------------
An upstream script may itself create worker processes that reseed from OS entropy.
For example, the pinned ProteinMPNN training loader calls ``np.random.seed()`` in
workers. Therefore this wrapper makes the main process reproducible but does not
claim bitwise determinism for third-party code that explicitly overrides worker
seeds. The pilot uses three independent baseline runs and records this limitation.
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


def seed_process(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script = args.script.resolve()
    if not script.exists():
        raise FileNotFoundError(script)
    trailing = list(args.script_args)
    if trailing and trailing[0] == "--":
        trailing = trailing[1:]

    seed_process(int(args.seed))
    sys.argv = [str(script), *trailing]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
