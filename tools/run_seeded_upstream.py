#!/usr/bin/env python3
"""Run an unmodified upstream Python entrypoint under a deterministic seed.

The wrapper keeps third-party checkouts pristine. It seeds Python/NumPy/PyTorch,
adds the upstream script directory to ``sys.path`` so its local imports behave as
in direct execution, forwards argv verbatim, then executes with ``runpy``.
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

    # Direct `python path/to/script.py` puts that script's directory at sys.path[0].
    # runpy does not guarantee that behavior, yet the pinned ProteinMPNN/NA-MPNN
    # entrypoints import sibling modules (e.g. `utils`, `cifutils`). Reproduce the
    # direct-execution import contract without modifying upstream source.
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
