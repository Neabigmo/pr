#!/usr/bin/env python3
"""Run a pinned upstream Python entrypoint under a controlled RNG seed.

This wrapper does *not* modify upstream source files.  It seeds Python, NumPy and
PyTorch before executing the upstream script with ``runpy``.  For ProteinMPNN's
legacy worker initializer, which calls ``numpy.random.seed()`` without an
argument, ``--deterministic-empty-numpy-seed`` converts that no-argument reseed
into a deterministic worker-specific seed.  This changes only randomness
initialization, not the model, loss, optimizer or data.
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


def _seed_everything(seed: int, deterministic_empty_numpy_seed: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_empty_numpy_seed:
        original = np.random.seed

        def seeded(value=None):
            if value is None:
                info = torch.utils.data.get_worker_info()
                worker_id = 0 if info is None else int(info.id)
                value = (seed + 10007 * worker_id) % (2**32)
            return original(value)

        np.random.seed = seeded  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--deterministic-empty-numpy-seed", action="store_true")
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script = args.script.resolve()
    if not script.exists():
        raise FileNotFoundError(script)
    upstream_args = list(args.upstream_args)
    if upstream_args and upstream_args[0] == "--":
        upstream_args = upstream_args[1:]

    _seed_everything(int(args.seed), bool(args.deterministic_empty_numpy_seed))
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *upstream_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
