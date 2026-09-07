#!/usr/bin/env python3
"""Run an unmodified upstream Python entrypoint under a deterministic seed.

The wrapper keeps third-party checkouts pristine. It seeds Python/NumPy/PyTorch,
adds the upstream script directory to ``sys.path`` so its local imports behave as
in direct execution, forwards argv verbatim, then executes with ``runpy``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import os
from pathlib import Path
import random
import runpy
import sys

import numpy as np
import torch


class _InlineFutureExecutor:
    """Drop-in synchronous executor for memory-constrained local upstream runs."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def submit(self, function, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:  # Future.result() re-raises in upstream code.
            future.set_exception(exc)
        return future


def _patch_serial_resource_profile() -> None:
    """Keep an unmodified upstream checkout within Windows host resources."""
    original_loader = torch.utils.data.DataLoader

    class SingleProcessDataLoader(original_loader):
        def __init__(self, *args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs.pop("persistent_workers", None)
            super().__init__(*args, **kwargs)

    torch.utils.data.DataLoader = SingleProcessDataLoader
    concurrent.futures.ProcessPoolExecutor = _InlineFutureExecutor


def _patch_na_data_paths(data_root: Path) -> None:
    """Redirect NA-MPNN's checked-in Linux-only data literals to local assets."""
    import builtins

    redirects = {
        "/home/akubaney/projects/na_mpnn/data/datasets/rcsb_cif/ligands.json.gz": data_root / "ligands.json.gz",
        "/home/akubaney/projects/na_mpnn/data/datasets/rcsb_cif/elements.txt": data_root / "elements.txt",
        "/home/aivan/git/chemnet/arch.22-10-28/data/elements.txt": data_root / "elements.txt",
    }
    original_open = builtins.open
    original_gzip_open = gzip.open

    def redirected_open(file, *args, **kwargs):
        target = redirects.get(str(file), file)
        return original_open(target, *args, **kwargs)

    def redirected_gzip_open(filename, *args, **kwargs):
        target = redirects.get(str(filename), filename)
        return original_gzip_open(target, *args, **kwargs)

    builtins.open = redirected_open
    gzip.open = redirected_gzip_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument(
        "--serial-resource-profile",
        action="store_true",
        help="Monkey-patch upstream DataLoader/process executor to one host process without editing the checkout.",
    )
    parser.add_argument(
        "--redirect-na-data-root",
        type=Path,
        help="Redirect NA-MPNN's hard-coded ligand/element paths to this local dependency directory.",
    )
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
    if ns.serial_resource_profile:
        _patch_serial_resource_profile()
    if ns.redirect_na_data_root:
        _patch_na_data_paths(ns.redirect_na_data_root.resolve())

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
