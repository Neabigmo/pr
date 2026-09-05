#!/usr/bin/env python3
"""Audited entrypoint for official-baseline training.

The implementation lives in ``run_official_baselines_core.py``. This shim fixes a
path contract that is easy to miss in the pinned NA-MPNN code: ``na_run.py``
concatenates ``BASE_FOLDER + 'log.txt'`` and therefore requires a trailing path
separator. The upstream checkout itself remains unmodified.
"""
from __future__ import annotations

import json
from pathlib import Path

import run_official_baselines_core as _core


# Re-export helpers used by preflight/tests.
clone_locked = _core.clone_locked
run_seed = _core.run_seed
_protein_command = _core._protein_command
_na_command = _core._na_command
_best_epoch = _core._best_epoch


def _write_na_run_config(template: Path, output_root: Path, destination: Path) -> Path:
    cfg = json.loads(template.read_text(encoding="utf-8"))
    # Pinned na_run.py uses string concatenation for log/checkpoint paths.
    cfg["BASE_FOLDER"] = str(output_root.resolve()) + "/"
    cfg["PREV_CHECKPOINT"] = ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    return destination


# run_seed resolves this helper through the core module's global namespace.
_core._write_na_run_config = _write_na_run_config


def main() -> None:
    _core.main()


if __name__ == "__main__":
    main()
