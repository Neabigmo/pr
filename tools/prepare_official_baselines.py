#!/usr/bin/env python3
"""Audited wrapper for official-baseline data conversion."""
from __future__ import annotations

import json
from pathlib import Path

import prepare_official_baselines_core as _core


read_manifest = _core.read_manifest
prepare_proteinmpnn = _core.prepare_proteinmpnn
extract_matching_chain = _core.extract_matching_chain
write_rna_only_pdb = _core.write_rna_only_pdb
_original_prepare_nampnn = _core.prepare_nampnn


def prepare_nampnn(
    train_path: Path,
    val_path: Path,
    out: Path,
    passes: int = 150,
    batch_tokens: int = 6000,
) -> dict:
    result = _original_prepare_nampnn(train_path, val_path, out, passes, batch_tokens)
    config_path = Path(result["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Pinned na_run.py concatenates BASE_FOLDER with filenames; preserve separator.
    config["BASE_FOLDER"] = str((out / "model").resolve()) + "/"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return result


# Core main resolves this name at call time.
_core.prepare_nampnn = prepare_nampnn


def main() -> None:
    _core.main()


if __name__ == "__main__":
    main()
