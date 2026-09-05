#!/usr/bin/env python3
"""Compatibility entrypoint for the hardened v2 pilot orchestrator.

The original implementation is retained in git history only.  Keeping two live
orchestrators created contradictory evaluation budgets and refit semantics, so
all executions now delegate to ``run_pilot_experiments_v2.py``.
"""
from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("run_pilot_experiments_v2.py")),
        run_name="__main__",
    )
