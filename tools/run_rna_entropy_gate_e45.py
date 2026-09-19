#!/usr/bin/env python3
"""Run only the registered E4/E5 scalar-gate entropy experiments.

The implementation reuses the validated E0--E3 runner but changes only the
experiment registry and output root.  E4 keeps the learned RNA scalar gate
and multiplies it by normalized prior entropy.  E5 keeps the scalar gate and
uses a fold-training-only raw-entropy median threshold.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

import run_rna_entropy_balance as runner


runner.OUT = Path(r"I:\PR_PILOT_SCIENTIFIC\20260919\rna_entropy_gate_e45")
runner.EXPERIMENTS = {
    "E4_original_entropy_scalar": {
        "rna_gate_mode": "entropy_scalar",
        "balanced": False,
    },
    "E5_original_entropy_threshold": {
        "rna_gate_mode": "entropy_threshold_scalar",
        "balanced": False,
    },
}


if __name__ == "__main__":
    runner.main()
