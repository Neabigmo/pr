#!/usr/bin/env python3
"""Summarize the locked A2 development CV across preregistered seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    args = parser.parse_args()
    metrics = {}
    seeds = []
    for root in args.roots:
        path = Path(root) / "cv_summary.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        protocol = data.get("protocol", {})
        seeds.append(int(protocol.get("seed", -1)))
        item = data["experiments"].get("A2")
        if item is None:
            raise ValueError(f"A2 missing from {path}")
        for key, value in item["mean_metrics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics.setdefault(key, []).append(float(value))
    aggregate = {}
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        aggregate[key] = {"n_seeds": int(arr.size), "mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0, "values": [float(x) for x in arr]}
    result = {"protocol": "A2 fixed architecture, 3 independent development seeds", "seeds": seeds, "test_read": False, "metrics": aggregate}
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"seeds": seeds, "test_read": False}))


if __name__ == "__main__":
    main()
