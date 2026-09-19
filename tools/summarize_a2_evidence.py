#!/usr/bin/env python3
"""Aggregate the inference-only A2 evidence JSONL into paired summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _bootstrap(values: np.ndarray, seed: int, repeats: int = 10_000) -> list[float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [float("nan"), float("nan")]
    key = int.from_bytes(hashlib.sha256(f"{seed}|{values.size}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(key)
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    root = Path(args.out)
    native: dict[str, dict] = {}
    groups: dict[str, list[dict]] = {}
    for path in sorted(root.glob("fold*/details.jsonl")):
        for line in path.open("r", encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row["sample_id"])
            if row.get("variant") == "native":
                native[sid] = row
                continue
            label = str(row.get("variant", "unknown"))
            for field in ("sigma", "amount", "probability", "partner_polymer"):
                if field in row:
                    label += f"|{field}={row[field]}"
            groups.setdefault(label, []).append(row)

    summaries = {}
    for label, rows in sorted(groups.items()):
        per_complex: dict[str, list[dict]] = {}
        for row in rows:
            per_complex.setdefault(str(row["sample_id"]), []).append(row)
        deltas_p, deltas_r, ratios_p, ratios_r, kls_p, kls_r = [], [], [], [], [], []
        for sid, entries in per_complex.items():
            ref = native.get(sid)
            if ref is None:
                continue
            p_nll = _mean([x.get("protein_interface_nll", np.nan) for x in entries])
            r_nll = _mean([x.get("rna_interface_nll", np.nan) for x in entries])
            deltas_p.append(p_nll - float(ref["protein_interface_nll"]))
            deltas_r.append(r_nll - float(ref["rna_interface_nll"]))
            ratios_p.append(_mean([x.get("protein_interface_ratio", np.nan) for x in entries]))
            ratios_r.append(_mean([x.get("rna_interface_ratio", np.nan) for x in entries]))
            kls_p.append(_mean([x.get("protein_interface_kl_to_native", np.nan) for x in entries]))
            kls_r.append(_mean([x.get("rna_interface_kl_to_native", np.nan) for x in entries]))
        p_delta = np.asarray(deltas_p, dtype=float)
        r_delta = np.asarray(deltas_r, dtype=float)
        summaries[label] = {
            "rows": len(rows), "complexes": len(per_complex),
            "protein_interface_delta_nll": _mean(deltas_p), "rna_interface_delta_nll": _mean(deltas_r),
            "protein_interface_ratio": _mean(ratios_p), "rna_interface_ratio": _mean(ratios_r),
            "protein_interface_kl_to_native": _mean(kls_p), "rna_interface_kl_to_native": _mean(kls_r),
            "protein_delta_ci95": _bootstrap(p_delta, args.seed), "rna_delta_ci95": _bootstrap(r_delta, args.seed + 1),
            "native_better_than_perturbed": {
                "protein": float(np.mean(p_delta > 0)) if p_delta.size else float("nan"),
                "rna": float(np.mean(r_delta > 0)) if r_delta.size else float("nan"),
            },
            "composition": "changed_by_design" if label.startswith("single_site_mutation") else "preserved",
            "active_mask": "preserved",
            "test_read": False,
        }
    result = {
        "protocol": "A2 inference-only evidence; development validation folds only",
        "test_read": False, "native_complexes": len(native), "experiments": summaries,
        "bootstrap_repeats": 10_000,
    }
    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(json.dumps({"native_complexes": len(native), "experiment_groups": len(summaries), "test_read": False}))


if __name__ == "__main__":
    main()
