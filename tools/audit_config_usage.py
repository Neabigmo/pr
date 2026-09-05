#!/usr/bin/env python3
"""Fail if pilot.yaml contains a runtime-looking leaf that no code consumes.

Scientific invariants belong under ``protocol`` and are explicitly declarative.
``evaluation.primary_hypotheses`` is also declarative. Every other leaf must be
referenced by Python source under ``src/`` or ``tools/``; this prevents YAML
switches from masquerading as implemented behavior.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import re

import yaml


DECLARATIVE_EXACT = {"evaluation.primary_hypotheses"}
DECLARATIVE_PREFIXES = ("protocol.",)


def _flatten(value, prefix: str = "") -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(child, path))
    else:
        rows.append((prefix, value))
    return rows


def _source_text(repo_root: Path) -> str:
    texts = []
    for base in [repo_root / "src", repo_root / "tools"]:
        for path in sorted(base.rglob("*.py")):
            if path.name == "audit_config_usage.py":
                continue
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(texts)


def audit_config(config_path: Path, repo_root: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = _source_text(repo_root)
    rows = []
    unknown = []
    for path, value in _flatten(config):
        if path in DECLARATIVE_EXACT or path.startswith(DECLARATIVE_PREFIXES):
            status = "declarative_assertion"
        else:
            key = path.rsplit(".", 1)[-1]
            pattern = re.compile(rf"[\"']{re.escape(key)}[\"']")
            status = "runtime_consumed" if pattern.search(source) else "unknown_or_dead"
        record = {"path": path, "status": status, "value": value}
        rows.append(record)
        if status == "unknown_or_dead":
            unknown.append(path)
    return {"config": str(config_path), "entries": rows, "unknown_or_dead": unknown}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit_config(args.config, args.repo_root.resolve())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    if result["unknown_or_dead"]:
        raise SystemExit(
            "Dead/unknown pilot config keys detected: "
            + ", ".join(result["unknown_or_dead"])
        )


if __name__ == "__main__":
    main()
