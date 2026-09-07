#!/usr/bin/env python3
"""Read-only full-repository integration gate for the offline Review-4 overlay.

A successful syntax scan is NOT a real-data smoke, full test pass, or paper result.
Unknown Brier call signatures are conservatively flagged for manual review.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path


def inspect_repository(root: Path) -> dict:
    blockers = []
    for filename in sorted(list((root / "src").rglob("*.py")) + list((root / "tools").rglob("*.py"))):
        relative = str(filename.relative_to(root))
        try:
            tree = ast.parse(filename.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeError) as exc:
            blockers.append({"file": relative, "reason": "parse_failure", "detail": str(exc)})
            continue
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
            name = aliases.get(name, name)
            if name == "native_probability_brier":
                reason = "native_only_probability_is_not_multiclass_brier"
            elif name == "brier_multiclass":
                if len(node.args) == 2 and not node.keywords and not any(isinstance(a, ast.Starred) for a in node.args):
                    continue
                reason = "legacy_or_unknown_brier_signature_requires_full_probability_migration"
            else:
                continue
            blockers.append({"file": relative, "line": node.lineno, "reason": reason})
    for relative in ["src/pr_pilot/training/engine.py", "src/pr_pilot/runtime/manifest_dataset.py",
                     "src/pr_pilot/evaluation/runner.py", "pyproject.toml"]:
        if not (root / relative).is_file():
            blockers.append({"file": relative, "reason": "full_repository_file_not_available"})
    return {"gate": "review4_static_integration", "passed": not blockers, "blockers": blockers,
            "scope_limit": "Static heuristic; dynamic call aliases, tensor schemas and real data require tests."}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    report = inspect_repository(args.repo.resolve())
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
