"""Contracts for pinned external baselines.

The pilot never vendors or silently edits upstream scientific baselines. A reported
run must use ``third_party/LOCK.json`` with immutable commit SHAs, clone those exact
commits into an external checkout, and record every command. This module contains
only lock validation and lightweight subprocess helpers; data conversion lives in
``tools/prepare_official_baselines.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import subprocess
from typing import Sequence


OFFICIAL = {
    "ProteinMPNN": ("proteinmpnn", "https://github.com/dauparas/ProteinMPNN"),
    "NA-MPNN": ("rna_fixbb", "https://github.com/baker-laboratory/NA-MPNN"),
}
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class PinnedUpstream:
    name: str
    url: str
    commit: str


@dataclass(frozen=True)
class UpstreamRepo:
    name: str
    url: str
    pinned_commit: str
    checkout: Path

    def assert_ready(self) -> None:
        if not self.checkout.exists():
            raise FileNotFoundError(f"Missing checkout for {self.name}: {self.checkout}")
        if not (self.checkout / ".git").exists():
            raise RuntimeError(f"{self.checkout} is not a git checkout")
        got = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.checkout, text=True
        ).strip()
        if got != self.pinned_commit:
            raise RuntimeError(
                f"{self.name} commit mismatch: expected {self.pinned_commit}, got {got}"
            )


def ensure_lock_file(repo_root: Path) -> dict:
    """Load and strictly validate ``third_party/LOCK.json``.

    Template placeholders are intentionally rejected. A reported run must never
    silently float to whatever upstream ``main`` happens to contain that day.
    """
    path = Path(repo_root) / "third_party" / "LOCK.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Copy third_party/LOCK.template.json to LOCK.json, "
            "replace both placeholders with immutable 40-character SHAs, and review them before running."
        )
    lock = json.loads(path.read_text(encoding="utf-8"))
    for name, (key, expected_url) in OFFICIAL.items():
        if key not in lock:
            raise ValueError(f"LOCK.json missing section {key!r} for {name}")
        item = lock[key]
        repo = str(item.get("repository", "")).rstrip("/")
        commit = str(item.get("commit", "")).strip()
        if repo != expected_url:
            raise ValueError(f"{name} must use official repository {expected_url}; got {repo!r}")
        if not SHA_RE.fullmatch(commit):
            raise ValueError(
                f"{name} commit must be an immutable 40-character SHA; got {commit!r}"
            )
    return lock


def pinned_upstream(name: str, lock: dict) -> PinnedUpstream:
    if name not in OFFICIAL:
        raise KeyError(f"Unknown official baseline {name!r}")
    key, expected_url = OFFICIAL[name]
    item = lock[key]
    return PinnedUpstream(name, expected_url, str(item["commit"]).strip())


def run_logged(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(map(str, command)) + "\n\n")
        proc = subprocess.run(
            list(map(str, command)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}); inspect {log_path}")


class ProteinMPNNBaseline:
    upstream_url = OFFICIAL["ProteinMPNN"][1]

    def __init__(self, repo: UpstreamRepo):
        if repo.url.rstrip("/") != self.upstream_url:
            raise ValueError("ProteinMPNN adapter must point to dauparas/ProteinMPNN")
        self.repo = repo

    def prepare(self, frozen_manifest: Path, output_dir: Path) -> Path:
        self.repo.assert_ready()
        output_dir.mkdir(parents=True, exist_ok=True)
        spec = output_dir / "adapter_request.json"
        spec.write_text(
            json.dumps(
                {
                    "baseline": "ProteinMPNN",
                    "upstream_commit": self.repo.pinned_commit,
                    "frozen_manifest": str(frozen_manifest),
                    "require_same_pool_as_dmicf": True,
                    "no_final_test_data": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return spec


class MPNNFixbbRNABaseline:
    upstream_url = OFFICIAL["NA-MPNN"][1]

    def __init__(self, repo: UpstreamRepo):
        if repo.url.rstrip("/") != self.upstream_url:
            raise ValueError("RNA fixed-backbone adapter must point to baker-laboratory/NA-MPNN")
        self.repo = repo

    def prepare(self, frozen_manifest: Path, output_dir: Path) -> Path:
        self.repo.assert_ready()
        output_dir.mkdir(parents=True, exist_ok=True)
        spec = output_dir / "adapter_request.json"
        spec.write_text(
            json.dumps(
                {
                    "baseline": "MPNN-fixbb/NA-MPNN",
                    "upstream_commit": self.repo.pinned_commit,
                    "frozen_manifest": str(frozen_manifest),
                    "require_same_pool_as_dmicf": True,
                    "no_final_test_data": True,
                    "rna_task": "fixed_backbone_sequence_design",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return spec


def standardized_prediction_schema() -> dict[str, str]:
    return {
        "sample_id": "string",
        "polymer": "protein|rna",
        "position": "0-based integer",
        "native_token": "canonical token index",
        "predicted_token": "canonical token index",
        "native_log_probability": "float",
        "max_probability": "float",
        "is_interface": "canonical full-heavy-atom interface bool",
        "model": "string",
        "seed": "integer",
        "probability_semantics": "string",
    }
