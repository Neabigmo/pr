"""External baseline wrappers and immutable upstream lock handling.

The pilot never vendors or silently edits ProteinMPNN / NA-MPNN.  Instead it
pins exact upstream commits, converts the frozen manifests to the documented
input formats, runs the upstream entrypoints, and converts outputs to one common
per-position schema.

A baseline run is invalid if the checkout HEAD differs from ``third_party/LOCK.json``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
from typing import Sequence


@dataclass(frozen=True)
class PinnedUpstream:
    name: str
    url: str
    commit: str
    checkout: Path

    def assert_ready(self) -> None:
        if not self.checkout.exists():
            raise FileNotFoundError(f"Missing checkout for {self.name}: {self.checkout}")
        if not (self.checkout / ".git").exists():
            raise RuntimeError(f"{self.checkout} is not a git checkout")
        got = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.checkout, text=True
        ).strip()
        if got != self.commit:
            raise RuntimeError(
                f"{self.name} commit mismatch: expected {self.commit}, got {got}"
            )


# Backward-compatible alias used by older helper code.
UpstreamRepo = PinnedUpstream


def ensure_lock_file(repo_root: Path) -> dict:
    """Load and validate the immutable baseline lock.

    ``LOCK.template.json`` is documentation only.  Reported experiments must use
    ``LOCK.json`` with full 40-character commit SHAs.
    """
    path = Path(repo_root) / "third_party" / "LOCK.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. The repository ships a pinned LOCK.json; restore it "
            "rather than running an unversioned upstream baseline."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("proteinmpnn", "rna_fixbb"):
        if key not in payload:
            raise ValueError(f"{path} missing {key!r}")
        entry = payload[key]
        sha = str(entry.get("commit", "")).strip()
        if len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha.lower()):
            raise ValueError(f"{path}:{key}.commit is not an immutable 40-char SHA")
        if "REPLACE_" in sha.upper():
            raise ValueError(f"{path}:{key}.commit still contains a placeholder")
    return payload


def pinned_upstream(name: str, lock: dict) -> PinnedUpstream:
    """Resolve one canonical baseline name from ``LOCK.json``."""
    aliases = {
        "ProteinMPNN": "proteinmpnn",
        "proteinmpnn": "proteinmpnn",
        "NA-MPNN": "rna_fixbb",
        "MPNN-fixbb": "rna_fixbb",
        "rna_fixbb": "rna_fixbb",
    }
    if name not in aliases:
        raise KeyError(f"Unknown upstream baseline {name!r}")
    key = aliases[name]
    entry = lock[key]
    return PinnedUpstream(
        name=str(entry["name"]),
        url=str(entry["repository"]).rstrip("/"),
        commit=str(entry["commit"]),
        checkout=Path(str(entry["checkout"])),
    )


def run_logged(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(map(str, command)) + "\n\n")
        proc = subprocess.run(
            list(command), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}); inspect {log_path}")


class ProteinMPNNBaseline:
    upstream_url = "https://github.com/dauparas/ProteinMPNN"

    def __init__(self, repo: PinnedUpstream):
        if repo.url.rstrip("/") != self.upstream_url:
            raise ValueError(
                "ProteinMPNN adapter must point to the official dauparas/ProteinMPNN repository"
            )
        self.repo = repo

    def prepare(self, frozen_manifest: Path, output_dir: Path) -> Path:
        self.repo.assert_ready()
        output_dir.mkdir(parents=True, exist_ok=True)
        spec = output_dir / "adapter_request.json"
        spec.write_text(
            json.dumps(
                {
                    "baseline": "ProteinMPNN",
                    "upstream_commit": self.repo.commit,
                    "frozen_manifest": str(frozen_manifest),
                    "require_same_1000_pool_as_dmicf": True,
                    "no_test_data": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return spec

    def train(self, command: Sequence[str], log_path: Path) -> None:
        self.repo.assert_ready()
        run_logged(command, self.repo.checkout, log_path)


class MPNNFixbbRNABaseline:
    upstream_url = "https://github.com/baker-laboratory/NA-MPNN"

    def __init__(self, repo: PinnedUpstream):
        if repo.url.rstrip("/") != self.upstream_url:
            raise ValueError("RNA fixbb adapter must point to baker-laboratory/NA-MPNN")
        self.repo = repo

    def prepare(self, frozen_manifest: Path, output_dir: Path) -> Path:
        self.repo.assert_ready()
        output_dir.mkdir(parents=True, exist_ok=True)
        spec = output_dir / "adapter_request.json"
        spec.write_text(
            json.dumps(
                {
                    "baseline": "MPNN-fixbb/NA-MPNN",
                    "upstream_commit": self.repo.commit,
                    "frozen_manifest": str(frozen_manifest),
                    "require_same_1000_pool_as_dmicf": True,
                    "no_test_data": True,
                    "rna_task": "fixed_backbone_sequence_design",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return spec

    def train(self, command: Sequence[str], log_path: Path) -> None:
        self.repo.assert_ready()
        run_logged(command, self.repo.checkout, log_path)


def standardized_prediction_schema() -> dict[str, str]:
    return {
        "sample_id": "string",
        "polymer": "protein|rna",
        "position": "0-based integer",
        "native_token": "canonical token",
        "predicted_token": "canonical token",
        "native_log_probability": "float",
        "max_probability": "float",
        "is_interface": "bool when applicable",
        "model": "string",
        "seed": "integer",
        "probability_semantics": "string",
    }
