"""External baseline contracts and immutable upstream locking.

The mini-pilot never vendors or silently edits ProteinMPNN/NA-MPNN. Reported
runs are tied to exact upstream SHAs in ``third_party/LOCK.json``. Helpers in this
module intentionally fail closed when the lock is missing, contains placeholders,
or points to a different repository/commit than the checked-out code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
import subprocess
from typing import Sequence


_SHA40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class PinnedUpstream:
    name: str
    url: str
    commit: str
    checkout: str
    training_entrypoint: str | None = None


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
        got = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.checkout, text=True).strip()
        if got != self.pinned_commit:
            raise RuntimeError(f"{self.name} commit mismatch: expected {self.pinned_commit}, got {got}")


def ensure_lock_file(repo_root: Path) -> dict:
    """Load and strictly validate ``third_party/LOCK.json``.

    A template or moving branch name is never accepted for a reported experiment.
    """
    path = Path(repo_root) / "third_party" / "LOCK.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing immutable upstream lock {path}. Do not run reported baselines from moving branches."
        )
    lock = json.loads(path.read_text(encoding="utf-8"))
    for key in ("proteinmpnn", "rna_fixbb"):
        if key not in lock:
            raise ValueError(f"LOCK.json missing {key}")
        item = lock[key]
        commit = str(item.get("commit", "")).lower()
        if not _SHA40.fullmatch(commit):
            raise ValueError(f"LOCK.json {key}.commit is not an immutable 40-char SHA: {commit!r}")
        if "REPLACE_" in json.dumps(item):
            raise ValueError(f"LOCK.json {key} still contains template placeholders")
    return lock


def pinned_upstream(name: str, lock: dict) -> PinnedUpstream:
    """Resolve a human-facing baseline name to its immutable lock record."""
    aliases = {
        "ProteinMPNN": "proteinmpnn",
        "proteinmpnn": "proteinmpnn",
        "NA-MPNN": "rna_fixbb",
        "MPNN-fixbb / NA-MPNN": "rna_fixbb",
        "rna_fixbb": "rna_fixbb",
    }
    if name not in aliases:
        raise KeyError(f"Unknown pinned upstream {name!r}")
    item = lock[aliases[name]]
    return PinnedUpstream(
        name=str(item["name"]),
        url=str(item["repository"]).rstrip("/"),
        commit=str(item["commit"]).lower(),
        checkout=str(item.get("checkout", "")),
        training_entrypoint=item.get("training_entrypoint"),
    )


def run_logged(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(map(str, command)) + "\n\n")
        proc = subprocess.run(list(command), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}); inspect {log_path}")


class ProteinMPNNBaseline:
    upstream_url = "https://github.com/dauparas/ProteinMPNN"

    def __init__(self, repo: UpstreamRepo):
        if repo.url.rstrip("/") != self.upstream_url:
            raise ValueError("ProteinMPNN adapter must point to official dauparas/ProteinMPNN")
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

    def __init__(self, repo: UpstreamRepo):
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
                    "upstream_commit": self.repo.pinned_commit,
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
        "is_interface": "canonical heavy-atom 6A interface bool",
        "model": "string",
        "seed": "integer",
        "probability_semantics": "string",
    }
