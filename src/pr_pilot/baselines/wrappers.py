"""External baseline wrappers.

We do not vendor upstream code or silently modify scientific baselines. Instead,
we pin an upstream checkout, adapt frozen manifests to its input format, execute
its documented training/inference entrypoints, and normalize outputs into this
project's evaluation schema.

The wrappers below intentionally use subprocess with explicit commands so every
run can be logged verbatim. The exact upstream entrypoint names can differ by
commit; therefore each adapter validates expected files and refuses to guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
from typing import Sequence


@dataclass(frozen=True)
class UpstreamRepo:
    name: str
    url: str
    pinned_commit: str
    checkout: Path

    def assert_ready(self) -> None:
        if not self.checkout.exists():
            raise FileNotFoundError(f"Missing checkout for {self.name}: {self.checkout}")
        git_dir = self.checkout / ".git"
        if not git_dir.exists():
            raise RuntimeError(f"{self.checkout} is not a git checkout")
        got = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.checkout, text=True).strip()
        if got != self.pinned_commit:
            raise RuntimeError(f"{self.name} commit mismatch: expected {self.pinned_commit}, got {got}")


def run_logged(command: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n\n")
        proc = subprocess.run(list(command), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}); inspect {log_path}")


class ProteinMPNNBaseline:
    upstream_url = "https://github.com/dauparas/ProteinMPNN"

    def __init__(self, repo: UpstreamRepo):
        if repo.url.rstrip("/") != self.upstream_url:
            raise ValueError("ProteinMPNN adapter must point to the official dauparas/ProteinMPNN repository")
        self.repo = repo

    def prepare(self, frozen_manifest: Path, output_dir: Path) -> Path:
        """Create an explicit adapter manifest; conversion code belongs in tools/.

        ProteinMPNN's upstream training data format is specialized. This pilot
        never fabricates fields. `tools/prepare_proteinmpnn.py` must derive every
        upstream record from the frozen manifest and preserve `sample_id`.
        """
        self.repo.assert_ready()
        output_dir.mkdir(parents=True, exist_ok=True)
        spec = output_dir / "adapter_request.json"
        spec.write_text(json.dumps({
            "baseline": "ProteinMPNN",
            "upstream_commit": self.repo.pinned_commit,
            "frozen_manifest": str(frozen_manifest),
            "require_same_1000_pool_as_dmicf": True,
            "no_test_data": True,
        }, indent=2), encoding="utf-8")
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
        spec.write_text(json.dumps({
            "baseline": "MPNN-fixbb/NA-MPNN",
            "upstream_commit": self.repo.pinned_commit,
            "frozen_manifest": str(frozen_manifest),
            "require_same_1000_pool_as_dmicf": True,
            "no_test_data": True,
            "rna_task": "fixed_backbone_sequence_design",
        }, indent=2), encoding="utf-8")
        return spec

    def train(self, command: Sequence[str], log_path: Path) -> None:
        self.repo.assert_ready()
        run_logged(command, self.repo.checkout, log_path)


def standardized_prediction_schema() -> dict[str, str]:
    """Schema that every baseline exporter must produce for fair evaluation."""
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
    }
