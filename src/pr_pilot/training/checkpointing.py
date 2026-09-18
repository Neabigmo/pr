"""Small, crash-safe checkpoint manager for long scientific runs.

Each run keeps one resumable ``last.pt`` and one validation-selected
``best.pt``.  Epoch history belongs in JSONL, not in a directory full of
large model copies.  Final refits use ``final.pt`` and never overwrite the
validation-selected checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint beside its destination and replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore the RNG state saved by :class:`CheckpointManager`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    # ``map_location=cuda`` can move the serialized CPU RNG tensor onto the
    # selected device even though torch.set_rng_state requires CPU storage.
    if isinstance(torch_state, torch.Tensor) and torch_state.device.type != "cpu":
        torch_state = torch_state.cpu()
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [value.cpu() if isinstance(value, torch.Tensor) else value for value in state["cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


@dataclass
class CheckpointManager:
    """Persist one resumable state and one best model per run."""

    directory: Path
    selection_metric: str
    minimize: bool = True

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.directory / "metrics.jsonl"

    def append_metrics(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            handle.flush()

    def _last_payload(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "format": "scientific_checkpoint_v1",
            "kind": "last",
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "metadata": metadata,
            "rng": _rng_state(),
        }

    def save_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Save/replace ``last.pt`` and update ``best.pt`` on improvement."""
        metadata = dict(metadata or {})
        payload = self._last_payload(model, optimizer, scheduler, epoch, metrics, metadata)
        atomic_torch_save(payload, self.directory / "last.pt")
        self.append_metrics({"epoch": int(epoch), **metrics})

        value = float(metrics[self.selection_metric])
        previous = self.directory / "best.pt"
        improved = True
        if previous.exists():
            old = torch.load(previous, map_location="cpu", weights_only=False)
            old_value = float(old["metrics"][self.selection_metric])
            improved = value < old_value if self.minimize else value > old_value
        if improved:
            best = {
                "format": "scientific_checkpoint_v1",
                "kind": "best",
                "epoch": int(epoch),
                "model": model.state_dict(),
                "metrics": metrics,
                "metadata": metadata,
            }
            atomic_torch_save(best, previous)
        return improved

    def save_final(self, model: torch.nn.Module, epoch: int, metrics: dict[str, Any], metadata: dict[str, Any] | None = None) -> Path:
        """Save the final refit model without optimizer state."""
        path = self.directory / "final.pt"
        atomic_torch_save(
            {
                "format": "scientific_checkpoint_v1",
                "kind": "final",
                "epoch": int(epoch),
                "model": model.state_dict(),
                "metrics": metrics,
                "metadata": dict(metadata or {}),
            },
            path,
        )
        return path

    def load_last(self, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        path = self.directory / "last.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        return torch.load(path, map_location=map_location, weights_only=False)

    def restore_last(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        map_location: str | torch.device = "cpu",
    ) -> int:
        """Restore a complete run state and return the next epoch number."""
        payload = self.load_last(map_location)
        if payload.get("format") != "scientific_checkpoint_v1" or payload.get("kind") != "last":
            raise ValueError("unsupported or non-resumable checkpoint")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        restore_rng_state(payload["rng"])
        return int(payload["epoch"]) + 1
