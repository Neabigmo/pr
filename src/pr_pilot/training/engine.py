"""Canonical audited training API.

The v3 orchestration implementation is in ``engine_v3_core.py`` and low-level
legacy tensor/corruption helpers are in ``engine_legacy.py``. This shim binds the
canonical full-heavy-atom interface cutoff from the resolved config into every
adapter construction, including legacy validation helpers.
"""
from __future__ import annotations

from pr_pilot.training import engine_v3_core as _core
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter


def _adapter(cfg: dict, epoch: int, training: bool) -> GemmiStructureAdapter:
    geometry = cfg["geometry"]
    sigma = float(geometry["coordinate_noise_angstrom"]) if training else 0.0
    return GemmiStructureAdapter(
        rbf_bins=int(geometry["rbf_bins"]),
        intra_max_neighbors=int(geometry["intra_max_neighbors"]),
        pr_cutoff_angstrom=float(geometry["pr_cutoff_angstrom"]),
        pr_max_neighbors=int(geometry["pr_max_neighbors"]),
        coordinate_noise_angstrom=sigma,
        seed=int(cfg["experiment"]["pilot_seed"]) + 1009 * epoch,
        rich_pr_geometry=bool(geometry["rich_pr_geometry"]),
        canonical_interface_cutoff_angstrom=float(
            cfg["structure_filters"]["interface_contact_angstrom"]
        ),
    )


# Resolve adapter construction dynamically inside both orchestration layers.
_core._adapter = _adapter
_core._legacy._adapter = _adapter

build_model_from_config = _core.build_model_from_config
_autocast = _core._autocast
_cosine_schedule = _core._cosine_schedule
_one_training_loss = _core._one_training_loss
_move_graph = _core._move_graph
_move_complex = _core._move_complex
_all_interface_corruption = _core._all_interface_corruption
sequential_joint_normalized_nll = _core.sequential_joint_normalized_nll
validate_stage = _core.validate_stage
train_stage = _core.train_stage

__all__ = [
    "build_model_from_config",
    "_adapter",
    "_autocast",
    "_cosine_schedule",
    "_one_training_loss",
    "_move_graph",
    "_move_complex",
    "_all_interface_corruption",
    "sequential_joint_normalized_nll",
    "validate_stage",
    "train_stage",
]
