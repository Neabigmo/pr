"""Stage orchestration, parameter ownership and optimizer contracts for DM-ICF.

Primary scientific ownership is intentionally strict:
C -> contextual q/DeltaC -> relevance alpha -> joint coordination.  The global C
anchor is frozen after Stage C so later modules cannot silently redefine the object
used for post-hoc biological interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import nn

from pr_pilot.model.dmicf import (
    JointPriorAndFieldModel,
    SimpleSparseBackboneEncoder,
    set_trainable_stage,
)


class Stage(str, Enum):
    PROTEIN_PRIOR = "protein_prior"
    RNA_PRIOR = "rna_prior"
    GLOBAL_C = "global_c"
    DELTA_C = "delta_c"
    ALPHA = "alpha"
    JOINT = "joint"


@dataclass(frozen=True)
class StageContract:
    stage: Stage
    dataset: str
    allowed_tasks: tuple[str, ...]
    checkpoint_metric: str
    forbidden_inputs: tuple[str, ...] = ()


CONTRACTS = {
    Stage.PROTEIN_PRIOR: StageContract(
        Stage.PROTEIN_PRIOR,
        "protein_train",
        ("protein_inverse_folding",),
        "protein_val_normalized_nll",
        ("rna_partner_sequence", "pr_edges"),
    ),
    Stage.RNA_PRIOR: StageContract(
        Stage.RNA_PRIOR,
        "rna_train",
        ("rna_inverse_folding",),
        "rna_val_normalized_nll",
        ("protein_partner_sequence", "pr_edges", "native_base_identity_atoms"),
    ),
    Stage.GLOBAL_C: StageContract(
        Stage.GLOBAL_C,
        "complex_train",
        ("protein_conditional_interface", "rna_conditional_interface"),
        "complex_val_bidirectional_interface_nll",
        ("predicted_structures", "test_manifest", "learned_delta_c", "learned_alpha"),
    ),
    Stage.DELTA_C: StageContract(
        Stage.DELTA_C,
        "complex_train",
        ("protein_conditional_interface", "rna_conditional_interface"),
        "complex_val_bidirectional_interface_nll",
        ("test_manifest", "learned_alpha"),
    ),
    Stage.ALPHA: StageContract(
        Stage.ALPHA,
        "complex_train",
        ("protein_conditional_interface", "rna_conditional_interface"),
        "complex_val_bidirectional_interface_nll",
        ("test_manifest",),
    ),
    Stage.JOINT: StageContract(
        Stage.JOINT,
        "complex_train",
        ("protein_conditional", "rna_conditional", "joint"),
        "complex_val_conditional_plus_sequential_joint_nll",
        ("test_manifest",),
    ),
}


def _set_module(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def _set_parameter(parameter: nn.Parameter, value: bool) -> None:
    parameter.requires_grad = value


def apply_joint_unfreezing(
    model: JointPriorAndFieldModel,
    progress: float,
    mode: str = "pretrained_gradual",
) -> dict[str, int]:
    """Configure joint-stage trainability without ever unfreezing global C.

    ``pretrained_gradual`` is the primary DM-ICF path.  ``all_trainable_from_start``
    is reserved for the random-initialized scratch control so that it is not
    handicapped by freezing random encoder layers.
    """
    progress = float(min(max(progress, 0.0), 1.0))
    if mode not in {"pretrained_gradual", "all_trainable_from_start"}:
        raise ValueError(f"Unknown joint unfreezing mode {mode!r}")

    # Joint heads/context field may coordinate; Stage-C global anchor cannot drift.
    for module in [
        model.dmicf.interaction,
        model.dmicf.delta,
        model.dmicf.relevance,
        model.protein_decoder,
        model.rna_decoder,
        model.protein_head,
        model.rna_head,
    ]:
        _set_module(module, True)
    _set_module(model.dmicf.global_c, False)
    _set_parameter(model.dmicf.raw_lambda_p, True)
    _set_parameter(model.dmicf.raw_lambda_r, True)

    released: dict[str, int] = {}
    for name, encoder in [("protein", model.protein_encoder), ("rna", model.rna_encoder)]:
        if mode == "all_trainable_from_start":
            _set_module(encoder, True)
            released[name] = len(encoder.message)
            continue

        _set_module(encoder, False)
        n_layers = len(encoder.message)
        n_release = min(n_layers, max(1, int(math.ceil(progress * n_layers))))
        for idx in range(n_layers - n_release, n_layers):
            _set_module(encoder.message[idx], True)
            _set_module(encoder.update[idx], True)
        _set_module(encoder.norm, True)
        if progress >= 0.80:
            _set_module(encoder.node_proj, True)
            _set_module(encoder.edge_proj, True)
        released[name] = n_release
    return released


def configure_stage(
    model: JointPriorAndFieldModel,
    stage: Stage,
    joint_unfreezing_mode: str = "pretrained_gradual",
) -> StageContract:
    """Apply the primary stage ownership contract."""
    set_trainable_stage(model, stage.value)
    if stage == Stage.ALPHA:
        # The contextual matrix is already learned. Alpha alone answers “who matters?”.
        _set_module(model.dmicf.interaction, False)
        _set_module(model.dmicf.delta, False)
        _set_module(model.dmicf.relevance, True)
    elif stage == Stage.JOINT:
        apply_joint_unfreezing(model, 0.0, mode=joint_unfreezing_mode)
    return CONTRACTS[stage]


def trainable_parameter_report(model: nn.Module) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    for name, param in model.named_parameters():
        parts = name.split(".")
        family = ".".join(parts[:2]) if parts[0] == "dmicf" and len(parts) > 1 else parts[0]
        bucket = report.setdefault(family, {"trainable": 0, "frozen": 0})
        bucket["trainable" if param.requires_grad else "frozen"] += param.numel()
    return report


def _params(module: nn.Module, trainable_only: bool = True) -> list[nn.Parameter]:
    return [p for p in module.parameters() if (p.requires_grad or not trainable_only)]


def _group(params: list[nn.Parameter], lr: float, wd: float) -> dict:
    return {"params": params, "lr": lr, "weight_decay": wd}


def _encoder_layer_groups(
    encoder: SimpleSparseBackboneEncoder,
    top_lr: float,
    bottom_lr: float,
    decay: float,
    wd: float,
) -> list[dict]:
    """Layer-wise discriminative LR, including frozen params for later release."""
    n_layers = len(encoder.message)
    groups: list[dict] = []
    for idx in range(n_layers):
        depth_from_top = n_layers - 1 - idx
        lr = max(bottom_lr, top_lr * (decay**depth_from_top))
        groups.append(
            _group(
                _params(encoder.message[idx], False) + _params(encoder.update[idx], False),
                lr,
                wd,
            )
        )
    groups.append(_group(_params(encoder.norm, False), max(bottom_lr, top_lr * decay), wd))
    groups.append(
        _group(_params(encoder.node_proj, False) + _params(encoder.edge_proj, False), bottom_lr, wd)
    )
    return groups


def build_optimizer(
    model: JointPriorAndFieldModel,
    stage: Stage,
    lr_heads: float = 1e-3,
    lr_projections: float = 5e-4,
    lr_encoder_top: float = 1e-4,
    lr_encoder_bottom: float = 2e-5,
    lr_global_c_joint: float = 0.0,
    weight_decay: float = 1e-2,
    layerwise_lr_decay: float = 0.85,
) -> torch.optim.Optimizer:
    """Construct non-overlapping parameter groups for the active stage.

    ``lr_global_c_joint`` is retained as an API-compatibility argument but the
    primary protocol intentionally does not place C in the joint optimizer.
    """
    del lr_global_c_joint
    groups: list[dict] = []
    if stage == Stage.PROTEIN_PRIOR:
        groups += [
            _group(_params(model.protein_encoder), lr_encoder_top, weight_decay),
            _group(_params(model.protein_decoder), lr_encoder_top, weight_decay),
            _group(_params(model.protein_head), lr_heads, weight_decay),
        ]
    elif stage == Stage.RNA_PRIOR:
        groups += [
            _group(_params(model.rna_encoder), lr_encoder_top, weight_decay),
            _group(_params(model.rna_decoder), lr_encoder_top, weight_decay),
            _group(_params(model.rna_head), lr_heads, weight_decay),
        ]
    elif stage == Stage.GLOBAL_C:
        groups.append(_group([model.dmicf.global_c.raw], lr_heads, 0.0))
    elif stage == Stage.DELTA_C:
        groups += [
            _group(_params(model.dmicf.interaction), lr_heads, weight_decay),
            _group(_params(model.dmicf.delta), lr_heads, weight_decay),
        ]
    elif stage == Stage.ALPHA:
        groups.append(_group(_params(model.dmicf.relevance), lr_heads, weight_decay))
    elif stage == Stage.JOINT:
        groups += [
            _group(_params(model.dmicf.interaction, False), lr_heads, weight_decay),
            _group(_params(model.dmicf.delta, False), lr_heads, weight_decay),
            _group(_params(model.dmicf.relevance, False), lr_heads, weight_decay),
            _group(_params(model.protein_decoder, False), lr_projections, weight_decay),
            _group(_params(model.rna_decoder, False), lr_projections, weight_decay),
            _group(_params(model.protein_head, False), lr_projections, weight_decay),
            _group(_params(model.rna_head, False), lr_projections, weight_decay),
            *_encoder_layer_groups(
                model.protein_encoder,
                lr_encoder_top,
                lr_encoder_bottom,
                layerwise_lr_decay,
                weight_decay,
            ),
            *_encoder_layer_groups(
                model.rna_encoder,
                lr_encoder_top,
                lr_encoder_bottom,
                layerwise_lr_decay,
                weight_decay,
            ),
            _group([model.dmicf.raw_lambda_p, model.dmicf.raw_lambda_r], lr_projections, 0.0),
        ]
    else:
        raise ValueError(stage)

    groups = [group for group in groups if group["params"]]
    if not groups:
        raise RuntimeError(f"No parameters for stage {stage}")

    # Guard against accidental duplicate ownership across optimizer groups.
    ids: list[int] = [id(p) for group in groups for p in group["params"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate parameter ownership detected for stage {stage.value}")
    return torch.optim.AdamW(groups)


class TaskRatioSchedule:
    def __init__(
        self,
        start=(2, 2, 1),
        end=(1, 1, 1),
        transition_fraction: float = 0.7,
    ):
        self.start, self.end = start, end
        self.transition_fraction = transition_fraction
        self.names = ("protein_conditional", "rna_conditional", "joint")

    def weights(self, progress: float) -> dict[str, float]:
        x = min(max(progress / self.transition_fraction, 0.0), 1.0)
        vals = [(1 - x) * a + x * b for a, b in zip(self.start, self.end)]
        total = sum(vals)
        return {name: value / total for name, value in zip(self.names, vals)}


def assert_stage_batch(contract: StageContract, batch_meta: dict) -> None:
    task = batch_meta.get("task")
    if task not in contract.allowed_tasks:
        raise AssertionError(f"Task {task!r} is illegal for {contract.stage.value}")
    for forbidden in contract.forbidden_inputs:
        if batch_meta.get(forbidden, False):
            raise AssertionError(f"Forbidden input {forbidden!r} present in {contract.stage.value}")
