"""Stage orchestration for the DM-ICF pilot.

This file does not hide stage transitions behind a generic trainer. The point of
this pilot is to make parameter ownership auditable: every stage declares exactly
which modules can move, which data task is sampled, and what checkpoint criterion
is legal.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch
from torch import nn

from pr_pilot.model.dmicf import JointPriorAndFieldModel, set_trainable_stage


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
        stage=Stage.PROTEIN_PRIOR,
        dataset="protein_train",
        allowed_tasks=("protein_inverse_folding",),
        checkpoint_metric="protein_val_normalized_nll",
        forbidden_inputs=("rna_partner_sequence", "pr_edges"),
    ),
    Stage.RNA_PRIOR: StageContract(
        stage=Stage.RNA_PRIOR,
        dataset="rna_train",
        allowed_tasks=("rna_inverse_folding",),
        checkpoint_metric="rna_val_normalized_nll",
        forbidden_inputs=("protein_partner_sequence", "pr_edges", "native_base_identity_atoms"),
    ),
    Stage.GLOBAL_C: StageContract(
        stage=Stage.GLOBAL_C,
        dataset="complex_train",
        allowed_tasks=("protein_conditional_interface", "rna_conditional_interface"),
        checkpoint_metric="complex_val_bidirectional_interface_nll",
        forbidden_inputs=("predicted_structures", "test_manifest", "learned_delta_c", "learned_alpha"),
    ),
    Stage.DELTA_C: StageContract(
        stage=Stage.DELTA_C,
        dataset="complex_train",
        allowed_tasks=("protein_conditional_interface", "rna_conditional_interface"),
        checkpoint_metric="complex_val_bidirectional_interface_nll",
        forbidden_inputs=("test_manifest", "learned_alpha"),
    ),
    Stage.ALPHA: StageContract(
        stage=Stage.ALPHA,
        dataset="complex_train",
        allowed_tasks=("protein_conditional_interface", "rna_conditional_interface"),
        checkpoint_metric="complex_val_bidirectional_interface_nll",
        forbidden_inputs=("test_manifest",),
    ),
    Stage.JOINT: StageContract(
        stage=Stage.JOINT,
        dataset="complex_train",
        allowed_tasks=("protein_conditional", "rna_conditional", "joint"),
        checkpoint_metric="complex_val_composite_normalized_nll",
        forbidden_inputs=("test_manifest",),
    ),
}


def configure_stage(model: JointPriorAndFieldModel, stage: Stage) -> StageContract:
    set_trainable_stage(model, stage.value)
    return CONTRACTS[stage]


def trainable_parameter_report(model: nn.Module) -> dict[str, dict[str, int]]:
    """Return trainable/frozen counts by top-level parameter family."""
    report: dict[str, dict[str, int]] = {}
    for name, param in model.named_parameters():
        family = name.split(".")[0]
        if family == "dmicf":
            parts = name.split(".")
            family = ".".join(parts[:2]) if len(parts) > 1 else family
        bucket = report.setdefault(family, {"trainable": 0, "frozen": 0})
        bucket["trainable" if param.requires_grad else "frozen"] += param.numel()
    return report


def _collect_params(module: nn.Module, lr: float, weight_decay: float) -> dict:
    params = [p for p in module.parameters() if p.requires_grad]
    return {"params": params, "lr": lr, "weight_decay": weight_decay}


def build_optimizer(
    model: JointPriorAndFieldModel,
    stage: Stage,
    lr_heads: float = 1e-3,
    lr_projections: float = 5e-4,
    lr_encoder_top: float = 1e-4,
    lr_encoder_bottom: float = 2e-5,
    lr_global_c_joint: float = 1e-5,
    weight_decay: float = 1e-2,
) -> torch.optim.Optimizer:
    """Build an explicit AdamW optimizer with stage-appropriate groups.

    During JOINT we use conservative learning rates for pretrained priors and C.
    The simple pilot encoder does not expose named layer-depth groups yet; when a
    production encoder replaces it, this function is where layer-wise decay must
    be expanded rather than silently using one LR everywhere.
    """
    groups = []
    if stage == Stage.PROTEIN_PRIOR:
        groups += [_collect_params(model.protein_encoder, lr_encoder_top, weight_decay), _collect_params(model.protein_head, lr_heads, weight_decay)]
    elif stage == Stage.RNA_PRIOR:
        groups += [_collect_params(model.rna_encoder, lr_encoder_top, weight_decay), _collect_params(model.rna_head, lr_heads, weight_decay)]
    elif stage == Stage.GLOBAL_C:
        groups.append({"params": [model.dmicf.global_c.raw], "lr": lr_heads, "weight_decay": 0.0})
        groups.append({"params": [model.dmicf.raw_lambda_p, model.dmicf.raw_lambda_r], "lr": lr_projections, "weight_decay": 0.0})
    elif stage == Stage.DELTA_C:
        groups += [_collect_params(model.dmicf.interaction, lr_heads, weight_decay), _collect_params(model.dmicf.delta, lr_heads, weight_decay)]
    elif stage == Stage.ALPHA:
        groups += [
            _collect_params(model.dmicf.interaction, lr_heads, weight_decay),
            _collect_params(model.dmicf.delta, lr_heads, weight_decay),
            _collect_params(model.dmicf.relevance, lr_heads, weight_decay),
        ]
    elif stage == Stage.JOINT:
        groups += [
            _collect_params(model.dmicf.interaction, lr_heads, weight_decay),
            _collect_params(model.dmicf.delta, lr_heads, weight_decay),
            _collect_params(model.dmicf.relevance, lr_heads, weight_decay),
            _collect_params(model.protein_head, lr_projections, weight_decay),
            _collect_params(model.rna_head, lr_projections, weight_decay),
            _collect_params(model.protein_encoder, lr_encoder_bottom, weight_decay),
            _collect_params(model.rna_encoder, lr_encoder_bottom, weight_decay),
            {"params": [model.dmicf.global_c.raw], "lr": lr_global_c_joint, "weight_decay": 0.0},
            {"params": [model.dmicf.raw_lambda_p, model.dmicf.raw_lambda_r], "lr": lr_projections, "weight_decay": 0.0},
        ]
    else:
        raise ValueError(stage)

    groups = [g for g in groups if len(g["params"]) > 0]
    if not groups:
        raise RuntimeError(f"No trainable parameters for stage {stage}")
    return torch.optim.AdamW(groups)


class TaskRatioSchedule:
    """Deterministic ratio schedule for joint-stage task sampling."""

    def __init__(self, start=(2, 2, 1), end=(1, 1, 1), transition_fraction: float = 0.7):
        self.start = start
        self.end = end
        self.transition_fraction = transition_fraction
        self.names = ("protein_conditional", "rna_conditional", "joint")

    def weights(self, progress: float) -> dict[str, float]:
        x = min(max(progress / self.transition_fraction, 0.0), 1.0)
        vals = [(1 - x) * a + x * b for a, b in zip(self.start, self.end)]
        total = sum(vals)
        return {name: value / total for name, value in zip(self.names, vals)}


def assert_stage_batch(contract: StageContract, batch_meta: dict) -> None:
    """Fail if a batch violates the scientific stage contract."""
    task = batch_meta.get("task")
    if task not in contract.allowed_tasks:
        raise AssertionError(f"Task {task!r} is illegal for {contract.stage.value}")
    for forbidden in contract.forbidden_inputs:
        if batch_meta.get(forbidden, False):
            raise AssertionError(f"Forbidden input {forbidden!r} present in {contract.stage.value}")
