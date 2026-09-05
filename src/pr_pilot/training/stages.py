"""Stage orchestration and optimizer ownership for DM-ICF.

The primary staged experiment deliberately separates explanatory roles:
C learns the global compatibility anchor, DeltaC learns contextual corrections,
alpha learns neighbour relevance, and primary joint adaptation never moves C.
Scratch controls are explicitly exempt from pretrained gradual-unfreezing and are
fully trainable from step 0, including their randomly initialized C matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import nn

from pr_pilot.model.dmicf import JointPriorAndFieldModel, SimpleSparseBackboneEncoder, set_trainable_stage


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
    Stage.PROTEIN_PRIOR: StageContract(Stage.PROTEIN_PRIOR, "protein_train", ("protein_inverse_folding",), "protein_val_normalized_nll", ("rna_partner_sequence", "pr_edges")),
    Stage.RNA_PRIOR: StageContract(Stage.RNA_PRIOR, "rna_train", ("rna_inverse_folding",), "rna_val_normalized_nll", ("protein_partner_sequence", "pr_edges", "native_base_identity_atoms")),
    Stage.GLOBAL_C: StageContract(Stage.GLOBAL_C, "complex_train", ("protein_conditional_interface", "rna_conditional_interface"), "complex_val_bidirectional_interface_nll", ("predicted_structures", "test_manifest", "learned_delta_c", "learned_alpha")),
    Stage.DELTA_C: StageContract(Stage.DELTA_C, "complex_train", ("protein_conditional_interface", "rna_conditional_interface"), "complex_val_bidirectional_interface_nll", ("test_manifest", "learned_alpha")),
    Stage.ALPHA: StageContract(Stage.ALPHA, "complex_train", ("protein_conditional_interface", "rna_conditional_interface"), "complex_val_bidirectional_interface_nll", ("test_manifest",)),
    Stage.JOINT: StageContract(Stage.JOINT, "complex_train", ("protein_conditional", "rna_conditional", "joint"), "complex_val_conditional_plus_sequential_joint_nll", ("test_manifest",)),
}


def _set_module(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def _freeze_global_anchor(model: JointPriorAndFieldModel) -> None:
    _set_module(model.dmicf.global_c, False)


def make_joint_fully_trainable(model: JointPriorAndFieldModel, include_global_c: bool = True) -> None:
    """Scratch-control semantics: no pretrained-specific freezing handicap."""
    for p in model.parameters():
        p.requires_grad = True
    if not include_global_c:
        _freeze_global_anchor(model)


def apply_joint_unfreezing(model: JointPriorAndFieldModel, progress: float) -> dict[str, int]:
    """Release pretrained encoders from output-proximal layers toward inputs.

    Primary joint adaptation freezes the Stage-C global compatibility anchor.
    """
    progress = float(min(max(progress, 0.0), 1.0))
    for module in [model.dmicf, model.protein_decoder, model.rna_decoder, model.protein_head, model.rna_head]:
        _set_module(module, True)
    _freeze_global_anchor(model)

    released = {}
    for name, enc in [("protein", model.protein_encoder), ("rna", model.rna_encoder)]:
        _set_module(enc, False)
        n = len(enc.message)
        n_release = min(n, max(1, int(math.ceil(progress * n))))
        for idx in range(n - n_release, n):
            _set_module(enc.message[idx], True)
            _set_module(enc.update[idx], True)
        _set_module(enc.norm, True)
        if progress >= 0.80:
            _set_module(enc.node_proj, True)
            _set_module(enc.edge_proj, True)
        released[name] = n_release
    return released


def configure_stage(model: JointPriorAndFieldModel, stage: Stage) -> StageContract:
    set_trainable_stage(model, stage.value)
    if stage == Stage.ALPHA:
        # Interaction/DeltaC have already learned their role. Freeze them and train
        # only neighbour relevance (score residual + tau) in the primary alpha stage.
        for p in model.parameters():
            p.requires_grad = False
        _set_module(model.dmicf.relevance, True)
    elif stage == Stage.JOINT:
        apply_joint_unfreezing(model, 0.0)
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


def _encoder_layer_groups(enc: SimpleSparseBackboneEncoder, top_lr: float, bottom_lr: float, decay: float, wd: float) -> list[dict]:
    n = len(enc.message)
    groups: list[dict] = []
    for idx in range(n):
        depth_from_top = n - 1 - idx
        lr = max(bottom_lr, top_lr * (decay ** depth_from_top))
        groups.append(_group(_params(enc.message[idx], False) + _params(enc.update[idx], False), lr, wd))
    groups.append(_group(_params(enc.norm, False), max(bottom_lr, top_lr * decay), wd))
    groups.append(_group(_params(enc.node_proj, False) + _params(enc.edge_proj, False), bottom_lr, wd))
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
            *_encoder_layer_groups(model.protein_encoder, lr_encoder_top, lr_encoder_bottom, layerwise_lr_decay, weight_decay),
            *_encoder_layer_groups(model.rna_encoder, lr_encoder_top, lr_encoder_bottom, layerwise_lr_decay, weight_decay),
            _group([model.dmicf.raw_lambda_p, model.dmicf.raw_lambda_r], lr_projections, 0.0),
        ]
        # Primary joint keeps C frozen. Scratch controls explicitly mark C
        # trainable; only then is it added to the optimizer.
        if model.dmicf.global_c.raw.requires_grad:
            groups.append(_group([model.dmicf.global_c.raw], lr_heads, 0.0))
    else:
        raise ValueError(stage)
    groups = [g for g in groups if g["params"]]
    if not groups:
        raise RuntimeError(f"No parameters for stage {stage}")
    return torch.optim.AdamW(groups)


class TaskRatioSchedule:
    def __init__(self, start=(2, 2, 1), end=(1, 1, 1), transition_fraction: float = 0.7):
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
