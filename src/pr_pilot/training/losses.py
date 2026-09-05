"""Strict loss construction for Protein/RNA conditional and joint tasks.

The pilot treats loss scaling as part of the scientific method. Protein and RNA
have different alphabet entropies, chain lengths differ, and interface tokens are
rare. This module prevents a silent return to pooled token-sum objectives.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F


LOG20 = math.log(20.0)
LOG4 = math.log(4.0)


@dataclass
class LossBreakdown:
    total: Tensor
    raw_pi: Tensor | None = None
    raw_pn: Tensor | None = None
    raw_ri: Tensor | None = None
    raw_rn: Tensor | None = None
    norm_pi: Tensor | None = None
    norm_pn: Tensor | None = None
    norm_ri: Tensor | None = None
    norm_rn: Tensor | None = None
    c_regularizer: Tensor | None = None
    alpha_entropy_term: Tensor | None = None

    def detached_scalars(self) -> dict[str, float]:
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                out[k] = float(v.detach().cpu())
        return out


def _group_ce(logits: Tensor, targets: Tensor, mask: Tensor, smoothing: float = 0.0) -> Tensor | None:
    mask = mask.bool()
    if mask.shape[0] != targets.shape[0] or logits.shape[0] != targets.shape[0]:
        raise ValueError("Logits/targets/mask length mismatch")
    if not mask.any():
        return None
    return F.cross_entropy(logits[mask], targets[mask], reduction="mean", label_smoothing=smoothing)


def _mean_valid(values: list[Tensor | None]) -> Tensor:
    valid = [v for v in values if v is not None]
    if not valid:
        raise ValueError("No valid loss groups in sample/task")
    return torch.stack(valid).mean()


def balanced_sequence_loss(
    protein_logits: Tensor | None,
    protein_targets: Tensor | None,
    protein_mask: Tensor | None,
    protein_interface: Tensor | None,
    rna_logits: Tensor | None,
    rna_targets: Tensor | None,
    rna_mask: Tensor | None,
    rna_interface: Tensor | None,
    task: str,
    protein_label_smoothing: float = 0.05,
    rna_label_smoothing: float = 0.05,
) -> LossBreakdown:
    """Build alphabet-normalized group-balanced objective.

    task: 'protein', 'rna', or 'joint'. A minibatch should contain one task type.
    """
    valid_tasks = {"protein", "rna", "joint"}
    if task not in valid_tasks:
        raise ValueError(f"task must be one of {sorted(valid_tasks)}")

    raw_pi = raw_pn = raw_ri = raw_rn = None

    if protein_logits is not None:
        if protein_targets is None or protein_mask is None or protein_interface is None:
            raise ValueError("Protein tensors must be supplied together")
        raw_pi = _group_ce(protein_logits, protein_targets, protein_mask.bool() & protein_interface.bool(), protein_label_smoothing)
        raw_pn = _group_ce(protein_logits, protein_targets, protein_mask.bool() & ~protein_interface.bool(), protein_label_smoothing)

    if rna_logits is not None:
        if rna_targets is None or rna_mask is None or rna_interface is None:
            raise ValueError("RNA tensors must be supplied together")
        raw_ri = _group_ce(rna_logits, rna_targets, rna_mask.bool() & rna_interface.bool(), rna_label_smoothing)
        raw_rn = _group_ce(rna_logits, rna_targets, rna_mask.bool() & ~rna_interface.bool(), rna_label_smoothing)

    norm_pi = None if raw_pi is None else raw_pi / LOG20
    norm_pn = None if raw_pn is None else raw_pn / LOG20
    norm_ri = None if raw_ri is None else raw_ri / LOG4
    norm_rn = None if raw_rn is None else raw_rn / LOG4

    p_loss = None
    r_loss = None
    if task in {"protein", "joint"}:
        p_loss = _mean_valid([norm_pi, norm_pn])
    if task in {"rna", "joint"}:
        r_loss = _mean_valid([norm_ri, norm_rn])

    if task == "protein":
        total = p_loss
    elif task == "rna":
        total = r_loss
    else:
        if p_loss is None or r_loss is None:
            raise ValueError("Joint task requires both polymers")
        total = 0.5 * (p_loss + r_loss)

    return LossBreakdown(
        total=total,
        raw_pi=raw_pi,
        raw_pn=raw_pn,
        raw_ri=raw_ri,
        raw_rn=raw_rn,
        norm_pi=norm_pi,
        norm_pn=norm_pn,
        norm_ri=norm_ri,
        norm_rn=norm_rn,
    )


def global_c_regularizer(C: Tensor, ce_objective: Tensor, target_fraction: float = 0.02) -> tuple[Tensor, Tensor]:
    """Scale C L2 so it begins as a small fraction of the CE objective.

    In production the coefficient should be fixed from training-only statistics
    after initialization rather than recomputed adaptively each step. This helper
    returns both the penalty and a suggested detached coefficient for initialization.
    """
    if not (0.0 <= target_fraction <= 0.05):
        raise ValueError("C regularizer target must remain a small [0,5%] fraction")
    norm = C.pow(2).mean()
    coeff = (target_fraction * ce_objective.detach() / norm.detach().clamp_min(1e-12)).detach()
    return coeff * norm, coeff


def alpha_entropy_regularizer(alpha: Tensor, group_index: Tensor, n_groups: int) -> Tensor:
    """Return negative entropy; adding positive weight encourages early spread."""
    if alpha.ndim != 1:
        raise ValueError("alpha must be one-dimensional over sparse PR edges")
    ent_terms = -(alpha.clamp_min(1e-12).log() * alpha)
    ent = torch.zeros(n_groups, device=alpha.device, dtype=alpha.dtype)
    ent.index_add_(0, group_index, ent_terms)
    valid = torch.zeros(n_groups, device=alpha.device, dtype=alpha.dtype)
    valid.index_add_(0, group_index, torch.ones_like(alpha))
    return -ent[valid > 0].mean()


def sample_weighted_mean(losses: Tensor, weights: Tensor) -> Tensor:
    """Apply quality weights only after per-sample normalized losses exist."""
    if losses.ndim != 1 or weights.ndim != 1 or losses.shape != weights.shape:
        raise ValueError("losses/weights must be aligned 1D tensors")
    if (weights <= 0).any():
        raise ValueError("Sample weights must be strictly positive")
    return (losses * weights).sum() / weights.sum()
