"""Audited joint sampler wrapper.

Generation logic lives in ``sampler_legacy.py``. SPIR position selection is patched
to use the sequence-neutral PR message graph (available from the supplied fixed
backbones) rather than the canonical full-heavy-atom reporting interface. Thus
native side-chain/base contact labels cannot leak into inference-time refinement.
"""
from __future__ import annotations

import math
import torch

from pr_pilot.inference import sampler_legacy as _legacy


Candidate = _legacy.Candidate


def _design_interface(sample, polymer: str) -> torch.Tensor:
    graph = sample.protein if polymer == "P" else sample.rna
    mask = torch.zeros(graph.node_x.shape[0], dtype=torch.bool, device=graph.node_x.device)
    indices = sample.pr.protein_index if polymer == "P" else sample.pr.rna_index
    if indices.numel():
        mask[torch.unique(indices)] = True
    return mask


def _reopen_uncertain(model, sample, pt, rt, polymer: str, fraction: float) -> list[int]:
    graph = sample.protein if polymer == "P" else sample.rna
    interface = _design_interface(sample, polymer)
    candidates = [int(i) for i in torch.where(interface & graph.valid & ~graph.fixed)[0]]
    scored = []
    for idx in candidates:
        pk = sample.protein.valid.clone().bool()
        rk = sample.rna.valid.clone().bool()
        if polymer == "P":
            pk[idx] = False
        else:
            rk[idx] = False
        out = _legacy._forward(model, sample, pt, rt, pk, rk)
        logits = out["protein_logits"][idx] if polymer == "P" else out["rna_logits"][idx]
        scored.append((_legacy._entropy(logits), idx))
    n = min(len(scored), max(0, int(math.ceil(len(scored) * fraction))))
    return [idx for _, idx in sorted(scored, reverse=True)[:n]]


_legacy._reopen_uncertain = _reopen_uncertain


def sample_joint(*args, **kwargs):
    return _legacy.sample_joint(*args, **kwargs)


__all__ = ["Candidate", "sample_joint"]
