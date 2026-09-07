"""Leakage-safe fixed-order and leave-one-out scoring of complete sequence pairs.

No own-token all-known score is called a likelihood. Static caching uses exactly
sampler.py's call-scoped inference cache; sequence context is always recomputed.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING
import torch
from torch import Tensor

from pr_pilot.model.dmicf import JointPriorAndFieldModel
from pr_pilot.inference.sampler import _evaluation_mode, _prepare_cache, _forward

if TYPE_CHECKING:
    from pr_pilot.runtime.dataset_adapter import ComplexTensorSample


def _validate_tokens(sample, pt: Tensor, rt: Tensor) -> None:
    for graph, tokens, alphabet in [(sample.protein, pt, 20), (sample.rna, rt, 4)]:
        if tokens.shape != graph.sequence.shape or tokens.ndim != 1 or tokens.dtype != torch.long:
            raise ValueError("Sequences must be aligned int64 vectors")
        if tokens.device != graph.node_x.device:
            raise ValueError("Token and graph devices differ")
        if ((tokens < 0) | (tokens >= alphabet)).any():
            raise ValueError("Token outside canonical alphabet")
        fixed = graph.fixed & graph.valid
        if not torch.equal(tokens[fixed], graph.sequence[fixed]):
            raise ValueError("Candidate changes a user-fixed position")


def _summarize(per_token: list[dict], kind: str) -> dict:
    result = {"score_kind": kind, "tokens": per_token}
    normalized = []
    for polymer, alphabet in [("P", 20), ("R", 4)]:
        rows = [row for row in per_token if row["polymer"] == polymer]
        result[f"{polymer}_n"] = len(rows)
        raw = -sum(row["log_probability"] for row in rows) / len(rows) if rows else None
        result[f"{polymer}_mean_nll"] = raw
        result[f"{polymer}_normalized_nll"] = raw / math.log(alphabet) if raw is not None else None
        if raw is not None:
            normalized.append(raw / math.log(alphabet))
        for group, value in [("interface", True), ("noninterface", False)]:
            subset = [row for row in rows if row["interface"] == value]
            result[f"{polymer}_{group}_n"] = len(subset)
            result[f"{polymer}_{group}_mean_nll"] = (
                -sum(row["log_probability"] for row in subset) / len(subset) if subset else None
            )
    result["balanced_normalized_score"] = sum(normalized) / len(normalized) if normalized else None
    return result


@torch.no_grad()
def teacher_forced_order_score(
    model: JointPriorAndFieldModel,
    sample: ComplexTensorSample,
    protein_tokens: Tensor,
    rna_tokens: Tensor,
    order: list[tuple[str, int]],
    *,
    use_cache: bool = True,
) -> dict:
    """Exact sequence likelihood of the chosen autoregressive factorization.

    Each designable position must appear once. Fixed tokens are context, not
    scored. This is untempered model likelihood, not sampling-temperature NLL.
    """
    _validate_tokens(sample, protein_tokens, rna_tokens)
    expected = {(letter, int(i)) for letter, g in [("P", sample.protein), ("R", sample.rna)]
                for i in torch.where(g.valid & ~g.fixed)[0]}
    if len(order) != len(expected) or len(set(order)) != len(order) or set(order) != expected:
        raise ValueError("Order must cover every designable position exactly once")
    pt, rt = protein_tokens.clone(), rna_tokens.clone()
    pk, rk = sample.protein.fixed & sample.protein.valid, sample.rna.fixed & sample.rna.valid
    pt[~pk] = 0
    rt[~rk] = 0
    records = []
    with _evaluation_mode(model):
        cache = _prepare_cache(model, sample) if use_cache else None
        for polymer, idx in order:
            out = _forward(model, sample, pt, rt, pk, rk, cache)
            graph, key, target = ((sample.protein, "protein_logits", protein_tokens[idx])
                                  if polymer == "P" else (sample.rna, "rna_logits", rna_tokens[idx]))
            logp = float(torch.log_softmax(out[key][idx].float(), -1)[target])
            records.append({"polymer": polymer, "index": idx,
                            "interface": bool(graph.interface[idx]), "log_probability": logp})
            if polymer == "P":
                pt[idx], pk[idx] = target, True
            else:
                rt[idx], rk[idx] = target, True
    result = _summarize(records, "fixed_order_autoregressive")
    result["sequence_log_probability"] = sum(r["log_probability"] for r in records)
    result["sequence_nll"] = -result["sequence_log_probability"]
    result["order"] = order
    return result


@torch.no_grad()
def leave_one_out_pair_score(
    model: JointPriorAndFieldModel,
    sample: ComplexTensorSample,
    protein_tokens: Tensor,
    rna_tokens: Tensor,
    *,
    use_cache: bool = True,
) -> dict:
    """Candidate compatibility score, NOT a normalized joint likelihood.

    The scored site's own token is hidden, even after both sequences are known.
    All other valid sites, including the partner, are supplied as context. This
    prevents own-token return paths through the multi-layer same-chain decoder.
    """
    _validate_tokens(sample, protein_tokens, rna_tokens)
    records = []
    with _evaluation_mode(model):
        cache = _prepare_cache(model, sample) if use_cache else None
        for polymer, graph, key, tokens in [
            ("P", sample.protein, "protein_logits", protein_tokens),
            ("R", sample.rna, "rna_logits", rna_tokens),
        ]:
            for idx in torch.where(graph.valid & ~graph.fixed)[0].tolist():
                pk, rk = sample.protein.valid.clone(), sample.rna.valid.clone()
                if polymer == "P":
                    pk[idx] = False
                else:
                    rk[idx] = False
                out = _forward(model, sample, protein_tokens, rna_tokens, pk, rk, cache)
                logp = float(torch.log_softmax(out[key][idx].float(), -1)[tokens[idx]])
                records.append({"polymer": polymer, "index": idx,
                                "interface": bool(graph.interface[idx]), "log_probability": logp})
    return _summarize(records, "leave_one_out_compatibility_not_joint_likelihood")
