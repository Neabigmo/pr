"""Mixed Protein/RNA autoregressive sampling and Single-Pass Interface Reconciliation."""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import TYPE_CHECKING
import hashlib
import math
import random

import torch
from torch import Tensor

from pr_pilot.model.dmicf import JointPriorAndFieldModel, InferenceGeometryCache

if TYPE_CHECKING:
    from pr_pilot.runtime.dataset_adapter import ComplexTensorSample


@dataclass
class Candidate:
    candidate_id: str
    protein_tokens: Tensor
    rna_tokens: Tensor
    pre_spir_protein: Tensor
    pre_spir_rna: Tensor
    token_logprobs: list[float]
    spir_cycles: int
    spir_direction: str
    order_mode: str = "mixed"
    # Legacy token_logprobs describe the temperature-adjusted INITIAL sampler,
    # not a post-SPIR score. These are the corresponding untempered model values.
    token_model_logprobs: list[float] = field(default_factory=list)
    decoding_order: list[tuple[str, int]] = field(default_factory=list)
    spir_interface_scope: str = "design_graph"


def _seed(base: int, sample_id: str, candidate_index: int, order_mode: str = "mixed") -> int:
    digest = hashlib.sha256(f"{base}|{sample_id}|{candidate_index}|{order_mode}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _entropy(logits: Tensor) -> float:
    p = torch.softmax(logits.float(), dim=-1)
    return float((-(p * p.clamp_min(1e-12).log()).sum()).detach().cpu())


def _sample_token(logits: Tensor, temperature: float, generator: torch.Generator) -> tuple[int, float]:
    if temperature <= 0:
        idx = int(logits.argmax())
        return idx, 0.0  # deterministic sampling probability is one
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    idx = int(torch.multinomial(probs, 1, generator=generator))
    return idx, float(torch.log(probs[idx].clamp_min(1e-12)).detach().cpu())


def _forward(model: JointPriorAndFieldModel, sample: ComplexTensorSample, pt: Tensor, rt: Tensor, pk: Tensor, rk: Tensor, cache: InferenceGeometryCache | None = None) -> dict[str, Tensor]:
    if cache is not None:
        return model.decode_cached(cache, pt, rt, pk, rk)
    return model(
        sample.protein.node_x, sample.protein.edge_index, sample.protein.edge_x,
        sample.rna.node_x, sample.rna.edge_index, sample.rna.edge_x,
        sample.pr, pt, rt, pk, rk, use_delta=True, learned_alpha=True,
    )


@contextmanager
def _evaluation_mode(model: JointPriorAndFieldModel):
    """Restore all per-module modes, including intentionally frozen submodules."""
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in states:
            module.training = training


def _prepare_cache(model: JointPriorAndFieldModel, sample: ComplexTensorSample):
    # Do not bypass custom forward semantics used by external/control wrappers.
    # Those models retain the reference path until they prove cache equivalence.
    if (type(model) is not JointPriorAndFieldModel or "forward" in model.__dict__
            or model._forward_hooks or model._forward_pre_hooks):
        return None
    return model.prepare_inference_cache(
        sample.protein.node_x, sample.protein.edge_index, sample.protein.edge_x,
        sample.rna.node_x, sample.rna.edge_index, sample.rna.edge_x, sample.pr,
    )


def _design_positions(sample: ComplexTensorSample, order_mode: str, rng: random.Random) -> list[tuple[str, int]]:
    protein = [("P", int(i)) for i in torch.where(sample.protein.valid & ~sample.protein.fixed)[0]]
    rna = [("R", int(i)) for i in torch.where(sample.rna.valid & ~sample.rna.fixed)[0]]
    rng.shuffle(protein); rng.shuffle(rna)
    if order_mode == "mixed":
        order = protein + rna; rng.shuffle(order); return order
    if order_mode == "protein_first": return protein + rna
    if order_mode == "rna_first": return rna + protein
    raise ValueError("order_mode must be one of: mixed, protein_first, rna_first")


def _design_interface_mask(sample: ComplexTensorSample, polymer: str, scope: str) -> Tensor:
    """Inference sites must be computable from the advertised backbone input.

    Canonical full-heavy-atom labels remain evaluation/training-loss metadata.
    They may depend on native side-chain/base atoms absent at design time. The
    explicit legacy scope exists only for a separately labelled ablation.
    """
    graph = sample.protein if polymer == "P" else sample.rna
    if scope == "canonical_legacy":
        return graph.interface & graph.valid
    if scope != "design_graph":
        raise ValueError("spir_interface_scope must be design_graph or canonical_legacy")
    mask = torch.zeros_like(graph.valid, dtype=torch.bool)
    indices = sample.pr.protein_index if polymer == "P" else sample.pr.rna_index
    mask[indices] = True
    return mask & graph.valid


def _reopen_uncertain(model: JointPriorAndFieldModel, sample: ComplexTensorSample, pt: Tensor, rt: Tensor, polymer: str, fraction: float, cache: InferenceGeometryCache | None = None, interface_scope: str = "design_graph") -> list[int]:
    graph = sample.protein if polymer == "P" else sample.rna
    candidates = [int(i) for i in torch.where(_design_interface_mask(sample, polymer, interface_scope) & ~graph.fixed)[0]]
    scored = []
    for idx in candidates:
        pk = sample.protein.valid.clone().bool(); rk = sample.rna.valid.clone().bool()
        if polymer == "P": pk[idx] = False
        else: rk[idx] = False
        out = _forward(model, sample, pt, rt, pk, rk, cache)
        logits = out["protein_logits"][idx] if polymer == "P" else out["rna_logits"][idx]
        scored.append((_entropy(logits), idx))
    n = min(len(scored), max(0, int(math.ceil(len(scored) * fraction))))
    return [idx for _, idx in sorted(scored, reverse=True)[:n]]


def _refine_polymer(model: JointPriorAndFieldModel, sample: ComplexTensorSample, pt: Tensor, rt: Tensor, polymer: str, reopen: list[int], temperature: float, generator: torch.Generator, rng: random.Random, cache: InferenceGeometryCache | None = None) -> None:
    if not reopen: return
    reopen = list(reopen); rng.shuffle(reopen)
    pk = sample.protein.valid.clone().bool(); rk = sample.rna.valid.clone().bool()
    indices = torch.tensor(reopen, device=pk.device)
    if polymer == "P": pk[indices] = False
    else: rk[indices] = False
    for idx in reopen:
        out = _forward(model, sample, pt, rt, pk, rk, cache)
        logits = out["protein_logits"][idx] if polymer == "P" else out["rna_logits"][idx]
        token, _ = _sample_token(logits, temperature, generator)
        if polymer == "P": pt[idx] = token; pk[idx] = True
        else: rt[idx] = token; rk[idx] = True


@torch.no_grad()
def sample_joint(
    model: JointPriorAndFieldModel,
    sample: ComplexTensorSample,
    candidates: int = 64,
    temperature: float = 1.0,
    seed: int = 20260905,
    spir_enabled: bool = True,
    spir_reopen_fraction: float = 0.30,
    spir_temperature: float = 0.5,
    spir_cycles: int = 1,
    reverse_direction_fraction: float = 0.5,
    order_mode: str = "mixed",
    use_cache: bool = True,
    spir_interface_scope: str = "design_graph",
) -> list[Candidate]:
    """Generate candidates under a controlled initial decoding-order regime.

    ``mixed`` is primary; Protein-first/RNA-first are order controls. In SPIR, the
    second polymer's uncertainty set is recomputed *after* the first polymer has
    been refined, so a nominal P->R (or R->P) pass truly conditions the second
    decision on the updated first-side sequence.
    """
    if spir_interface_scope not in {"design_graph", "canonical_legacy"}:
        raise ValueError("spir_interface_scope must be design_graph or canonical_legacy")
    if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates <= 0:
        raise ValueError("candidates must be positive")
    if not 0.0 <= spir_reopen_fraction <= 1.0:
        raise ValueError("spir_reopen_fraction must be in [0,1]")
    if not 0.0 <= reverse_direction_fraction <= 1.0:
        raise ValueError("reverse_direction_fraction must be in [0,1]")
    if not isinstance(spir_cycles, int) or isinstance(spir_cycles, bool) or spir_cycles < 0:
        raise ValueError("spir_cycles must be a nonnegative integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(spir_temperature) or spir_temperature < 0:
        raise ValueError("spir_temperature must be finite and nonnegative")
    results = []
    with _evaluation_mode(model):
        cache = _prepare_cache(model, sample) if use_cache else None
        for candidate_index in range(candidates):
            sd = _seed(seed, sample.sample_id, candidate_index, order_mode)
            rng = random.Random(sd)
            generator = torch.Generator(device=sample.protein.node_x.device); generator.manual_seed(sd)
            pt = sample.protein.sequence.clone(); rt = sample.rna.sequence.clone()
            pk = sample.protein.fixed & sample.protein.valid; rk = sample.rna.fixed & sample.rna.valid
            pt[~pk] = 0; rt[~rk] = 0
            order = _design_positions(sample, order_mode, rng); logprobs = []; model_logprobs = []
            for polymer, idx in order:
                out = _forward(model, sample, pt, rt, pk, rk, cache)
                logits = out["protein_logits"][idx] if polymer == "P" else out["rna_logits"][idx]
                token, logprob = _sample_token(logits, temperature, generator); logprobs.append(logprob)
                model_logprobs.append(float(torch.log_softmax(logits.float(), -1)[token]))
                if polymer == "P": pt[idx] = token; pk[idx] = True
                else: rt[idx] = token; rk[idx] = True

            pre_p, pre_r = pt.clone(), rt.clone(); direction = "none"
            if spir_enabled and spir_cycles > 0:
                direction = "R_then_P" if rng.random() < reverse_direction_fraction else "P_then_R"
                for _ in range(spir_cycles):
                    if direction == "P_then_R":
                        reopen_p = _reopen_uncertain(model, sample, pt, rt, "P", spir_reopen_fraction, cache, spir_interface_scope)
                        _refine_polymer(model, sample, pt, rt, "P", reopen_p, spir_temperature, generator, rng, cache)
                        reopen_r = _reopen_uncertain(model, sample, pt, rt, "R", spir_reopen_fraction, cache, spir_interface_scope)
                        _refine_polymer(model, sample, pt, rt, "R", reopen_r, spir_temperature, generator, rng, cache)
                    else:
                        reopen_r = _reopen_uncertain(model, sample, pt, rt, "R", spir_reopen_fraction, cache, spir_interface_scope)
                        _refine_polymer(model, sample, pt, rt, "R", reopen_r, spir_temperature, generator, rng, cache)
                        reopen_p = _reopen_uncertain(model, sample, pt, rt, "P", spir_reopen_fraction, cache, spir_interface_scope)
                        _refine_polymer(model, sample, pt, rt, "P", reopen_p, spir_temperature, generator, rng, cache)
            results.append(Candidate(f"{sample.sample_id}__{order_mode}__{candidate_index:04d}", pt.clone(), rt.clone(), pre_p, pre_r, logprobs, spir_cycles if spir_enabled else 0, direction, order_mode, model_logprobs, order, spir_interface_scope))
    return results
