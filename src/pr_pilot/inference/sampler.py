"""Mixed Protein/RNA autoregressive sampling and Single-Pass Interface Reconciliation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random

import torch
from torch import Tensor

from pr_pilot.model.dmicf import JointPriorAndFieldModel, PRBatch
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


def _seed(base: int, sample_id: str, candidate_index: int) -> int:
    d = hashlib.sha256(f"{base}|{sample_id}|{candidate_index}".encode()).digest()
    return int.from_bytes(d[:8], "little") % (2**31)


def _entropy(logits: Tensor) -> float:
    p = torch.softmax(logits.float(), dim=-1)
    return float((-(p * p.clamp_min(1e-12).log()).sum()).detach().cpu())


def _sample_token(logits: Tensor, temperature: float, generator: torch.Generator) -> tuple[int, float]:
    if temperature <= 0:
        idx = int(logits.argmax())
        lp = float(torch.log_softmax(logits.float(), -1)[idx])
        return idx, lp
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    idx = int(torch.multinomial(probs, 1, generator=generator))
    return idx, float(torch.log(probs[idx].clamp_min(1e-12)).detach().cpu())


def _forward(model: JointPriorAndFieldModel, s: ComplexTensorSample, pt: Tensor, rt: Tensor, pk: Tensor, rk: Tensor) -> dict[str, Tensor]:
    return model(
        s.protein.node_x,s.protein.edge_index,s.protein.edge_x,
        s.rna.node_x,s.rna.edge_index,s.rna.edge_x,s.pr,
        pt,rt,pk,rk,use_delta=True,learned_alpha=True,
    )


def _design_positions(s: ComplexTensorSample) -> list[tuple[str,int]]:
    pos = [("P",int(i)) for i in torch.where(s.protein.valid & ~s.protein.fixed)[0]]
    pos += [("R",int(i)) for i in torch.where(s.rna.valid & ~s.rna.fixed)[0]]
    return pos


def _reopen_uncertain(model: JointPriorAndFieldModel, s: ComplexTensorSample, pt: Tensor, rt: Tensor, polymer: str, fraction: float) -> list[int]:
    graph = s.protein if polymer == "P" else s.rna
    candidates = [int(i) for i in torch.where(graph.interface & graph.valid & ~graph.fixed)[0]]
    scored = []
    for idx in candidates:
        pk = s.protein.valid.clone().bool(); rk = s.rna.valid.clone().bool()
        if polymer == "P": pk[idx] = False
        else: rk[idx] = False
        out = _forward(model,s,pt,rt,pk,rk)
        logits = out["protein_logits"][idx] if polymer == "P" else out["rna_logits"][idx]
        scored.append((_entropy(logits),idx))
    n = min(len(scored), max(0,int(math.ceil(len(scored)*fraction))))
    return [idx for _,idx in sorted(scored,reverse=True)[:n]]


def _refine_polymer(model: JointPriorAndFieldModel, s: ComplexTensorSample, pt: Tensor, rt: Tensor, polymer: str, reopen: list[int], temperature: float, generator: torch.Generator, rng: random.Random) -> None:
    if not reopen: return
    rng.shuffle(reopen)
    pk = s.protein.valid.clone().bool(); rk = s.rna.valid.clone().bool()
    if polymer == "P": pk[torch.tensor(reopen,device=pk.device)] = False
    else: rk[torch.tensor(reopen,device=rk.device)] = False
    for idx in reopen:
        out = _forward(model,s,pt,rt,pk,rk)
        logits = out["protein_logits"][idx] if polymer == "P" else out["rna_logits"][idx]
        token,_ = _sample_token(logits,temperature,generator)
        if polymer == "P": pt[idx]=token; pk[idx]=True
        else: rt[idx]=token; rk[idx]=True


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
) -> list[Candidate]:
    """Generate independent candidates. Unknown partner edges contribute zero by construction."""
    model.eval(); results=[]
    for cidx in range(candidates):
        sd=_seed(seed,sample.sample_id,cidx)
        rng=random.Random(sd)
        gen=torch.Generator(device=sample.protein.node_x.device); gen.manual_seed(sd)
        pt=sample.protein.sequence.clone(); rt=sample.rna.sequence.clone()
        pk=sample.protein.fixed & sample.protein.valid; rk=sample.rna.fixed & sample.rna.valid
        # Unknown token values are irrelevant because known masks are false; use 0 deterministically.
        pt[~pk]=0; rt[~rk]=0
        order=_design_positions(sample); rng.shuffle(order); lps=[]
        for polymer,idx in order:
            out=_forward(model,sample,pt,rt,pk,rk)
            logits=out["protein_logits"][idx] if polymer=="P" else out["rna_logits"][idx]
            token,lp=_sample_token(logits,temperature,gen); lps.append(lp)
            if polymer=="P": pt[idx]=token; pk[idx]=True
            else: rt[idx]=token; rk[idx]=True
        pre_p,pre_r=pt.clone(),rt.clone()
        direction="none"
        if spir_enabled and spir_cycles>0:
            direction="R_then_P" if rng.random()<reverse_direction_fraction else "P_then_R"
            for _ in range(spir_cycles):
                rp=_reopen_uncertain(model,sample,pt,rt,"P",spir_reopen_fraction)
                rr=_reopen_uncertain(model,sample,pt,rt,"R",spir_reopen_fraction)
                if direction=="P_then_R":
                    _refine_polymer(model,sample,pt,rt,"P",rp,spir_temperature,gen,rng)
                    _refine_polymer(model,sample,pt,rt,"R",rr,spir_temperature,gen,rng)
                else:
                    _refine_polymer(model,sample,pt,rt,"R",rr,spir_temperature,gen,rng)
                    _refine_polymer(model,sample,pt,rt,"P",rp,spir_temperature,gen,rng)
        results.append(Candidate(f"{sample.sample_id}__{cidx:04d}",pt.clone(),rt.clone(),pre_p,pre_r,lps,spir_cycles if spir_enabled else 0,direction))
    return results
