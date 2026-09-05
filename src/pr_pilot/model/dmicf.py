"""DM-ICF model components for the mini-pilot.

This module is deliberately explicit about tensor contracts. It is not a toy
pseudo-code file: all core compatibility, residual, alpha and logit-combination
operations are implemented in PyTorch and unit-testable. The only deliberately
modular pieces are the backbone encoders and raw geometry featurizer, because
those must be fed by real parsed structures.

Notation:
  B      batch size
  NP     number of padded protein residues
  NR     number of padded RNA nucleotides
  E      number of sparse PR edges
  H      hidden dimension
  A=20   protein alphabet
  R=4    RNA alphabet

Key scientific contract:
  final preference = structural prior + cross-molecular correction
  Gamma_ij = alpha_ij * (C + DeltaC_ij)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn, Tensor
import torch.nn.functional as F


PROTEIN_ALPHABET = 20
RNA_ALPHABET = 4


def double_center(matrix: Tensor) -> Tensor:
    """Remove row/column additive gauge from (...,20,4) compatibility matrices."""
    if matrix.shape[-2:] != (PROTEIN_ALPHABET, RNA_ALPHABET):
        raise ValueError(f"Expected (...,20,4), got {tuple(matrix.shape)}")
    row_mean = matrix.mean(dim=-1, keepdim=True)
    col_mean = matrix.mean(dim=-2, keepdim=True)
    grand = matrix.mean(dim=(-2, -1), keepdim=True)
    return matrix - row_mean - col_mean + grand


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        inner = dim * hidden_mult
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class SimpleSparseBackboneEncoder(nn.Module):
    """Minimal trainable prior encoder used by the pilot.

    The geometry preprocessing layer is responsible for constructing node_feature
    and sparse edge messages. This encoder intentionally does not read sequence
    identity. It gives the pilot a complete from-scratch trainable implementation
    while keeping raw atom featurization replaceable.
    """

    def __init__(self, node_in: int, edge_in: int, hidden: int = 256, layers: int = 6):
        super().__init__()
        self.node_proj = nn.Linear(node_in, hidden)
        self.edge_proj = nn.Linear(edge_in, hidden)
        self.message = nn.ModuleList([nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden)) for _ in range(layers)])
        self.update = nn.ModuleList([ResidualMLP(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, node_x: Tensor, edge_index: Tensor, edge_x: Tensor) -> Tensor:
        """Encode one concatenated graph.

        Args:
          node_x: [N,node_in]
          edge_index: [2,E], directed source->target
          edge_x: [E,edge_in]
        """
        h = self.node_proj(node_x)
        e = self.edge_proj(edge_x)
        src, dst = edge_index
        for msg_net, upd in zip(self.message, self.update):
            m = msg_net(torch.cat([h[src], h[dst], e], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
            h = upd(h + agg)
        return self.norm(h)


@dataclass
class PRBatch:
    """Sparse protein-RNA edge representation.

    protein_index and rna_index index local nodes in their respective tensors.
    edge_batch identifies which complex each edge belongs to if batching multiple
    complexes. edge_features must be strictly sequence-neutral geometry.
    """

    protein_index: Tensor  # [E] long
    rna_index: Tensor      # [E] long
    edge_features: Tensor  # [E,Fe]
    effective_distance: Tensor  # [E]
    edge_batch: Tensor | None = None


class CrossInteractionEncoder(nn.Module):
    def __init__(self, hidden: int, edge_in: int, edge_hidden: int = 256, layers: int = 3):
        super().__init__()
        self.p_proj = nn.Linear(hidden, edge_hidden)
        self.r_proj = nn.Linear(hidden, edge_hidden)
        self.e_proj = nn.Sequential(nn.Linear(edge_in, edge_hidden), nn.GELU(), nn.Linear(edge_hidden, edge_hidden))
        self.fuse = nn.Sequential(
            nn.Linear(edge_hidden * 3, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, edge_hidden),
        )
        self.blocks = nn.ModuleList([ResidualMLP(edge_hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(edge_hidden)

    def forward(self, hp: Tensor, hr: Tensor, pr: PRBatch) -> Tensor:
        p = self.p_proj(hp[pr.protein_index])
        r = self.r_proj(hr[pr.rna_index])
        e = self.e_proj(pr.edge_features)
        q = self.fuse(torch.cat([p, r, e], dim=-1))
        for block in self.blocks:
            q = block(q)
        return self.norm(q)


class GlobalCompatibility(nn.Module):
    """Learned global 20x4 matrix C with small random initialization."""

    def __init__(self, init_std: float = 1e-3):
        super().__init__()
        self.raw = nn.Parameter(torch.empty(PROTEIN_ALPHABET, RNA_ALPHABET))
        nn.init.normal_(self.raw, mean=0.0, std=init_std)

    def forward(self) -> Tensor:
        return double_center(self.raw)


class ContextualResidual(nn.Module):
    """Per-edge DeltaC_ij. Final layer is exactly zero-initialized."""

    def __init__(self, edge_hidden: int):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(edge_hidden, edge_hidden), nn.GELU())
        self.out = nn.Linear(edge_hidden, PROTEIN_ALPHABET * RNA_ALPHABET)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q: Tensor) -> Tensor:
        dc = self.out(self.pre(q)).view(-1, PROTEIN_ALPHABET, RNA_ALPHABET)
        return double_center(dc)


class RelationalRelevance(nn.Module):
    """Distance-prior alpha with a learned zero-initialized residual score."""

    def __init__(self, edge_hidden: int, initial_tau: float = 2.0):
        super().__init__()
        if initial_tau <= 0:
            raise ValueError("initial_tau must be positive")
        # inverse softplus initialization
        self.raw_tau = nn.Parameter(torch.tensor(math.log(math.expm1(initial_tau)), dtype=torch.float32))
        self.score = nn.Linear(edge_hidden, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    @property
    def tau(self) -> Tensor:
        return F.softplus(self.raw_tau).clamp_min(1e-4)

    def edge_scores(self, q: Tensor, distance: Tensor) -> Tensor:
        return -distance / self.tau + self.score(q).squeeze(-1)

    @staticmethod
    def neighborhood_softmax(scores: Tensor, group_index: Tensor, n_groups: int) -> Tensor:
        """Stable sparse softmax over arbitrary neighbourhood IDs."""
        if scores.ndim != 1 or group_index.ndim != 1 or scores.shape[0] != group_index.shape[0]:
            raise ValueError("scores/group_index must be aligned 1D tensors")
        # Use PyTorch scatter primitives available without torch_scatter.
        max_per = torch.full((n_groups,), -torch.inf, device=scores.device, dtype=scores.dtype)
        if hasattr(max_per, "scatter_reduce_"):
            max_per.scatter_reduce_(0, group_index, scores, reduce="amax", include_self=True)
        else:
            # Conservative fallback for older torch; pilot target is torch>=2.2.
            for g in range(n_groups):
                mask = group_index == g
                if mask.any():
                    max_per[g] = scores[mask].max()
        ex = torch.exp(scores - max_per[group_index])
        denom = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype)
        denom.index_add_(0, group_index, ex)
        return ex / denom[group_index].clamp_min(1e-12)

    def forward(self, q: Tensor, pr: PRBatch, n_protein: int, n_rna: int) -> tuple[Tensor, Tensor, Tensor]:
        scores = self.edge_scores(q, pr.effective_distance)
        alpha_p = self.neighborhood_softmax(scores, pr.protein_index, n_protein)
        alpha_r = self.neighborhood_softmax(scores, pr.rna_index, n_rna)
        return alpha_p, alpha_r, scores


class DMICF(nn.Module):
    """Complete compatibility field independent of the choice of backbone encoders."""

    def __init__(self, hidden: int, pr_edge_in: int, edge_hidden: int = 256, interaction_layers: int = 3):
        super().__init__()
        self.interaction = CrossInteractionEncoder(hidden, pr_edge_in, edge_hidden, interaction_layers)
        self.global_c = GlobalCompatibility()
        self.delta = ContextualResidual(edge_hidden)
        self.relevance = RelationalRelevance(edge_hidden)
        # softplus gains keep contribution sign convention interpretable and non-negative.
        self.raw_lambda_p = nn.Parameter(torch.tensor(-2.0))
        self.raw_lambda_r = nn.Parameter(torch.tensor(-2.0))

    @property
    def lambda_p(self) -> Tensor:
        return F.softplus(self.raw_lambda_p)

    @property
    def lambda_r(self) -> Tensor:
        return F.softplus(self.raw_lambda_r)

    def field(self, hp: Tensor, hr: Tensor, pr: PRBatch) -> dict[str, Tensor]:
        q = self.interaction(hp, hr, pr)
        c = self.global_c()
        dc = self.delta(q)
        alpha_p, alpha_r, scores = self.relevance(q, pr, hp.shape[0], hr.shape[0])
        return {"q": q, "C": c, "DeltaC": dc, "alpha_p": alpha_p, "alpha_r": alpha_r, "scores": scores}

    def protein_correction(
        self,
        field: dict[str, Tensor],
        pr: PRBatch,
        rna_tokens: Tensor,
        rna_known: Tensor,
        n_protein: int,
    ) -> Tensor:
        """Return [NP,20] correction using known RNA neighbour identities only."""
        cedge = field["C"].unsqueeze(0) + field["DeltaC"]  # [E,20,4]
        b = rna_tokens[pr.rna_index]
        known = rna_known[pr.rna_index].to(cedge.dtype)
        selected = cedge.gather(-1, b[:, None, None].expand(-1, PROTEIN_ALPHABET, 1)).squeeze(-1)
        selected = selected * field["alpha_p"][:, None] * known[:, None]
        out = torch.zeros((n_protein, PROTEIN_ALPHABET), device=selected.device, dtype=selected.dtype)
        out.index_add_(0, pr.protein_index, selected)
        return self.lambda_p * out

    def rna_correction(
        self,
        field: dict[str, Tensor],
        pr: PRBatch,
        protein_tokens: Tensor,
        protein_known: Tensor,
        n_rna: int,
    ) -> Tensor:
        """Return [NR,4] correction using known protein neighbour identities only."""
        cedge = field["C"].unsqueeze(0) + field["DeltaC"]
        a = protein_tokens[pr.protein_index]
        known = protein_known[pr.protein_index].to(cedge.dtype)
        selected = cedge.gather(-2, a[:, None, None].expand(-1, 1, RNA_ALPHABET)).squeeze(-2)
        selected = selected * field["alpha_r"][:, None] * known[:, None]
        out = torch.zeros((n_rna, RNA_ALPHABET), device=selected.device, dtype=selected.dtype)
        out.index_add_(0, pr.rna_index, selected)
        return self.lambda_r * out


class JointPriorAndFieldModel(nn.Module):
    """Reference end-to-end model used by the mini-pilot.

    Raw structure featurizers provide protein/RNA graph tensors. The encoders are
    independent; their hidden dimensions match. PR geometry is injected only in
    DM-ICF, preserving the structural-prior / cross-molecular-selection split.
    """

    def __init__(
        self,
        protein_node_in: int,
        protein_edge_in: int,
        rna_node_in: int,
        rna_edge_in: int,
        pr_edge_in: int,
        hidden: int = 256,
        protein_layers: int = 6,
        rna_layers: int = 6,
    ):
        super().__init__()
        self.protein_encoder = SimpleSparseBackboneEncoder(protein_node_in, protein_edge_in, hidden, protein_layers)
        self.rna_encoder = SimpleSparseBackboneEncoder(rna_node_in, rna_edge_in, hidden, rna_layers)
        self.protein_head = nn.Linear(hidden, PROTEIN_ALPHABET)
        self.rna_head = nn.Linear(hidden, RNA_ALPHABET)
        self.dmicf = DMICF(hidden=hidden, pr_edge_in=pr_edge_in, edge_hidden=hidden)

    def encode_priors(
        self,
        protein_node_x: Tensor,
        protein_edge_index: Tensor,
        protein_edge_x: Tensor,
        rna_node_x: Tensor,
        rna_edge_index: Tensor,
        rna_edge_x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        hp = self.protein_encoder(protein_node_x, protein_edge_index, protein_edge_x)
        hr = self.rna_encoder(rna_node_x, rna_edge_index, rna_edge_x)
        return hp, hr, self.protein_head(hp), self.rna_head(hr)

    def forward(
        self,
        protein_node_x: Tensor,
        protein_edge_index: Tensor,
        protein_edge_x: Tensor,
        rna_node_x: Tensor,
        rna_edge_index: Tensor,
        rna_edge_x: Tensor,
        pr: PRBatch,
        protein_tokens: Tensor,
        rna_tokens: Tensor,
        protein_known: Tensor,
        rna_known: Tensor,
    ) -> dict[str, Tensor]:
        hp, hr, zp_struct, zr_struct = self.encode_priors(
            protein_node_x,
            protein_edge_index,
            protein_edge_x,
            rna_node_x,
            rna_edge_index,
            rna_edge_x,
        )
        field = self.dmicf.field(hp, hr, pr)
        zp_delta = self.dmicf.protein_correction(field, pr, rna_tokens, rna_known, hp.shape[0])
        zr_delta = self.dmicf.rna_correction(field, pr, protein_tokens, protein_known, hr.shape[0])
        return {
            "protein_hidden": hp,
            "rna_hidden": hr,
            "protein_struct_logits": zp_struct,
            "rna_struct_logits": zr_struct,
            "protein_delta_logits": zp_delta,
            "rna_delta_logits": zr_delta,
            "protein_logits": zp_struct + zp_delta,
            "rna_logits": zr_struct + zr_delta,
            **field,
        }


def set_trainable_stage(model: JointPriorAndFieldModel, stage: Literal["protein_prior", "rna_prior", "global_c", "delta_c", "alpha", "joint"]) -> None:
    """Enforce the staged training contract by toggling requires_grad.

    Joint-stage discriminative learning rates are handled by the optimizer builder;
    this function only decides which parameter families are allowed to move.
    """
    for p in model.parameters():
        p.requires_grad = False

    if stage == "protein_prior":
        for module in [model.protein_encoder, model.protein_head]:
            for p in module.parameters():
                p.requires_grad = True
    elif stage == "rna_prior":
        for module in [model.rna_encoder, model.rna_head]:
            for p in module.parameters():
                p.requires_grad = True
    elif stage == "global_c":
        for p in model.dmicf.global_c.parameters():
            p.requires_grad = True
        model.dmicf.raw_lambda_p.requires_grad = True
        model.dmicf.raw_lambda_r.requires_grad = True
    elif stage == "delta_c":
        for module in [model.dmicf.interaction, model.dmicf.delta]:
            for p in module.parameters():
                p.requires_grad = True
    elif stage == "alpha":
        for module in [model.dmicf.interaction, model.dmicf.delta, model.dmicf.relevance]:
            for p in module.parameters():
                p.requires_grad = True
    elif stage == "joint":
        # Fine-grained gradual unfreezing is applied by training scheduler.
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")
