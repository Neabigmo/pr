"""Executable DM-ICF core for the mini-pilot.

Scientific decomposition:
    final preference = intramolecular structural prior + local cross-molecular selection
    Gamma_ij = alpha_ij * (C + DeltaC_ij)

Important implementation choices:
- protein and RNA backbone encoders are independent and sequence-neutral;
- a *within-polymer* masked/autoregressive context decoder consumes only already
  known tokens, making the variable-mask and mixed-order training protocol real;
- PR interaction representations use structure and rich geometry, not native token
  identity; partner identities select entries from C + DeltaC only;
- C and DeltaC retain row/column main effects. Only their global scalar mean is
  removed during the forward pass. This is essential for base-independent partner
  effects (for example phosphate-facing Lys/Arg preferences). Double-centering is
  available only as a post-hoc interaction-only visualization gauge;
- lambda_P/lambda_R are fixed at 1 during C/Delta/alpha learning and become bounded
  learnable gains only during final joint coordination, avoiding a C-scale
  identifiability problem.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

PROTEIN_ALPHABET = 20
RNA_ALPHABET = 4


def global_center(matrix: Tensor) -> Tensor:
    """Remove only the unidentifiable all-entry scalar offset."""
    return matrix - matrix.mean(dim=(-2, -1), keepdim=True)


def double_center(matrix: Tensor) -> Tensor:
    """Post-hoc interaction-only gauge; do NOT use this in the model forward path."""
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
            nn.Linear(dim, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class SimpleSparseBackboneEncoder(nn.Module):
    """Sequence-neutral sparse geometric encoder.

    The adapter supplies invariant/local-frame scalar geometry. Raw global xyz is
    never fed directly to this MLP, so rigid rotations/translations cannot change
    predictions merely because of coordinate frame choice.
    """

    def __init__(self, node_in: int, edge_in: int, hidden: int = 256, layers: int = 6):
        super().__init__()
        self.node_proj = nn.Linear(node_in, hidden)
        self.edge_proj = nn.Linear(edge_in, hidden)
        self.message = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden)) for _ in range(layers)]
        )
        self.update = nn.ModuleList([ResidualMLP(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, node_x: Tensor, edge_index: Tensor, edge_x: Tensor) -> Tensor:
        h = self.node_proj(node_x)
        e = self.edge_proj(edge_x)
        src, dst = edge_index
        for msg_net, upd in zip(self.message, self.update):
            if src.numel() == 0:
                h = upd(h)
                continue
            m = msg_net(torch.cat([h[src], h[dst], e], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            h = upd(h + agg / deg.clamp_min(1.0).unsqueeze(-1))
        return self.norm(h)


class SparseSequenceContextDecoder(nn.Module):
    """Within-polymer known-token context on top of sequence-neutral structure.

    A target token that is masked has zero token embedding and therefore cannot
    leak into its own prediction. Known neighbouring tokens can influence it,
    which makes variable masking and random-order decoding meaningful while PR
    information remains cleanly separated in DM-ICF.
    """

    def __init__(self, alphabet: int, edge_in: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.alphabet = alphabet
        self.token_embedding = nn.Embedding(alphabet, hidden)
        self.edge_proj = nn.Linear(edge_in, hidden)
        self.message = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Linear(hidden, hidden)) for _ in range(layers)]
        )
        self.update = nn.ModuleList([ResidualMLP(hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h_struct: Tensor, edge_index: Tensor, edge_x: Tensor, tokens: Tensor, known: Tensor) -> Tensor:
        if tokens.shape[0] != h_struct.shape[0] or known.shape[0] != h_struct.shape[0]:
            raise ValueError("tokens/known must align with polymer nodes")
        if tokens.numel() and ((tokens < 0).any() or (tokens >= self.alphabet).any()):
            raise ValueError("Token indices outside alphabet")
        token_h = self.token_embedding(tokens) * known.to(h_struct.dtype).unsqueeze(-1)
        edge_h = self.edge_proj(edge_x)
        h = h_struct
        src, dst = edge_index
        for msg_net, upd in zip(self.message, self.update):
            if src.numel() == 0:
                h = upd(h)
                continue
            m = msg_net(torch.cat([h[src], h[dst], token_h[src], edge_h], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, m)
            deg = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            h = upd(h + agg / deg.clamp_min(1.0).unsqueeze(-1))
        return self.norm(h)


@dataclass
class PRBatch:
    protein_index: Tensor
    rna_index: Tensor
    edge_features: Tensor
    effective_distance: Tensor
    edge_batch: Tensor | None = None


class CrossInteractionEncoder(nn.Module):
    """q_ij = G(Pi_P hP_i + Pi_R hR_j + f_e(e_ij))."""

    def __init__(self, hidden: int, edge_in: int, edge_hidden: int = 256, layers: int = 3):
        super().__init__()
        self.p_proj = nn.Linear(hidden, edge_hidden)
        self.r_proj = nn.Linear(hidden, edge_hidden)
        self.e_proj = nn.Sequential(nn.Linear(edge_in, edge_hidden), nn.GELU(), nn.Linear(edge_hidden, edge_hidden))
        self.input_norm = nn.LayerNorm(edge_hidden)
        self.blocks = nn.ModuleList([ResidualMLP(edge_hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(edge_hidden)

    def forward(self, hp: Tensor, hr: Tensor, pr: PRBatch) -> Tensor:
        p = self.p_proj(hp[pr.protein_index])
        r = self.r_proj(hr[pr.rna_index])
        e = self.e_proj(pr.edge_features)
        q = self.input_norm(p + r + e)
        for block in self.blocks:
            q = block(q)
        return self.norm(q)


class GlobalCompatibility(nn.Module):
    def __init__(self, init_std: float = 1e-3):
        super().__init__()
        self.raw = nn.Parameter(torch.empty(PROTEIN_ALPHABET, RNA_ALPHABET))
        nn.init.normal_(self.raw, mean=0.0, std=init_std)

    def forward(self) -> Tensor:
        return global_center(self.raw)

    def interaction_only_view(self) -> Tensor:
        return double_center(self.forward())


class ContextualResidual(nn.Module):
    def __init__(self, edge_hidden: int):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(edge_hidden, edge_hidden), nn.GELU())
        self.out = nn.Linear(edge_hidden, PROTEIN_ALPHABET * RNA_ALPHABET)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q: Tensor) -> Tensor:
        dc = self.out(self.pre(q)).view(-1, PROTEIN_ALPHABET, RNA_ALPHABET)
        return global_center(dc)


class RelationalRelevance(nn.Module):
    def __init__(self, edge_hidden: int, initial_tau: float = 2.0):
        super().__init__()
        if initial_tau <= 0:
            raise ValueError("initial_tau must be positive")
        self.raw_tau = nn.Parameter(torch.tensor(math.log(math.expm1(initial_tau)), dtype=torch.float32))
        self.score = nn.Linear(edge_hidden, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    @property
    def tau(self) -> Tensor:
        return F.softplus(self.raw_tau).clamp_min(1e-4)

    def edge_scores(self, q: Tensor, distance: Tensor, learned: bool = True) -> Tensor:
        residual = self.score(q).squeeze(-1) if learned else torch.zeros_like(distance)
        return -distance / self.tau + residual

    @staticmethod
    def neighborhood_softmax(scores: Tensor, group_index: Tensor, n_groups: int) -> Tensor:
        if scores.ndim != 1 or group_index.ndim != 1 or scores.shape[0] != group_index.shape[0]:
            raise ValueError("scores/group_index must be aligned 1D tensors")
        if scores.numel() == 0:
            return scores
        max_per = torch.full((n_groups,), -torch.inf, device=scores.device, dtype=scores.dtype)
        if hasattr(max_per, "scatter_reduce_"):
            max_per.scatter_reduce_(0, group_index, scores, reduce="amax", include_self=True)
        else:
            for g in range(n_groups):
                mask = group_index == g
                if mask.any():
                    max_per[g] = scores[mask].max()
        ex = torch.exp(scores - max_per[group_index])
        denom = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype)
        denom.index_add_(0, group_index, ex)
        return ex / denom[group_index].clamp_min(1e-12)

    def forward(self, q: Tensor, pr: PRBatch, n_protein: int, n_rna: int, learned: bool = True) -> tuple[Tensor, Tensor, Tensor]:
        scores = self.edge_scores(q, pr.effective_distance, learned=learned)
        return (
            self.neighborhood_softmax(scores, pr.protein_index, n_protein),
            self.neighborhood_softmax(scores, pr.rna_index, n_rna),
            scores,
        )


class DMICF(nn.Module):
    def __init__(self, hidden: int, pr_edge_in: int, edge_hidden: int = 256, interaction_layers: int = 3):
        super().__init__()
        self.interaction = CrossInteractionEncoder(hidden, pr_edge_in, edge_hidden, interaction_layers)
        self.global_c = GlobalCompatibility()
        self.delta = ContextualResidual(edge_hidden)
        self.relevance = RelationalRelevance(edge_hidden)
        # Bounded gains in (0,2), initialized exactly at 1. They are frozen until joint.
        self.raw_lambda_p = nn.Parameter(torch.tensor(0.0))
        self.raw_lambda_r = nn.Parameter(torch.tensor(0.0))

    @property
    def lambda_p(self) -> Tensor:
        return 2.0 * torch.sigmoid(self.raw_lambda_p)

    @property
    def lambda_r(self) -> Tensor:
        return 2.0 * torch.sigmoid(self.raw_lambda_r)

    def field(self, hp: Tensor, hr: Tensor, pr: PRBatch, use_delta: bool = True, learned_alpha: bool = True) -> dict[str, Tensor]:
        q = self.interaction(hp, hr, pr)
        c = self.global_c()
        dc = self.delta(q) if use_delta else torch.zeros((pr.protein_index.shape[0], 20, 4), device=q.device, dtype=q.dtype)
        alpha_p, alpha_r, scores = self.relevance(q, pr, hp.shape[0], hr.shape[0], learned=learned_alpha)
        return {"q": q, "C": c, "DeltaC": dc, "alpha_p": alpha_p, "alpha_r": alpha_r, "scores": scores}

    def protein_correction(self, field: dict[str, Tensor], pr: PRBatch, rna_tokens: Tensor, rna_known: Tensor, n_protein: int) -> Tensor:
        cedge = field["C"].unsqueeze(0) + field["DeltaC"]
        b = rna_tokens[pr.rna_index]
        known = rna_known[pr.rna_index].to(cedge.dtype)
        selected = cedge.gather(-1, b[:, None, None].expand(-1, PROTEIN_ALPHABET, 1)).squeeze(-1)
        selected = selected * field["alpha_p"][:, None] * known[:, None]
        out = torch.zeros((n_protein, PROTEIN_ALPHABET), device=selected.device, dtype=selected.dtype)
        out.index_add_(0, pr.protein_index, selected)
        return self.lambda_p * out

    def rna_correction(self, field: dict[str, Tensor], pr: PRBatch, protein_tokens: Tensor, protein_known: Tensor, n_rna: int) -> Tensor:
        cedge = field["C"].unsqueeze(0) + field["DeltaC"]
        a = protein_tokens[pr.protein_index]
        known = protein_known[pr.protein_index].to(cedge.dtype)
        selected = cedge.gather(-2, a[:, None, None].expand(-1, 1, RNA_ALPHABET)).squeeze(-2)
        selected = selected * field["alpha_r"][:, None] * known[:, None]
        out = torch.zeros((n_rna, RNA_ALPHABET), device=selected.device, dtype=selected.dtype)
        out.index_add_(0, pr.rna_index, selected)
        return self.lambda_r * out


class JointPriorAndFieldModel(nn.Module):
    def __init__(self, protein_node_in: int, protein_edge_in: int, rna_node_in: int, rna_edge_in: int, pr_edge_in: int, hidden: int = 256, protein_layers: int = 6, rna_layers: int = 6, decoder_layers: int = 2):
        super().__init__()
        self.protein_encoder = SimpleSparseBackboneEncoder(protein_node_in, protein_edge_in, hidden, protein_layers)
        self.rna_encoder = SimpleSparseBackboneEncoder(rna_node_in, rna_edge_in, hidden, rna_layers)
        self.protein_decoder = SparseSequenceContextDecoder(PROTEIN_ALPHABET, protein_edge_in, hidden, decoder_layers)
        self.rna_decoder = SparseSequenceContextDecoder(RNA_ALPHABET, rna_edge_in, hidden, decoder_layers)
        self.protein_head = nn.Linear(hidden, PROTEIN_ALPHABET)
        self.rna_head = nn.Linear(hidden, RNA_ALPHABET)
        self.dmicf = DMICF(hidden=hidden, pr_edge_in=pr_edge_in, edge_hidden=hidden)

    def encode_backbones(self, protein_node_x: Tensor, protein_edge_index: Tensor, protein_edge_x: Tensor, rna_node_x: Tensor, rna_edge_index: Tensor, rna_edge_x: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.protein_encoder(protein_node_x, protein_edge_index, protein_edge_x),
            self.rna_encoder(rna_node_x, rna_edge_index, rna_edge_x),
        )

    def protein_prior_logits(self, node_x: Tensor, edge_index: Tensor, edge_x: Tensor, tokens: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
        h = self.protein_encoder(node_x, edge_index, edge_x)
        hc = self.protein_decoder(h, edge_index, edge_x, tokens, known)
        return self.protein_head(hc), h

    def rna_prior_logits(self, node_x: Tensor, edge_index: Tensor, edge_x: Tensor, tokens: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
        h = self.rna_encoder(node_x, edge_index, edge_x)
        hc = self.rna_decoder(h, edge_index, edge_x, tokens, known)
        return self.rna_head(hc), h

    def forward(self, protein_node_x: Tensor, protein_edge_index: Tensor, protein_edge_x: Tensor, rna_node_x: Tensor, rna_edge_index: Tensor, rna_edge_x: Tensor, pr: PRBatch, protein_tokens: Tensor, rna_tokens: Tensor, protein_known: Tensor, rna_known: Tensor, use_delta: bool = True, learned_alpha: bool = True) -> dict[str, Tensor]:
        hp, hr = self.encode_backbones(protein_node_x, protein_edge_index, protein_edge_x, rna_node_x, rna_edge_index, rna_edge_x)
        hp_ctx = self.protein_decoder(hp, protein_edge_index, protein_edge_x, protein_tokens, protein_known)
        hr_ctx = self.rna_decoder(hr, rna_edge_index, rna_edge_x, rna_tokens, rna_known)
        zp_struct = self.protein_head(hp_ctx)
        zr_struct = self.rna_head(hr_ctx)
        field = self.dmicf.field(hp, hr, pr, use_delta=use_delta, learned_alpha=learned_alpha)
        zp_delta = self.dmicf.protein_correction(field, pr, rna_tokens, rna_known, hp.shape[0])
        zr_delta = self.dmicf.rna_correction(field, pr, protein_tokens, protein_known, hr.shape[0])
        return {
            "protein_hidden": hp, "rna_hidden": hr,
            "protein_context_hidden": hp_ctx, "rna_context_hidden": hr_ctx,
            "protein_struct_logits": zp_struct, "rna_struct_logits": zr_struct,
            "protein_delta_logits": zp_delta, "rna_delta_logits": zr_delta,
            "protein_logits": zp_struct + zp_delta, "rna_logits": zr_struct + zr_delta,
            **field,
        }


def set_trainable_stage(model: JointPriorAndFieldModel, stage: Literal["protein_prior", "rna_prior", "global_c", "delta_c", "alpha", "joint"]) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if stage == "protein_prior":
        for module in [model.protein_encoder, model.protein_decoder, model.protein_head]:
            for p in module.parameters(): p.requires_grad = True
    elif stage == "rna_prior":
        for module in [model.rna_encoder, model.rna_decoder, model.rna_head]:
            for p in module.parameters(): p.requires_grad = True
    elif stage == "global_c":
        for p in model.dmicf.global_c.parameters(): p.requires_grad = True
        # lambda gains deliberately remain fixed at exactly 1 here.
    elif stage == "delta_c":
        for module in [model.dmicf.interaction, model.dmicf.delta]:
            for p in module.parameters(): p.requires_grad = True
    elif stage == "alpha":
        for module in [model.dmicf.interaction, model.dmicf.delta, model.dmicf.relevance]:
            for p in module.parameters(): p.requires_grad = True
    elif stage == "joint":
        for p in model.parameters(): p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")
