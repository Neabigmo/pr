"""Zero-initialized reciprocal adapters with an explicit partner-use V2 path."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AdapterConfig:
    geometry: str = "G2"
    aggregation: str = "A2"
    hidden_dim: int = 128
    hidden_projection_dim: int = 64
    edge_dim: int = 64
    token_dim: int = 32
    message_dim: int = 128
    layers: int = 1
    dropout: float = 0.1
    rbf_bins: int = 16
    separate_edge_encoders: bool = False
    sequence_independent_attention: bool = False
    partner_centered_residual: bool = False
    modality_projector: bool = False
    conservative_gate: bool = False
    gate_init: float = 0.1


def _mlp(input_dim: int, output_dim: int, layers: int, dropout: float) -> nn.Sequential:
    blocks: list[nn.Module] = []
    for index in range(max(1, layers)):
        blocks.append(nn.Linear(input_dim if index == 0 else output_dim, output_dim))
        blocks.append(nn.GELU())
        blocks.append(nn.Dropout(dropout))
    return nn.Sequential(*blocks)


class _Direction(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        token_dim: int,
        edge_dim: int,
        message_dim: int,
        partner_vocab: int,
        output_vocab: int,
        aggregation: str,
        layers: int,
        dropout: float,
        sequence_independent_attention: bool = False,
        partner_centered_residual: bool = False,
        conservative_gate: bool = False,
        gate_init: float = 0.1,
        correct_null_softmax: bool = False,
    ):
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be strictly between zero and one")
        if aggregation not in {"A0", "A1", "A2"}:
            raise ValueError(f"unknown aggregation {aggregation}")
        if partner_centered_residual and not sequence_independent_attention:
            raise ValueError("partner-centered residual requires sequence-independent attention")

        self.token_embedding = nn.Embedding(partner_vocab, token_dim)
        self.sequence_independent_attention = bool(sequence_independent_attention)
        self.partner_centered_residual = bool(partner_centered_residual)
        self.correct_null_softmax = bool(correct_null_softmax)
        self.aggregation = aggregation

        if self.sequence_independent_attention:
            # u_ij is deliberately token-free. It decides which partner is
            # useful; the value network decides what sequence information it
            # contributes.
            self.target_norm = nn.LayerNorm(hidden_dim)
            self.partner_norm = nn.LayerNorm(hidden_dim)
            content_dim = hidden_dim * 2 + edge_dim
            self.content = _mlp(content_dim, message_dim, layers, dropout)
            self.value = _mlp(message_dim + token_dim, message_dim, layers, dropout)
        else:
            # Compatibility path for the original B0/B1 Adapter. Its
            # attention score still sees token-bearing messages by design.
            input_dim = hidden_dim * 2 + token_dim + edge_dim
            self.message = _mlp(input_dim, message_dim, layers, dropout)

        self.attention = nn.Linear(message_dim, 1) if aggregation in {"A1", "A2"} else None
        self.null_logit = nn.Parameter(torch.zeros(())) if aggregation == "A2" else None
        self.norm = nn.LayerNorm(message_dim)
        self.output = nn.Linear(message_dim, output_vocab)
        # ControlNet-style safety boundary: before training the adapter is
        # mathematically absent regardless of the random branch.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        self.conservative_gate = bool(conservative_gate)
        if self.conservative_gate:
            self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init))))

    def _gate(self, reference: Tensor) -> Tensor:
        if not self.conservative_gate:
            return reference.new_ones(())
        return torch.sigmoid(self.gate_logit).to(dtype=reference.dtype)

    def _aggregate(
        self,
        values: Tensor,
        scores: Tensor | None,
        target_index: Tensor,
        length: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Aggregate values and return edge weights, null weights, and scores."""
        aggregate = values.new_zeros((length, values.shape[-1]))
        edge_weights = values.new_zeros((values.shape[0],))
        null_weights = values.new_zeros((length,))
        if self.aggregation == "A0":
            aggregate.index_add_(0, target_index, values)
            count = values.new_zeros((length, 1))
            count.index_add_(0, target_index, values.new_ones((target_index.shape[0], 1)))
            return aggregate / count.clamp_min(1.0), edge_weights, null_weights

        assert scores is not None
        maximum_edge = values.new_full((length,), -torch.inf)
        maximum_edge.scatter_reduce_(0, target_index, scores, reduce="amax", include_self=True)
        if self.null_logit is None:
            maximum = maximum_edge
            edge_exp = torch.exp(scores - maximum[target_index])
            denominator = values.new_zeros((length,))
            denominator.index_add_(0, target_index, edge_exp)
        elif self.correct_null_softmax:
            # Correct null-neighbor softmax: null and edge logits share the
            # same stabilized denominator and therefore sum to one.
            null_score = self.null_logit.to(dtype=values.dtype).expand(length)
            maximum = torch.maximum(maximum_edge, null_score)
            edge_exp = torch.exp(scores - maximum[target_index])
            null_exp = torch.exp(null_score - maximum)
            denominator = null_exp.clone()
            denominator.index_add_(0, target_index, edge_exp)
            null_weights = null_exp / denominator.clamp_min(torch.finfo(values.dtype).tiny)
        else:
            # Preserve the original B0/B1 behavior for an apples-to-apples
            # baseline, while exposing its legacy null weight.
            edge_exp = torch.exp(scores - maximum_edge[target_index])
            denominator = values.new_zeros((length,))
            denominator.index_add_(0, target_index, edge_exp)
            null_exp = torch.exp(self.null_logit.to(dtype=values.dtype))
            denominator = denominator + null_exp
            null_weights = null_exp / denominator.clamp_min(torch.finfo(values.dtype).tiny)
        edge_weights = edge_exp / denominator[target_index].clamp_min(torch.finfo(values.dtype).tiny)
        aggregate.index_add_(0, target_index, values * edge_weights.unsqueeze(-1))
        return aggregate, edge_weights, null_weights

    def forward(
        self,
        target_h: Tensor,
        partner_h: Tensor,
        partner_tokens: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        token_off: bool = False,
    ) -> dict[str, Tensor]:
        length = target_h.shape[0]
        if edge_index.numel() == 0:
            return {
                "delta": target_h.new_zeros((length, self.output.out_features)),
                "edge_weights": target_h.new_zeros((0,)),
                "null_weight": target_h.new_ones((length,)),
                "content_norm": target_h.new_zeros((0,)),
                "gate": self._gate(target_h),
            }
        target_index, partner_index = edge_index[0], edge_index[1]
        token = (
            torch.zeros((partner_index.shape[0], self.token_embedding.embedding_dim), device=target_h.device, dtype=target_h.dtype)
            if token_off
            else self.token_embedding(partner_tokens[partner_index]).to(dtype=target_h.dtype)
        )
        if self.sequence_independent_attention:
            content = self.content(
                torch.cat([self.target_norm(target_h)[target_index], self.partner_norm(partner_h)[partner_index], edge_features], dim=-1)
            )
            scores = self.attention(content).squeeze(-1) if self.attention is not None else None
            native_value = self.value(torch.cat([content, token], dim=-1))
            if self.partner_centered_residual:
                off_value = self.value(torch.cat([content, torch.zeros_like(token)], dim=-1))
                values = native_value - off_value
            else:
                values = native_value
            content_norm = content.norm(dim=-1)
        else:
            message = self.message(torch.cat([target_h[target_index], partner_h[partner_index], edge_features, token], dim=-1))
            scores = self.attention(message).squeeze(-1) if self.attention is not None else None
            values = message
            content_norm = message.norm(dim=-1)
        aggregate, edge_weights, null_weight = self._aggregate(values, scores, target_index, length)
        raw_delta = self.output(self.norm(aggregate))
        gate = self._gate(raw_delta)
        delta = raw_delta * gate
        return {"delta": delta, "edge_weights": edge_weights, "null_weight": null_weight, "content_norm": content_norm, "gate": gate}


class ReciprocalAdapter(nn.Module):
    """R->P and P->R residuals over frozen upstream hidden states."""

    def __init__(self, config: AdapterConfig | None = None):
        super().__init__()
        self.config = config or AdapterConfig()
        dims = {"G0": self.config.rbf_bins + 1, "G1": 6 * self.config.rbf_bins + 6, "G2": 6 * self.config.rbf_bins + 6 + 3 + 3 + 6}
        if self.config.geometry not in dims:
            raise ValueError(f"unknown geometry {self.config.geometry}")
        self.raw_geometry_dim = dims[self.config.geometry]
        self.shared_edge_encoder = nn.Sequential(nn.Linear(self.raw_geometry_dim, self.config.edge_dim), nn.GELU())
        self.edge_encoder_r2p = (
            nn.Sequential(nn.Linear(self.raw_geometry_dim, self.config.edge_dim), nn.GELU())
            if self.config.separate_edge_encoders else None
        )
        self.edge_encoder_p2r = (
            nn.Sequential(nn.Linear(self.raw_geometry_dim, self.config.edge_dim), nn.GELU())
            if self.config.separate_edge_encoders else None
        )
        if self.config.modality_projector:
            self.protein_norm = nn.LayerNorm(self.config.hidden_dim)
            self.rna_norm = nn.LayerNorm(self.config.hidden_dim)
            self.protein_projector = nn.Linear(self.config.hidden_dim, self.config.hidden_projection_dim)
            self.rna_projector = nn.Linear(self.config.hidden_dim, self.config.hidden_projection_dim)
            direction_hidden_dim = self.config.hidden_projection_dim
        else:
            direction_hidden_dim = self.config.hidden_dim
        v2 = bool(self.config.sequence_independent_attention)
        direction_args = dict(
            sequence_independent_attention=v2,
            partner_centered_residual=bool(self.config.partner_centered_residual),
            conservative_gate=bool(self.config.conservative_gate),
            gate_init=float(self.config.gate_init),
            correct_null_softmax=v2,
        )
        self.r2p = _Direction(direction_hidden_dim, self.config.token_dim, self.config.edge_dim, self.config.message_dim, 4, 20, self.config.aggregation, self.config.layers, self.config.dropout, **direction_args)
        self.p2r = _Direction(direction_hidden_dim, self.config.token_dim, self.config.edge_dim, self.config.message_dim, 20, 4, self.config.aggregation, self.config.layers, self.config.dropout, **direction_args)

    def _select_geometry(self, full_geometry: Tensor) -> Tensor:
        bins = self.config.rbf_bins
        if self.config.geometry == "G2":
            return full_geometry
        if self.config.geometry == "G1":
            return full_geometry[:, : 6 * bins + 6]
        return torch.cat([full_geometry[:, :bins], full_geometry[:, 6 * bins : 6 * bins + 1]], dim=-1)

    def _hidden_inputs(self, p_h: Tensor, r_h: Tensor) -> tuple[Tensor, Tensor]:
        if not self.config.modality_projector:
            return p_h, r_h
        return self.protein_projector(self.protein_norm(p_h)), self.rna_projector(self.rna_norm(r_h))

    def _residual(self, direction: str, p_h: Tensor, r_h: Tensor, p_tokens: Tensor, r_tokens: Tensor, edge_index: Tensor, full_geometry: Tensor, token_off: bool) -> dict[str, Tensor]:
        geometry = self._select_geometry(full_geometry)
        if self.config.separate_edge_encoders:
            encoder = self.edge_encoder_r2p if direction == "protein" else self.edge_encoder_p2r
            assert encoder is not None
        else:
            encoder = self.shared_edge_encoder
        edge = encoder(geometry)
        if direction == "protein":
            return self.r2p(p_h, r_h, r_tokens, edge_index, edge, token_off)
        if direction == "rna":
            # Cached cross edges are indexed (protein, RNA); the RNA branch
            # receives (RNA target, protein partner).
            return self.p2r(r_h, p_h, p_tokens, edge_index[[1, 0]], edge, token_off)
        raise ValueError(direction)

    def forward(
        self,
        p_base: Tensor,
        r_base: Tensor,
        p_h: Tensor,
        r_h: Tensor,
        p_tokens: Tensor,
        r_tokens: Tensor,
        edge_index_r2p: Tensor,
        edge_geometry_r2p: Tensor,
        edge_index_p2r: Tensor | None = None,
        edge_geometry_p2r: Tensor | None = None,
        token_off: bool = False,
    ) -> dict[str, Tensor]:
        # Keep the old single-edge call valid for existing correctness tests
        # and old diagnostic scripts.
        if edge_index_p2r is None:
            edge_index_p2r = edge_index_r2p
        if edge_geometry_p2r is None:
            edge_geometry_p2r = edge_geometry_r2p
        p_h_input, r_h_input = self._hidden_inputs(p_h, r_h)
        p_details = self._residual("protein", p_h_input, r_h_input, p_tokens, r_tokens, edge_index_r2p, edge_geometry_r2p, token_off)
        r_details = self._residual("rna", p_h_input, r_h_input, p_tokens, r_tokens, edge_index_p2r, edge_geometry_p2r, token_off)
        p_delta, r_delta = p_details["delta"], r_details["delta"]
        return {
            "protein_logits": p_base + p_delta,
            "rna_logits": r_base + r_delta,
            "protein_delta": p_delta,
            "rna_delta": r_delta,
            "protein_edge_weights": p_details["edge_weights"],
            "rna_edge_weights": r_details["edge_weights"],
            "protein_null_weight": p_details["null_weight"],
            "rna_null_weight": r_details["null_weight"],
            "protein_content_norm": p_details["content_norm"],
            "rna_content_norm": r_details["content_norm"],
            "protein_gate": p_details["gate"],
            "rna_gate": r_details["gate"],
        }
