"""Zero-initialized reciprocal adapters with an explicit partner-use V2 path."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .geometry import geometry_dimension


@dataclass(frozen=True)
class AdapterConfig:
    geometry: str = "G2"
    aggregation: str = "A2"
    interaction: str = "concat"
    residual: str = "direct"
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
    rna_gate_mode: str = "baseline"


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
        gate_scale: float = 1.0,
        fixed_gate: float | None = None,
        correct_null_softmax: bool = False,
        interaction: str = "concat",
        residual: str = "direct",
    ):
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("gate_init must be strictly between zero and one")
        if gate_scale < 0.0:
            raise ValueError("gate_scale must be non-negative")
        if fixed_gate is not None and not 0.0 <= fixed_gate <= 1.0:
            raise ValueError("fixed_gate must be between zero and one")
        if aggregation not in {"A0", "A1", "A2"}:
            raise ValueError(f"unknown aggregation {aggregation}")
        if interaction not in {"concat", "centered", "multiplicative", "film"}:
            raise ValueError(f"unknown interaction {interaction}")
        if residual not in {"direct", "partner_centered", "scalar_gate", "confidence_gate"}:
            raise ValueError(f"unknown residual {residual}")
        if (partner_centered_residual or interaction == "centered" or residual == "partner_centered") and not sequence_independent_attention:
            raise ValueError("partner-centered residual requires sequence-independent attention")

        self.token_embedding = nn.Embedding(partner_vocab, token_dim)
        self.sequence_independent_attention = bool(sequence_independent_attention)
        self.partner_centered_residual = bool(partner_centered_residual)
        self.interaction = str(interaction)
        self.residual = str(residual)
        self.correct_null_softmax = bool(correct_null_softmax)
        self.aggregation = aggregation
        self.gate_scale = float(gate_scale)
        self.fixed_gate = None if fixed_gate is None else float(fixed_gate)

        if self.sequence_independent_attention:
            # u_ij is deliberately token-free. It decides which partner is
            # useful; the value network decides what sequence information it
            # contributes.
            self.target_norm = nn.LayerNorm(hidden_dim)
            self.partner_norm = nn.LayerNorm(hidden_dim)
            content_dim = hidden_dim * 2 + edge_dim
            self.content = _mlp(content_dim, message_dim, layers, dropout)
            if interaction == "film":
                self.film = nn.Linear(token_dim, message_dim * 2)
                self.value = _mlp(message_dim, message_dim, layers, dropout)
            elif interaction == "multiplicative":
                self.token_projection = nn.Linear(token_dim, message_dim)
                self.value = _mlp(message_dim, message_dim, layers, dropout)
            else:
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

        self.conservative_gate = bool(conservative_gate or residual == "scalar_gate")
        self.position_gate = residual == "confidence_gate"
        if self.conservative_gate and self.fixed_gate is None:
            self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init))))
        if self.position_gate:
            self.confidence_gate = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.confidence_gate.weight)
            nn.init.constant_(self.confidence_gate.bias, math.log(gate_init / (1.0 - gate_init)))

    def _gate(self, reference: Tensor, target_h: Tensor | None = None) -> Tensor:
        if self.fixed_gate is not None:
            return reference.new_tensor(self.fixed_gate)
        if self.position_gate:
            if target_h is None:
                raise ValueError("confidence gate requires target hidden states")
            return (self.gate_scale * torch.sigmoid(self.confidence_gate(target_h).squeeze(-1))).to(dtype=reference.dtype)
        if not self.conservative_gate:
            return reference.new_ones(())
        return (self.gate_scale * torch.sigmoid(self.gate_logit)).to(dtype=reference.dtype)

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
        partner_known: Tensor | None = None,
    ) -> dict[str, Tensor]:
        length = target_h.shape[0]
        if edge_index.numel() == 0:
            return {
                "delta": target_h.new_zeros((length, self.output.out_features)),
                "edge_weights": target_h.new_zeros((0,)),
                "null_weight": target_h.new_ones((length,)),
                "content_norm": target_h.new_zeros((0,)),
                "gate": self._gate(target_h, target_h),
            }
        target_index, partner_index = edge_index[0], edge_index[1]
        token = (
            torch.zeros((partner_index.shape[0], self.token_embedding.embedding_dim), device=target_h.device, dtype=target_h.dtype)
            if token_off
            else self.token_embedding(partner_tokens[partner_index]).to(dtype=target_h.dtype)
        )
        if partner_known is not None and not token_off:
            if partner_known.ndim != 1 or partner_known.shape[0] != partner_h.shape[0]:
                raise ValueError("partner_known must align with partner hidden states")
            token = token * partner_known[partner_index].to(dtype=token.dtype).unsqueeze(-1)
        if self.sequence_independent_attention:
            content = self.content(
                torch.cat([self.target_norm(target_h)[target_index], self.partner_norm(partner_h)[partner_index], edge_features], dim=-1)
            )
            scores = self.attention(content).squeeze(-1) if self.attention is not None else None
            centered = self.partner_centered_residual or self.interaction == "centered" or self.residual == "partner_centered"
            if self.interaction == "film":
                gamma, beta = self.film(token).chunk(2, dim=-1)
                native_value = self.value(content * (1.0 + gamma) + beta)
                off_value = self.value(content)
            elif self.interaction == "multiplicative":
                native_value = self.value(content * self.token_projection(token))
                off_value = self.value(torch.zeros_like(content))
            else:
                native_value = self.value(torch.cat([content, token], dim=-1))
                off_value = self.value(torch.cat([content, torch.zeros_like(token)], dim=-1))
            values = native_value - off_value if centered else native_value
            content_norm = content.norm(dim=-1)
        else:
            message = self.message(torch.cat([target_h[target_index], partner_h[partner_index], edge_features, token], dim=-1))
            scores = self.attention(message).squeeze(-1) if self.attention is not None else None
            values = message
            content_norm = message.norm(dim=-1)
        aggregate, edge_weights, null_weight = self._aggregate(values, scores, target_index, length)
        raw_delta = self.output(self.norm(aggregate))
        gate = self._gate(raw_delta, target_h)
        delta = raw_delta * gate.unsqueeze(-1)
        return {"delta": delta, "edge_weights": edge_weights, "null_weight": null_weight, "content_norm": content_norm, "gate": gate}


class ReciprocalAdapter(nn.Module):
    """R->P and P->R residuals over frozen upstream hidden states."""

    def __init__(self, config: AdapterConfig | None = None):
        super().__init__()
        self.config = config or AdapterConfig()
        dims = {mode: geometry_dimension(mode, self.config.rbf_bins) for mode in ("G0", "G1", "G2", "G3")}
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
        residual = self.config.residual
        if self.config.partner_centered_residual and residual == "direct":
            residual = "partner_centered"
        direction_args = dict(
            sequence_independent_attention=v2,
            partner_centered_residual=bool(self.config.partner_centered_residual),
            conservative_gate=bool(self.config.conservative_gate),
            gate_init=float(self.config.gate_init),
            correct_null_softmax=v2,
            interaction=self.config.interaction,
            residual=residual,
        )
        rna_gate_mode = str(self.config.rna_gate_mode)
        if rna_gate_mode not in {"baseline", "fixed_quarter", "capped_quarter", "entropy_scaled"}:
            raise ValueError(f"unknown rna_gate_mode {rna_gate_mode}")
        rna_gate_scale = 0.25 if rna_gate_mode == "capped_quarter" else 1.0
        rna_fixed_gate = 0.25 if rna_gate_mode == "fixed_quarter" else None
        self.r2p = _Direction(
            direction_hidden_dim, self.config.token_dim, self.config.edge_dim, self.config.message_dim,
            4, 20, self.config.aggregation, self.config.layers, self.config.dropout, **direction_args,
        )
        # Entropy-scaled RNA residuals deliberately remove the learned scalar
        # gate.  The prior uncertainty is the only position-wise gate; the
        # Protein direction keeps the registered scalar-gate configuration.
        rna_direction_args = dict(direction_args)
        if rna_gate_mode == "entropy_scaled":
            rna_direction_args["residual"] = "direct"
            rna_direction_args["conservative_gate"] = False
        self.p2r = _Direction(
            direction_hidden_dim, self.config.token_dim, self.config.edge_dim, self.config.message_dim,
            20, 4, self.config.aggregation, self.config.layers, self.config.dropout,
            gate_scale=rna_gate_scale, fixed_gate=rna_fixed_gate, **rna_direction_args,
        )
        self.rna_gate_mode = rna_gate_mode

    def _select_geometry(self, full_geometry: Tensor) -> Tensor:
        bins = self.config.rbf_bins
        expected = geometry_dimension(self.config.geometry, bins)
        if full_geometry.shape[-1] == expected:
            return full_geometry
        if self.config.geometry in {"G2", "G3"}:
            if self.config.geometry == "G2":
                return full_geometry[:, : geometry_dimension("G2", bins)]
            raise ValueError("G3 requires a cache built with G3 geometry")
        if self.config.geometry == "G1":
            return full_geometry[:, : 6 * bins + 6]
        return torch.cat([full_geometry[:, :bins], full_geometry[:, 6 * bins : 6 * bins + 1]], dim=-1)

    def _hidden_inputs(self, p_h: Tensor, r_h: Tensor) -> tuple[Tensor, Tensor]:
        if not self.config.modality_projector:
            return p_h, r_h
        return self.protein_projector(self.protein_norm(p_h)), self.rna_projector(self.rna_norm(r_h))

    def _residual(
        self,
        direction: str,
        p_h: Tensor,
        r_h: Tensor,
        p_tokens: Tensor,
        r_tokens: Tensor,
        edge_index: Tensor,
        full_geometry: Tensor,
        token_off: bool,
        p_known: Tensor | None,
        r_known: Tensor | None,
    ) -> dict[str, Tensor]:
        geometry = self._select_geometry(full_geometry)
        if self.config.separate_edge_encoders:
            encoder = self.edge_encoder_r2p if direction == "protein" else self.edge_encoder_p2r
            assert encoder is not None
        else:
            encoder = self.shared_edge_encoder
        edge = encoder(geometry)
        if direction == "protein":
            return self.r2p(p_h, r_h, r_tokens, edge_index, edge, token_off, r_known)
        if direction == "rna":
            # Cached cross edges are indexed (protein, RNA); the RNA branch
            # receives (RNA target, protein partner).
            return self.p2r(r_h, p_h, p_tokens, edge_index[[1, 0]], edge, token_off, p_known)
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
        protein_known: Tensor | None = None,
        rna_known: Tensor | None = None,
    ) -> dict[str, Tensor]:
        # Keep the old single-edge call valid for existing correctness tests
        # and old diagnostic scripts.
        if edge_index_p2r is None:
            edge_index_p2r = edge_index_r2p
        if edge_geometry_p2r is None:
            edge_geometry_p2r = edge_geometry_r2p
        p_h_input, r_h_input = self._hidden_inputs(p_h, r_h)
        p_details = self._residual("protein", p_h_input, r_h_input, p_tokens, r_tokens, edge_index_r2p, edge_geometry_r2p, token_off, protein_known, rna_known)
        r_details = self._residual("rna", p_h_input, r_h_input, p_tokens, r_tokens, edge_index_p2r, edge_geometry_p2r, token_off, protein_known, rna_known)
        p_delta, r_delta = p_details["delta"], r_details["delta"]
        if self.rna_gate_mode == "entropy_scaled":
            # r_base is the frozen prior log-probability over A/U/G/C.
            # Entropy is normalized by log(4), so the gate is in [0, 1].
            prior_prob = r_base.exp().clamp_min(torch.finfo(r_base.dtype).tiny)
            entropy = -(prior_prob * r_base).sum(dim=-1)
            entropy_gate = entropy.div(math.log(4.0)).clamp(0.0, 1.0)
            r_delta = r_delta * entropy_gate.unsqueeze(-1)
            r_gate = entropy_gate
        else:
            r_gate = r_details["gate"]
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
            "rna_gate": r_gate,
            "rna_entropy_gate": r_gate,
            "rna_delta_raw": r_details["delta"],
        }
