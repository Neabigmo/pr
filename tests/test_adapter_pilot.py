from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pr_pilot.adapter_pilot.geometry import _one_feature
from pr_pilot.adapter_pilot.model import AdapterConfig, ReciprocalAdapter
from pr_pilot.adapter_pilot.priors import NAMPrior, ProteinMPNNPrior, RNA_MODEL_COLUMNS, _protein_features


ROOT = Path(r"F:\111临时\PR PILOT")
CHECKOUTS = ROOT / "third_party_checkouts_local_20260907"
PROTEIN_CHECKPOINT = ROOT / "prior_benchmark_local_20260906" / "03_compute" / "official_baselines" / "seed20260905" / "development" / "ProteinMPNN" / "model_weights" / "epoch61_step2806.pt"
RNA_CHECKPOINT = ROOT / "prior_benchmark_local_20260906" / "03_compute" / "official_baselines" / "seed20260905" / "development" / "NA-MPNN" / "s_996.pt"
PROTEIN_PDB = ROOT / "prior_benchmark_local_20260906" / "02_data_views" / "frozen_complex_test_86" / "protein_pdb" / "1TFW-assembly1.pdb"
RNA_PDB = ROOT / "prior_benchmark_local_20260906" / "02_data_views" / "frozen_complex_test_86" / "rna_pdb" / "1TFW-assembly1.pdb"


def test_zero_initialized_adapter_is_exact():
    torch.manual_seed(1)
    p_len, r_len = 7, 5
    edge_index = torch.tensor([[0, 1, 2, 4], [0, 1, 3, 2]])
    model = ReciprocalAdapter(AdapterConfig()).eval()
    p_base = torch.randn(p_len, 20)
    r_base = torch.randn(r_len, 4)
    out = model(
        p_base,
        r_base,
        torch.randn(p_len, 128),
        torch.randn(r_len, 128),
        torch.randint(0, 20, (p_len,)),
        torch.randint(0, 4, (r_len,)),
        edge_index,
        torch.randn(edge_index.shape[1], 114),
    )
    assert float((out["protein_logits"] - p_base).abs().max()) < 1e-6
    assert float((out["rna_logits"] - r_base).abs().max()) < 1e-6


def test_geometry_is_se3_invariant():
    rng = np.random.default_rng(7)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    translation = rng.normal(size=3)
    p_anchor = np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    r_anchor = np.asarray([[4.0, 1.0, 2.0], [5.0, 2.0, 1.0], [3.0, 4.0, 2.0]], dtype=np.float32)
    p_frame = np.eye(3, dtype=np.float32)
    r_frame = rotation.astype(np.float32)
    original = _one_feature(p_anchor, np.ones(2, bool), p_frame, True, r_anchor, np.ones(3, bool), r_frame, True, p_anchor[0], r_anchor[0], 16, "G2")
    p_anchor_t = (p_anchor @ rotation.T + translation).astype(np.float32)
    r_anchor_t = (r_anchor @ rotation.T + translation).astype(np.float32)
    transformed = _one_feature(p_anchor_t, np.ones(2, bool), (rotation @ p_frame).astype(np.float32), True, r_anchor_t, np.ones(3, bool), (rotation @ r_frame).astype(np.float32), True, p_anchor_t[0], r_anchor_t[0], 16, "G2")
    assert float(np.max(np.abs(original - transformed))) < 1e-5


def test_missing_anchors_do_not_create_geometry():
    feature = _one_feature(
        np.asarray([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32),
        np.asarray([True, False]),
        np.eye(3, dtype=np.float32),
        True,
        np.asarray([[4.0, 1.0, 2.0], [5.0, 2.0, 1.0], [3.0, 4.0, 2.0]], dtype=np.float32),
        np.asarray([False, False, False]),
        np.eye(3, dtype=np.float32),
        True,
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        np.asarray([4.0, 1.0, 2.0], dtype=np.float32),
        16,
        "G1",
    )
    assert float(np.abs(feature).sum()) == 0.0


def test_reciprocal_partner_vocabularies_are_distinct():
    model = ReciprocalAdapter(AdapterConfig())
    assert model.r2p.token_embedding.num_embeddings == 4
    assert model.r2p.output.out_features == 20
    assert model.p2r.token_embedding.num_embeddings == 20
    assert model.p2r.output.out_features == 4


def test_directional_edge_encoders_and_edge_sets_are_independent():
    shared = ReciprocalAdapter(AdapterConfig(separate_edge_encoders=False))
    separate = ReciprocalAdapter(AdapterConfig(separate_edge_encoders=True))
    assert separate.edge_encoder_r2p is not None
    assert separate.edge_encoder_p2r is not None
    assert separate.edge_encoder_r2p is not separate.edge_encoder_p2r
    assert shared.edge_encoder_r2p is None
    assert shared.edge_encoder_p2r is None

    p_base = torch.randn(3, 20)
    r_base = torch.randn(4, 4)
    common = (torch.randn(3, 128), torch.randn(4, 128), torch.randint(0, 20, (3,)), torch.randint(0, 4, (4,)))
    r2p_edges = torch.tensor([[0, 1], [2, 3]])
    p2r_edges = torch.tensor([[2], [1]])
    out = separate(p_base, r_base, *common, r2p_edges, torch.randn(2, 114), p2r_edges, torch.randn(1, 114))
    assert out["protein_logits"].shape == p_base.shape
    assert out["rna_logits"].shape == r_base.shape


def _v2_toy_inputs():
    p_len, r_len = 4, 3
    p_base = torch.randn(p_len, 20)
    r_base = torch.randn(r_len, 4)
    p_hidden = torch.randn(p_len, 128)
    r_hidden = torch.randn(r_len, 128)
    p_tokens = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    r_tokens = torch.tensor([0, 1, 2], dtype=torch.long)
    edges = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]], dtype=torch.long)
    geometry = torch.randn(edges.shape[1], 114)
    return p_base, r_base, p_hidden, r_hidden, p_tokens, r_tokens, edges, geometry


def _v2_model(**kwargs):
    model = ReciprocalAdapter(
        AdapterConfig(
            sequence_independent_attention=True,
            partner_centered_residual=True,
            token_dim=64,
            **kwargs,
        )
    ).eval()
    with torch.no_grad():
        model.r2p.output.weight.normal_(0.0, 0.2)
        model.p2r.output.weight.normal_(0.0, 0.2)
    return model


def test_v2_null_softmax_has_one_common_normalization():
    model = _v2_model()
    inputs = _v2_toy_inputs()
    out = model(*inputs)
    edge_index = inputs[6]
    weights = out["protein_edge_weights"]
    null = out["protein_null_weight"]
    total = torch.zeros(inputs[0].shape[0])
    total.index_add_(0, edge_index[0], weights)
    assert torch.allclose(total + null, torch.ones_like(total), atol=1e-6)


def test_v2_partner_centered_token_off_is_exact_prior():
    model = _v2_model()
    inputs = _v2_toy_inputs()
    native = model(*inputs, token_off=False)
    off = model(*inputs, token_off=True)
    assert torch.count_nonzero(off["protein_delta"]) == 0
    assert torch.count_nonzero(off["rna_delta"]) == 0
    assert torch.allclose(off["protein_logits"], inputs[0])
    assert torch.allclose(off["rna_logits"], inputs[1])
    with torch.no_grad():
        model.r2p.token_embedding.weight.zero_()
    zero_identity = model(*inputs, token_off=False)
    assert torch.allclose(zero_identity["protein_delta"], torch.zeros_like(native["protein_delta"]), atol=1e-6)


def test_v2_partner_token_changes_residual():
    model = _v2_model()
    inputs = list(_v2_toy_inputs())
    native = model(*inputs)["protein_delta"]
    inputs[5] = inputs[5].roll(1, 0)
    changed = model(*inputs)["protein_delta"]
    assert torch.count_nonzero(native - changed) > 0


def test_v2_no_partner_edge_has_zero_residual_and_unit_null_weight():
    model = _v2_model()
    inputs = list(_v2_toy_inputs())
    inputs[6] = torch.zeros((2, 0), dtype=torch.long)
    inputs[7] = torch.zeros((0, 114))
    out = model(*inputs)
    assert torch.count_nonzero(out["protein_delta"]) == 0
    assert torch.count_nonzero(out["rna_delta"]) == 0
    assert torch.allclose(out["protein_null_weight"], torch.ones(4))
    assert torch.allclose(out["rna_null_weight"], torch.ones(3))


def test_v2_gate_zero_returns_prior_exactly():
    model = _v2_model(conservative_gate=True)
    inputs = _v2_toy_inputs()
    with torch.no_grad():
        model.r2p.gate_logit.fill_(float("-inf"))
        model.p2r.gate_logit.fill_(float("-inf"))
    out = model(*inputs)
    assert torch.count_nonzero(out["protein_delta"]) == 0
    assert torch.count_nonzero(out["rna_delta"]) == 0
    assert torch.allclose(out["protein_logits"], inputs[0])
    assert torch.allclose(out["rna_logits"], inputs[1])


def test_v2_directional_indexing_uses_target_and_partner_axes():
    model = _v2_model()
    inputs = list(_v2_toy_inputs())
    inputs[6] = torch.tensor([[2], [1]], dtype=torch.long)
    inputs[7] = torch.randn(1, 114)
    baseline = model(*inputs)["rna_delta"]
    inputs[4] = inputs[4].clone()
    inputs[4][2] = (inputs[4][2] + 5) % 20
    changed_partner = model(*inputs)["rna_delta"]
    assert torch.count_nonzero(baseline - changed_partner) > 0
    inputs[4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    inputs[4][0] = (inputs[4][0] + 5) % 20
    changed_nonpartner = model(*inputs)["rna_delta"]
    assert torch.allclose(baseline, changed_nonpartner)


@pytest.mark.skipif(not (PROTEIN_PDB.exists() and RNA_PDB.exists() and PROTEIN_CHECKPOINT.exists() and RNA_CHECKPOINT.exists()), reason="frozen prior fixtures are not present")
def test_wrappers_reproduce_pinned_prior_probabilities():
    device = torch.device("cpu")
    protein = ProteinMPNNPrior(CHECKOUTS / "ProteinMPNN", PROTEIN_CHECKPOINT, device)
    entry, tensors = _protein_features(protein.module, PROTEIN_PDB, ["A"], device)
    x, _s, mask, _lengths, _chain_m, chain_encoding, *_ = tensors
    direct = protein.model.unconditional_probs(x, mask, tensors[12], chain_encoding)
    direct = F.log_softmax(direct[..., :20], dim=-1).squeeze(0)
    assert float((direct - protein.encode_backbone(PROTEIN_PDB, ["A"])["log_probs"]).abs().max()) < 1e-6

    rna = NAMPrior(CHECKOUTS / "NA-MPNN", RNA_CHECKPOINT, device)
    parsed, features, _keys = rna._features(RNA_PDB, ["A"])
    direct_rna = rna.model.unconditional_probs(features)["log_probs"][0]
    indices = torch.tensor([rna.restype_to_int[name] for name in RNA_MODEL_COLUMNS])
    direct_rna = direct_rna[:, indices]
    direct_rna = direct_rna - torch.logsumexp(direct_rna, dim=-1, keepdim=True)
    mask_rna = parsed["rna_mask"].bool()
    assert float((direct_rna[mask_rna] - rna.encode_backbone(RNA_PDB, ["A"])["log_probs"]).abs().max()) < 1e-6
