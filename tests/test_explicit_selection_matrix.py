import torch

from pr_pilot.adapter_pilot.explicit_matrix import ExplicitMatrixConfig, ExplicitSelectionAdapter


def _inputs():
    p_base = torch.zeros(2, 20)
    r_base = torch.zeros(3, 4)
    p_h = torch.zeros(2, 128)
    r_h = torch.zeros(3, 128)
    p_tokens = torch.tensor([2, 7])
    r_tokens = torch.tensor([1, 3, 0])
    # One shared pair appears in both directional edge lists.
    edges = torch.tensor([[0], [1]])
    geometry = torch.zeros(1, 114)
    return p_base, r_base, p_h, r_h, p_tokens, r_tokens, edges, geometry


def test_zero_initialization_is_exact_prior_identity():
    model = ExplicitSelectionAdapter(ExplicitMatrixConfig(variant="A2"))
    values = _inputs()
    out = model(*values[:6], values[6], values[7], values[6], values[7])
    assert torch.equal(out["protein_logits"], values[0])
    assert torch.equal(out["rna_logits"], values[1])


def test_a1_selects_rna_column_for_protein_and_protein_row_for_rna():
    model = ExplicitSelectionAdapter(ExplicitMatrixConfig(variant="A1"))
    with torch.no_grad():
        model.global_matrix.zero_()
        model.global_matrix[2, 1] = 3.0
        model.global_matrix[2, 3] = -2.0
        model.protein_gate_logit.fill_(0.0)
        model.rna_gate_logit.fill_(0.0)
    values = _inputs()
    out = model(*values[:6], values[6], values[7], values[6], values[7])
    # Protein target 0 selects RNA partner base 3: [..., -2].
    assert torch.isclose(out["protein_logits"][0, 2], torch.tensor(-1.0))
    # RNA target 1 selects protein partner AA 2: [..., 3, ..., -2].
    assert torch.isclose(out["rna_logits"][1, 1], torch.tensor(1.5))
    assert torch.isclose(out["rna_logits"][1, 3], torch.tensor(-1.0))
    assert torch.equal(out["protein_logits"][1], values[0][1])
    assert torch.equal(out["rna_logits"][0], values[1][0])


def test_no_selected_neighbor_has_zero_correction():
    model = ExplicitSelectionAdapter(ExplicitMatrixConfig(variant="A1"))
    values = _inputs()
    empty = torch.zeros((2, 0), dtype=torch.long)
    empty_geometry = torch.zeros((0, 114))
    out = model(*values[:6], empty, empty_geometry, empty, empty_geometry)
    assert torch.equal(out["protein_logits"], values[0])
    assert torch.equal(out["rna_logits"], values[1])


def test_ablation_variants_start_at_exact_prior_identity():
    values = _inputs()
    for variant in ("B1", "B2", "B3"):
        model = ExplicitSelectionAdapter(ExplicitMatrixConfig(variant=variant))
        out = model(*values[:6], values[6], values[7], values[6], values[7])
        assert torch.equal(out["protein_logits"], values[0])
        assert torch.equal(out["rna_logits"], values[1])
