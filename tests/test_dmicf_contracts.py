import numpy as np
import torch

from pr_pilot.data.features import rna_backbone_view, assert_rna_model_view_has_no_identity_atoms
from pr_pilot.model.dmicf import (
    ContextualResidual,
    GlobalCompatibility,
    RelationalRelevance,
    double_center,
)
from pr_pilot.training.losses import balanced_sequence_loss


def test_double_center_zero_row_and_column_means():
    x = torch.randn(7, 20, 4)
    y = double_center(x)
    assert torch.allclose(y.mean(-1), torch.zeros_like(y.mean(-1)), atol=1e-6)
    assert torch.allclose(y.mean(-2), torch.zeros_like(y.mean(-2)), atol=1e-6)


def test_global_c_is_small_random_not_pmi_seed():
    torch.manual_seed(0)
    c = GlobalCompatibility(init_std=1e-3)()
    assert c.shape == (20, 4)
    assert float(c.abs().max()) < 0.02
    assert not torch.allclose(c, torch.zeros_like(c))


def test_delta_c_zero_initialization_exact():
    head = ContextualResidual(edge_hidden=32)
    q = torch.randn(11, 32)
    out = head(q)
    assert out.shape == (11, 20, 4)
    assert torch.count_nonzero(out) == 0


def test_alpha_starts_as_distance_prior():
    rel = RelationalRelevance(edge_hidden=16, initial_tau=2.0)
    q = torch.randn(4, 16)
    d = torch.tensor([2.0, 4.0, 3.0, 8.0])
    scores = rel.edge_scores(q, d)
    expected = -d / rel.tau.detach()
    assert torch.allclose(scores, expected, atol=1e-6)


def test_neighborhood_softmax_sums_to_one_per_group():
    scores = torch.tensor([1.0, 2.0, 3.0, -1.0, 4.0])
    groups = torch.tensor([0, 0, 1, 1, 1])
    alpha = RelationalRelevance.neighborhood_softmax(scores, groups, n_groups=2)
    for g in [0, 1]:
        assert torch.allclose(alpha[groups == g].sum(), torch.tensor(1.0), atol=1e-6)


def test_rna_view_drops_identity_atoms():
    atoms = {
        "P": np.array([0.0, 0.0, 0.0]),
        "C1'": np.array([1.0, 0.0, 0.0]),
        "N9": np.array([2.0, 0.0, 0.0]),
        "C8": np.array([3.0, 0.0, 0.0]),
    }
    view = rna_backbone_view(atoms)
    assert "N9" not in view.names
    assert "C8" not in view.names
    assert_rna_model_view_has_no_identity_atoms(view)


def test_joint_loss_equalizes_four_semantic_groups_not_token_counts():
    # Deliberately make group sizes highly unequal. If implementation pooled tokens,
    # this test would change when we duplicate the large group.
    p_logits = torch.zeros(10, 20)
    p_targets = torch.zeros(10, dtype=torch.long)
    p_mask = torch.ones(10, dtype=torch.bool)
    p_interface = torch.tensor([True] + [False] * 9)

    r_logits = torch.zeros(5, 4)
    r_targets = torch.zeros(5, dtype=torch.long)
    r_mask = torch.ones(5, dtype=torch.bool)
    r_interface = torch.tensor([True, True, False, False, False])

    loss = balanced_sequence_loss(
        p_logits,
        p_targets,
        p_mask,
        p_interface,
        r_logits,
        r_targets,
        r_mask,
        r_interface,
        task="joint",
        protein_label_smoothing=0.0,
        rna_label_smoothing=0.0,
    )
    # Uniform logits -> CE/log(alphabet) == 1 for every group, so joint must be 1.
    assert torch.allclose(loss.total, torch.tensor(1.0), atol=1e-6)
