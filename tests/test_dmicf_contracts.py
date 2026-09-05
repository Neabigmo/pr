import numpy as np
import torch

from pr_pilot.data.features import rna_backbone_view, assert_rna_model_view_has_no_identity_atoms
from pr_pilot.model.dmicf import (
    ContextualResidual, GlobalCompatibility, RelationalRelevance,
    SparseSequenceContextDecoder, PRBatch, DMICF, double_center, global_center,
    JointPriorAndFieldModel, set_trainable_stage,
)
from pr_pilot.training.losses import balanced_sequence_loss


def test_double_center_zero_row_and_column_means():
    x=torch.randn(7,20,4); y=double_center(x)
    assert torch.allclose(y.mean(-1),torch.zeros_like(y.mean(-1)),atol=1e-6)
    assert torch.allclose(y.mean(-2),torch.zeros_like(y.mean(-2)),atol=1e-6)


def test_forward_center_preserves_base_independent_row_effects():
    x=torch.zeros(20,4); x[1,:]=2.0
    y=global_center(x)
    # Row 1 remains different from row 0 across every base; double-centering would erase this.
    assert torch.all((y[1]-y[0])>1.9)
    assert abs(float(y.mean()))<1e-6


def test_global_c_is_small_random_not_pmi_seed():
    torch.manual_seed(0); c=GlobalCompatibility(init_std=1e-3)()
    assert c.shape==(20,4); assert float(c.abs().max())<0.02; assert not torch.allclose(c,torch.zeros_like(c)); assert abs(float(c.mean()))<1e-6


def test_delta_c_zero_initialization_exact():
    head=ContextualResidual(edge_hidden=32); q=torch.randn(11,32); out=head(q)
    assert out.shape==(11,20,4); assert torch.count_nonzero(out)==0


def test_alpha_starts_as_distance_prior():
    rel=RelationalRelevance(edge_hidden=16,initial_tau=2.0); q=torch.randn(4,16); d=torch.tensor([2.,4.,3.,8.])
    assert torch.allclose(rel.edge_scores(q,d),-d/rel.tau.detach(),atol=1e-6)


def test_neighborhood_softmax_sums_to_one_per_group():
    scores=torch.tensor([1.,2.,3.,-1.,4.]); groups=torch.tensor([0,0,1,1,1]); alpha=RelationalRelevance.neighborhood_softmax(scores,groups,2)
    for g in [0,1]: assert torch.allclose(alpha[groups==g].sum(),torch.tensor(1.0),atol=1e-6)


def test_rna_view_drops_identity_atoms():
    atoms={"P":np.array([0.,0.,0.]),"C1'":np.array([1.,0.,0.]),"N9":np.array([2.,0.,0.]),"C8":np.array([3.,0.,0.])}
    view=rna_backbone_view(atoms); assert "N9" not in view.names and "C8" not in view.names; assert_rna_model_view_has_no_identity_atoms(view)


def test_unknown_same_chain_token_value_cannot_leak_when_known_false():
    dec=SparseSequenceContextDecoder(alphabet=4,edge_in=3,hidden=8,layers=2)
    h=torch.randn(3,8); ei=torch.tensor([[0,1,1,2],[1,0,2,1]]); ex=torch.randn(4,3); known=torch.zeros(3,dtype=torch.bool)
    a=dec(h,ei,ex,torch.tensor([0,0,0]),known); b=dec(h,ei,ex,torch.tensor([3,2,1]),known)
    assert torch.allclose(a,b,atol=1e-6)


def test_unknown_partner_contributes_exactly_zero():
    d=DMICF(hidden=8,pr_edge_in=5,edge_hidden=8,interaction_layers=1)
    hp=torch.randn(2,8); hr=torch.randn(2,8); pr=PRBatch(torch.tensor([0,0,1]),torch.tensor([0,1,1]),torch.randn(3,5),torch.tensor([3.,4.,5.]))
    field=d.field(hp,hr,pr,use_delta=False,learned_alpha=False)
    pc=d.protein_correction(field,pr,torch.tensor([0,1]),torch.zeros(2,dtype=torch.bool),2)
    rc=d.rna_correction(field,pr,torch.tensor([0,1]),torch.zeros(2,dtype=torch.bool),2)
    assert torch.count_nonzero(pc)==0 and torch.count_nonzero(rc)==0


def test_lambda_is_one_and_frozen_until_joint():
    m=JointPriorAndFieldModel(3,4,5,4,6,hidden=8,protein_layers=1,rna_layers=1,decoder_layers=1)
    assert torch.allclose(m.dmicf.lambda_p,torch.tensor(1.0)) and torch.allclose(m.dmicf.lambda_r,torch.tensor(1.0))
    for stage in ["global_c","delta_c","alpha"]:
        set_trainable_stage(m,stage)
        assert not m.dmicf.raw_lambda_p.requires_grad and not m.dmicf.raw_lambda_r.requires_grad
    set_trainable_stage(m,"joint")
    assert m.dmicf.raw_lambda_p.requires_grad and m.dmicf.raw_lambda_r.requires_grad


def test_joint_loss_equalizes_four_semantic_groups_not_token_counts():
    p_logits=torch.zeros(10,20); p_targets=torch.zeros(10,dtype=torch.long); p_mask=torch.ones(10,dtype=torch.bool); p_interface=torch.tensor([True]+[False]*9)
    r_logits=torch.zeros(5,4); r_targets=torch.zeros(5,dtype=torch.long); r_mask=torch.ones(5,dtype=torch.bool); r_interface=torch.tensor([True,True,False,False,False])
    loss=balanced_sequence_loss(p_logits,p_targets,p_mask,p_interface,r_logits,r_targets,r_mask,r_interface,"joint",0.0,0.0)
    assert torch.allclose(loss.total,torch.tensor(1.0),atol=1e-6)


def test_missing_semantic_group_renormalizes_instead_of_zero_fill():
    p_logits=torch.zeros(3,20); target=torch.zeros(3,dtype=torch.long); mask=torch.tensor([True,True,True]); interface=torch.tensor([False,False,False])
    loss=balanced_sequence_loss(p_logits,target,mask,interface,None,None,None,None,"protein",0.0,0.0)
    assert loss.raw_pi is None; assert torch.allclose(loss.total,torch.tensor(1.0),atol=1e-6)
