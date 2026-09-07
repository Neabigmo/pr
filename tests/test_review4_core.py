"""Synthetic tensor contracts, NOT experimental-structure or biological tests."""
from types import SimpleNamespace
from unittest.mock import patch
import math
import numpy as np
import pytest
import torch

from pr_pilot.model.dmicf import JointPriorAndFieldModel, PRBatch, set_trainable_stage
from pr_pilot.inference.sampler import sample_joint, _sample_token
from pr_pilot.inference.scoring import teacher_forced_order_score, leave_one_out_pair_score


def fixture():
    torch.manual_seed(214)
    torch.set_num_threads(1)
    model = JointPriorAndFieldModel(3, 4, 5, 4, 6, hidden=8,
                                   protein_layers=2, rna_layers=2, decoder_layers=2,
                                   interaction_layers=2, drop_path=.05)
    def graph(n, width, alphabet):
        src = torch.arange(n-1); dst = src+1
        ei = torch.stack([torch.cat([src,dst]), torch.cat([dst,src])])
        return SimpleNamespace(node_x=torch.randn(n,width), edge_index=ei,
                               edge_x=torch.randn(ei.shape[1],4),
                               sequence=torch.arange(n) % alphabet,
                               valid=torch.ones(n,dtype=torch.bool),
                               fixed=torch.zeros(n,dtype=torch.bool),
                               interface=torch.tensor([True]*(n-1)+[False]))
    p, r = graph(4,3,20), graph(3,5,4)
    p.fixed[0] = True
    pr = PRBatch(torch.tensor([0,1,1,2]), torch.tensor([0,0,1,1]),
                 torch.randn(4,6), torch.tensor([3.,4.,3.5,5.]))
    sample = SimpleNamespace(protein=p,rna=r,pr=pr,sample_id="SYNTHETIC_NOT_A_PDB")
    return model, sample


def static_args(sample):
    p,r=sample.protein,sample.rna
    return (p.node_x,p.edge_index,p.edge_x,r.node_x,r.edge_index,r.edge_x,sample.pr)


@pytest.mark.parametrize("use_delta,learned_alpha", [(False,False),(True,False),(True,True)])
@pytest.mark.parametrize("known_mode", ["none","partial","all"])
def test_cached_logits_equal_reference(use_delta,learned_alpha,known_mode):
    model,s=fixture(); model.eval()
    with torch.no_grad():
        model.dmicf.delta.out.weight.normal_(std=.05)
        model.dmicf.relevance.score.weight.normal_(std=.05)
    pk=torch.ones(4,dtype=torch.bool); rk=torch.ones(3,dtype=torch.bool)
    if known_mode=="none": pk[:]=False; rk[:]=False
    if known_mode=="partial": pk[1:]=False; rk[::2]=False
    cache=model.prepare_inference_cache(*static_args(s),use_delta,learned_alpha)
    a=model(*static_args(s),s.protein.sequence,s.rna.sequence,pk,rk,use_delta,learned_alpha)
    b=model.decode_cached(cache,s.protein.sequence,s.rna.sequence,pk,rk)
    assert a.keys()==b.keys()
    for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


def test_cached_field_never_depends_on_partner_token():
    model,s=fixture(); model.eval()
    cache=model.prepare_inference_cache(*static_args(s))
    pk=torch.zeros(4,dtype=torch.bool); rk=torch.ones(3,dtype=torch.bool)
    a=model.decode_cached(cache,s.protein.sequence,s.rna.sequence,pk,rk)
    changed=(s.rna.sequence+1)%4
    b=model.decode_cached(cache,s.protein.sequence,changed,pk,rk)
    for key in ['protein_hidden','rna_hidden','C','DeltaC','alpha_p','alpha_r','protein_struct_logits']:
        assert torch.equal(a[key],b[key])
    assert not torch.equal(a['protein_delta_logits'],b['protein_delta_logits'])


def test_unknown_partner_exact_zero_with_cache():
    model,s=fixture(); model.eval(); c=model.prepare_inference_cache(*static_args(s))
    out=model.decode_cached(c,s.protein.sequence,s.rna.sequence,torch.zeros(4,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    assert torch.count_nonzero(out['protein_delta_logits'])==0
    assert torch.count_nonzero(out['rna_delta_logits'])==0


@pytest.mark.parametrize("mutation", ['parameter','geometry','replace_pr','mode'])
def test_stale_cache_rejected(mutation):
    model,s=fixture(); model.eval(); c=model.prepare_inference_cache(*static_args(s))
    with torch.no_grad():
        if mutation=='parameter': model.protein_head.weight.add_(1)
        elif mutation=='geometry': s.pr.edge_features.add_(1)
        elif mutation=='replace_pr': s.pr.edge_features=s.pr.edge_features.clone()
        else: model.protein_decoder.train()
    with pytest.raises(RuntimeError):
        model.decode_cached(c,s.protein.sequence,s.rna.sequence,torch.ones(4,dtype=torch.bool),torch.ones(3,dtype=torch.bool))


def test_cache_rejects_training_and_other_model():
    model,s=fixture()
    with pytest.raises(RuntimeError): model.prepare_inference_cache(*static_args(s))
    model.eval(); c=model.prepare_inference_cache(*static_args(s))
    other,_=fixture(); other.eval()
    with pytest.raises(RuntimeError): other.decode_cached(c,s.protein.sequence,s.rna.sequence,torch.zeros(4,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))


@pytest.mark.parametrize('stage,prefixes', [
    ('protein_prior',('protein_encoder.','protein_decoder.','protein_head.')),
    ('rna_prior',('rna_encoder.','rna_decoder.','rna_head.')),
    ('global_c',('dmicf.global_c.',)),
    ('delta_c',('dmicf.interaction.','dmicf.delta.')),
    ('alpha',('dmicf.relevance.',)),
])
def test_low_level_stage_ownership(stage,prefixes):
    model,_=fixture(); set_trainable_stage(model,stage)
    for name,param in model.named_parameters():
        assert param.requires_grad==name.startswith(prefixes), name


def test_primary_joint_freezes_c_without_changing_checkpoint_keys():
    model,_=fixture(); keys=set(model.state_dict()); set_trainable_stage(model,'joint')
    assert not model.dmicf.global_c.raw.requires_grad
    assert model.dmicf.raw_lambda_p.requires_grad
    assert set(model.state_dict())==keys


@pytest.mark.parametrize('order',['mixed','protein_first','rna_first'])
@pytest.mark.parametrize('cycles',[0,1,3])
def test_cached_sampler_matches_reference_and_respects_constraints(order,cycles):
    model,s=fixture()
    a=sample_joint(model,s,candidates=2,seed=47,spir_cycles=cycles,order_mode=order,use_cache=False)
    b=sample_joint(model,s,candidates=2,seed=47,spir_cycles=cycles,order_mode=order,use_cache=True)
    for x,y in zip(a,b):
        assert torch.equal(x.protein_tokens,y.protein_tokens)
        assert torch.equal(x.rna_tokens,y.rna_tokens)
        assert x.token_model_logprobs==y.token_model_logprobs
        assert torch.equal(y.protein_tokens[s.protein.fixed],s.protein.sequence[s.protein.fixed])
        assert torch.equal(y.protein_tokens[~s.protein.interface],y.pre_spir_protein[~s.protein.interface])
        assert torch.equal(y.rna_tokens[~s.rna.interface],y.pre_spir_rna[~s.rna.interface])


def test_sampling_only_encodes_once_and_restores_module_modes():
    model,s=fixture(); model.protein_encoder.eval()
    modes=[m.training for m in model.modules()]
    with patch.object(model.protein_encoder,'forward',wraps=model.protein_encoder.forward) as p:
        sample_joint(model,s,candidates=3,spir_cycles=1,use_cache=True)
        assert p.call_count==1
    assert [m.training for m in model.modules()]==modes
    with patch.object(model.protein_encoder,'forward',wraps=model.protein_encoder.forward) as p:
        sample_joint(model,s,candidates=3,spir_cycles=1,use_cache=False)
        assert p.call_count>18


def test_sampling_restores_modes_after_exception():
    model,s=fixture(); modes=[m.training for m in model.modules()]
    with pytest.raises(ValueError): sample_joint(model,s,candidates=1,order_mode='invalid')
    assert [m.training for m in model.modules()]==modes


@pytest.mark.parametrize('kwargs',[{'candidates':0},{'temperature':float('nan')},{'temperature':-1},
                                   {'spir_cycles':-1},{'spir_reopen_fraction':2}])
def test_sampler_parameter_validation(kwargs):
    model,s=fixture()
    with pytest.raises(ValueError): sample_joint(model,s,**kwargs)


def test_greedy_sampling_log_probability_is_zero():
    i,lp=_sample_token(torch.tensor([1.,0.,-1.]),0,torch.Generator().manual_seed(1))
    assert i==0 and lp==0


def test_order_score_equals_model_path_not_temperature_sampling_path():
    model,s=fixture()
    c=sample_joint(model,s,candidates=1,temperature=.6,spir_enabled=False)[0]
    score=teacher_forced_order_score(model,s,c.protein_tokens,c.rna_tokens,c.decoding_order)
    assert score['sequence_log_probability']==pytest.approx(sum(c.token_model_logprobs))
    assert not np.isclose(sum(c.token_logprobs),score['sequence_log_probability'])


def test_loo_masks_own_token_and_never_calls_it_joint_likelihood():
    model,s=fixture()
    import pr_pilot.inference.scoring as module
    actual=module._forward; masks=[]
    def checked(model,sample,pt,rt,pk,rk,cache):
        masks.append((pk.clone(),rk.clone()))
        return actual(model,sample,pt,rt,pk,rk,cache)
    with patch.object(module,'_forward',side_effect=checked):
        report=leave_one_out_pair_score(model,s,s.protein.sequence,s.rna.sequence)
    assert report['score_kind']=='leave_one_out_compatibility_not_joint_likelihood'
    for row,(pk,rk) in zip(report['tokens'],masks):
        assert int((~pk).sum()+(~rk).sum())==1
        assert not (pk if row['polymer']=='P' else rk)[row['index']]
    assert len(report['tokens'])==6
    assert 'sequence_log_probability' not in report


def test_scoring_does_not_accept_fixed_token_changes_or_incomplete_order():
    model,s=fixture(); p=s.protein.sequence.clone(); p[0]=19
    with pytest.raises(ValueError): leave_one_out_pair_score(model,s,p,s.rna.sequence)
    with pytest.raises(ValueError): teacher_forced_order_score(model,s,s.protein.sequence,s.rna.sequence,[])


def test_no_designable_positions_have_absent_loo_not_fake_zero():
    model,s=fixture(); s.protein.fixed[:]=True; s.rna.fixed[:]=True
    report=leave_one_out_pair_score(model,s,s.protein.sequence,s.rna.sequence)
    assert report['balanced_normalized_score'] is None
    assert report['P_mean_nll'] is None


def test_backbone_only_spir_does_not_use_native_heavy_atom_interface_labels():
    model,s=fixture()
    a=sample_joint(model,s,candidates=2,seed=1234,spir_cycles=1)
    s.protein.interface[:]=False; s.rna.interface[:]=False
    b=sample_joint(model,s,candidates=2,seed=1234,spir_cycles=1)
    for x,y in zip(a,b):
        assert torch.equal(x.protein_tokens,y.protein_tokens)
        assert torch.equal(x.rna_tokens,y.rna_tokens)
        assert y.spir_interface_scope=='design_graph'


def test_legacy_spir_scope_is_explicit_and_preserves_empty_interface():
    model,s=fixture(); s.protein.interface[:]=False; s.rna.interface[:]=False
    a=sample_joint(model,s,candidates=1,spir_interface_scope='canonical_legacy')[0]
    assert torch.equal(a.protein_tokens,a.pre_spir_protein)
    assert torch.equal(a.rna_tokens,a.pre_spir_rna)
    assert a.spir_interface_scope=='canonical_legacy'


@pytest.mark.parametrize('fault',['distance_column','float_index','negative_index','outside_index','nonfinite'])
def test_pr_geometry_contract_rejects_silent_broadcast_and_bad_values(fault):
    _,s=fixture()
    if fault=='distance_column': s.pr.effective_distance=s.pr.effective_distance[:,None]
    elif fault=='float_index': s.pr.protein_index=s.pr.protein_index.float()
    elif fault=='negative_index': s.pr.protein_index[0]=-1
    elif fault=='outside_index': s.pr.rna_index[0]=3
    else: s.pr.edge_features[0,0]=float('nan')
    with pytest.raises(ValueError): s.pr.validate(4,3,check_values=True)


@pytest.mark.parametrize("customization", ["instance_forward", "forward_hook"])
def test_cache_does_not_bypass_custom_model_forward(customization):
    model, sample = fixture()
    calls = []
    if customization == "instance_forward":
        original = model.forward
        def customized(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)
        model.forward = customized
    else:
        model.register_forward_hook(lambda module, args, output: calls.append(True))
    with patch.object(model, 'prepare_inference_cache', wraps=model.prepare_inference_cache) as cached:
        sample_joint(model, sample, candidates=1, spir_enabled=False, use_cache=True)
        assert cached.call_count == 0
    assert len(calls) == 6
