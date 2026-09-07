"""Synthetic analytical checks; none of these numbers are model benchmark results."""
import numpy as np
import pandas as pd
import pytest
import torch
from pr_pilot.evaluation.audit_metrics import (
    multiclass_brier_score, top_label_brier_score, order_mixture_log_probability,
    compatibility_components, residual_drift_audit, directional_coefficient_gap,
)
from pr_pilot.evaluation.battery import brier_multiclass, native_probability_brier, paired_bootstrap, paired_wilcoxon, expected_calibration_error, holm_adjust
from pr_pilot.evaluation.paired_statistics import strict_align_pairs, signflip_test
from pr_pilot.evaluation.confirmatory import analyze_confirmatory, HYPOTHESES
from pr_pilot.evaluation.robustness import calibration_table
from pr_pilot.evaluation.runner import _rows_from_logits


@pytest.mark.parametrize('k',[4,20])
def test_true_brier_uniform(k):
    p=np.full((3,k),1/k); y=np.array([0,1,2])
    assert multiclass_brier_score(p,y)==pytest.approx(1-1/k)
    assert brier_multiclass(p,y)==pytest.approx(1-1/k)


def test_brier_perfect_and_confident_wrong():
    assert multiclass_brier_score(np.eye(4),np.arange(4))==0
    assert multiclass_brier_score(np.array([[1.,0.]]),np.array([1]))==2
    assert top_label_brier_score([.8],[1])==pytest.approx(.04)


@pytest.mark.parametrize('p,y', [([[.2,.3]],[0]), ([[np.nan,1]],[0]), ([[-1,2]],[0]), ([],[]), ([[.5,.5]],[2]), ([[.5,.5]],[0.5])])
def test_brier_invalid_inputs_fail(p,y):
    with pytest.raises(ValueError): multiclass_brier_score(p,np.asarray(y))


def test_legacy_brier_fail_closed():
    with pytest.raises(ValueError,match='Legacy'): brier_multiclass(np.array([.25]),np.array([.25]),np.array([1]))
    with pytest.raises(ValueError): native_probability_brier(np.array([.25]))


def test_missing_pairs_and_duplicate_targets_never_silently_drop():
    with pytest.raises(ValueError): strict_align_pairs(pd.Series([1,2],index=['a','b']),pd.Series([1,2],index=['a','c']))
    with pytest.raises(ValueError): strict_align_pairs(pd.Series([1,np.nan]),pd.Series([1,2]))
    with pytest.raises(ValueError): strict_align_pairs(pd.Series([1,2],index=['a','a']),pd.Series([1,2],index=['a','a']))
    paired=strict_align_pairs(pd.Series([1,2],index=['a','b']),pd.Series([3,4],index=['b','a']))
    assert paired.loc['a','b']==4


def test_exact_signflip_known_small_case():
    assert signflip_test([-1]*5,alternative='less')['p']==pytest.approx(1/32)
    assert signflip_test([-1]*5,alternative='two-sided')['p']==pytest.approx(2/32)
    assert signflip_test([0]*5)['p']==1


def test_mc_permutation_never_reports_zero_p():
    r=signflip_test([-1]*20,resamples=99,seed=1)
    assert r['p']>=1/100
    assert r['method']=='monte_carlo_signflip_plus_one'


def test_bootstrap_tail_is_not_mistaken_for_null_p():
    r=paired_bootstrap(pd.Series([-1.]*5),pd.Series([0.]*5),resamples=99)
    assert r['permutation_p_two_sided']==pytest.approx(2/32)
    assert r['bootstrap_p_two_sided']==r['permutation_p_two_sided']
    assert r['legacy_p_key_is_permutation_alias']
    assert paired_wilcoxon(pd.Series([1,2]),pd.Series([1,2]))['p']==1


def test_empty_or_invalid_calibration_rejected():
    with pytest.raises(ValueError): expected_calibration_error(np.array([]),np.array([]))
    with pytest.raises(ValueError): expected_calibration_error(np.array([1]),np.array([1.1]))
    assert expected_calibration_error(np.array([1]),np.array([1.]))==0


def test_calibration_uses_full_probability_vector():
    predictions=pd.DataFrame({
        'native_token':[0,1], 'predicted_token':[0,0], 'max_probability':[.7,.6],
        'probability_0':[.7,.6], 'probability_1':[.2,.3], 'probability_2':[.1,.1],
    })
    out=calibration_table(predictions)
    assert out['multiclass_brier']==pytest.approx(.5)
    assert out['top_label_brier']==pytest.approx((.3**2+.6**2)/2)
    with pytest.raises(ValueError): calibration_table(predictions.drop(columns=['probability_2']))


def test_runner_exports_full_probability_vector():
    rows=_rows_from_logits('synthetic', 'rna', torch.tensor([[1., 2., 3., 4.]]), torch.tensor([2]),
                           torch.tensor([True]), torch.tensor([True]), 'DMICF', 1)
    assert [rows[0][f'probability_{i}'] for i in range(4)]==pytest.approx(torch.softmax(torch.tensor([1.,2.,3.,4.]),0).tolist())


def test_holm_known_values_and_invalid():
    out=holm_adjust({'a':.01,'b':.03,'c':.04})
    assert out==pytest.approx({'a':.03,'b':.06,'c':.06})
    with pytest.raises(ValueError): holm_adjust({'a':float('nan')})


def test_order_mixture_not_average_log_probability():
    lp=np.log([.1,.9])
    assert order_mixture_log_probability(lp)==pytest.approx(np.log(.5))
    assert order_mixture_log_probability(lp)>lp.mean()
    assert np.isfinite(order_mixture_log_probability([-1000,-1001]))
    assert order_mixture_log_probability([-np.inf,-np.inf])==-np.inf
    with pytest.raises(ValueError): order_mixture_log_probability([1,2])
    with pytest.raises(ValueError): order_mixture_log_probability([-1,-2],[.4,.4])


def test_c_main_effect_is_not_partner_specific_interaction():
    c=np.zeros((20,4)); c[1,:]=2
    part=compatibility_components(c)
    assert np.allclose(part['interaction'],0)
    assert np.allclose(c,part['grand']+part['row_main']+part['column_main']+part['interaction'])
    assert np.linalg.norm(part['row_main'])>0


def test_residual_drift_zero_denominator_and_population_offset():
    c=np.zeros((20,4)); dc=np.ones((2,20,4))
    out=residual_drift_audit(c,dc)
    assert out['delta_rms_to_c'] is None
    assert out['mean_delta_to_delta_rms']==pytest.approx(1)
    assert residual_drift_audit(c,np.zeros_like(dc))['mean_delta_to_delta_rms'] is None
    with pytest.raises(ValueError): residual_drift_audit(c,dc,[-1,2])


def test_shared_scores_do_not_imply_reciprocal_weights():
    out=directional_coefficient_gap([.5,.5,1],[.5,1,.5])
    assert out['max_abs_gap']==.5


def effect_fixture():
    roster=pd.DataFrame({'sample_id':[f's{i}' for i in range(10)],'group_id':[f'g{i//2}' for i in range(10)]})
    seeds=[11,22,33]; rows=[]
    for h,cs in HYPOTHESES.items():
        for c in cs:
            for sid,gid in zip(roster.sample_id,roster.group_id):
                for seed in seeds:
                    rows.append(dict(hypothesis=h,component=c,sample_id=sid,group_id=gid,seed=seed,effect=-1.))
    return pd.DataFrame(rows),roster,seeds


def test_confirmatory_groups_and_seeds_not_pseudoreplicated():
    e,r,s=effect_fixture(); result=analyze_confirmatory(e,r,s,resamples=50)
    h=result['hypotheses']['full_vs_dual_prior_interface_nll']['components']['primary']
    assert h['n_units']==5 and h['n_complexes']==10 and h['n_training_seeds']==3
    assert h['p']==pytest.approx(1/32)
    assert len(result['hypotheses'])==5


def test_h2_requires_both_controls_not_favorable_one():
    e,r,s=effect_fixture()
    e.loc[(e.hypothesis=='full_vs_partner_identity_controls') & (e.component=='geometry_only'),'effect']=1.
    result=analyze_confirmatory(e,r,s,resamples=50)['hypotheses']['full_vs_partner_identity_controls']
    assert result['p_unadjusted']==1
    assert not result['reject_in_prespecified_direction']


@pytest.mark.parametrize('fault',['missing_row','duplicate','nonfinite','wrong_group','missing_hypothesis','extra_seed'])
def test_confirmatory_incomplete_exports_fail(fault):
    e,r,s=effect_fixture()
    if fault=='missing_row': e=e.iloc[1:]
    elif fault=='duplicate': e=pd.concat([e,e.iloc[:1]])
    elif fault=='nonfinite': e.loc[0,'effect']=np.nan
    elif fault=='wrong_group': e.loc[0,'group_id']='not_registered'
    elif fault=='missing_hypothesis': e=e[e.hypothesis!='partner_scramble']
    else: e.loc[0,'seed']=1234
    with pytest.raises(ValueError): analyze_confirmatory(e,r,s,resamples=50)


def test_token_metrics_do_not_silently_skip_missing_log_probabilities():
    from pr_pilot.evaluation.battery import token_metrics, empirical_pmi
    table=pd.DataFrame({'native_log_probability':[-1.,np.nan], 'native_token':[0,1], 'predicted_token':[0,0]})
    with pytest.raises(ValueError): token_metrics(table,4)
    with pytest.raises(ValueError): empirical_pmi(np.zeros((20,4)))
    assert np.isfinite(empirical_pmi(np.ones((20,4)))).all()
