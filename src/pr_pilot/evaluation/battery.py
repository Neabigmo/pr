"""Held-out 100-complex evaluation primitives and mandatory test registry."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class MandatoryTest:
    name: str
    family: str
    primary: bool
    description: str


MANDATORY_TESTS = [
    MandatoryTest("core_sequence_metrics", "prediction", True, "Protein/RNA raw and entropy-normalized NLL, recovery; interface/non-interface; macro/micro."),
    MandatoryTest("conditional_protein", "prediction", True, "RNA-known protein design; official ProteinMPNN is a partner-blind external reference, not a same-information causal control."),
    MandatoryTest("conditional_rna", "prediction", True, "Protein-known RNA design; MPNN-fixbb/NA-MPNN is a partner-blind external reference."),
    MandatoryTest("joint_mixed_order", "prediction", True, "Teacher-forced native pseudo-NLL under multiple mixed Protein/RNA orders."),
    MandatoryTest("partner_scramble", "mechanism", True, "Composition-preserving partner scrambling; interface DeltaNLL."),
    MandatoryTest("counterfactual_partner_mutation", "mechanism", True, "Bidirectional single partner-token perturbation; local KL vs distance."),
    MandatoryTest("global_c_vs_pmi", "interpretability", True, "Post-hoc interaction-only C vs held-out PMI with permutation null; PMI never initializes the main model."),
    MandatoryTest("c_seed_stability", "interpretability", True, "Across-seed correlation and sign stability of learned C."),
    MandatoryTest("delta_c_context", "interpretability", True, "Residual magnitude, geometry stratification and contextual reversal cases."),
    MandatoryTest("alpha_relevance", "interpretability", True, "Entropy/effective neighbours/distance relation/non-nearest relevance."),
    MandatoryTest("alpha_edge_removal", "interpretability", True, "Delete top-alpha edge vs distance-matched lower-alpha edge and compare target NLL change; alpha heatmaps alone are not mechanistic evidence."),
    MandatoryTest("coordinate_noise_robustness", "robustness", True, "0/0.05/0.10/0.20 A coordinate noise."),
    MandatoryTest("pr_edge_dropout_robustness", "robustness", True, "0/5/10/20% cross-edge removal."),
    MandatoryTest("partner_hide_robustness", "robustness", True, "0/10/20/40% known partner token hiding."),
    MandatoryTest("geometry_permutation", "robustness", True, "Permute rich PR geometry among edges while preserving graph topology to test use of local geometry."),
    MandatoryTest("decoding_order_sensitivity", "inference", True, "Mixed random-order variance plus protein-first/RNA-first directional bias."),
    MandatoryTest("spir_ablation", "inference", True, "No refinement vs one SPIR vs repeated refinement using identical candidate budgets."),
    MandatoryTest("sequence_collapse_audit", "inference", True, "Candidate diversity, composition and pre/post-SPIR collapse."),
    MandatoryTest("calibration", "reliability", True, "ECE/Brier/reliability for protein and RNA separately."),
    MandatoryTest("partner_blind_complex_control", "fairness", True, "Same 1000 complex backbones and same optimization schedule, but cross-partner identities are never revealed."),
    MandatoryTest("geometry_only_capacity_control", "fairness", True, "Comparable extra PR geometry capacity but no partner token identity, separating capacity/extra-structure exposure from sequence coupling."),
    MandatoryTest("official_from_scratch_baselines", "fairness", True, "Exact frozen 1000 single-molecule IDs, approximately matched data passes, random initialization."),
    MandatoryTest("published_checkpoint_reference", "fairness", False, "Published pretrained checkpoints are a separate reference track and never pooled with primary from-scratch comparisons."),
    MandatoryTest("parameter_compute_report", "fairness", True, "Trainable/total parameters, data passes/tokens, peak memory, wall time and GPU-hours."),
    MandatoryTest("strict_ood_shift_audit", "data", True, "Dev vs strict bilateral OOD test distributions for length/interface size/resolution/method/type; final test is not called IID random."),
    MandatoryTest("ablation_ladder", "ablation", True, "Scratch -> priors -> C -> DeltaC -> alpha -> joint -> SPIR."),
    MandatoryTest("rich_vs_distance_geometry", "ablation", True, "Rich multi-atom/frame geometry vs distance-only with matched training data."),
    MandatoryTest("fixed_vs_learned_alpha", "ablation", True, "Distance prior alpha vs learned relational relevance."),
    MandatoryTest("fixed_training_pmi_baseline", "ablation", True, "A separate PMI-fixed baseline; main C remains randomly initialized."),
    MandatoryTest("data_efficiency", "ablation", True, "Nested identical 10/25/50/100% complex subsets: scratch vs pretrained/DM-ICF."),
    MandatoryTest("seed_stability", "reproducibility", True, "At least three primary training seeds; separate target variance from training-seed variance."),
    MandatoryTest("complex_level_statistics", "statistics", True, "10,000 paired bootstraps and paired non-parametric secondary tests; Holm primary and BH exploratory correction."),
    MandatoryTest("independent_structure_prediction", "external", False, "Matched candidate/predictor/MSA/template/seed budget; predictor confidence is never called binding energy."),
]


def token_metrics(df: pd.DataFrame, alphabet_size: int) -> dict[str, float]:
    required={"native_log_probability","native_token","predicted_token"}; missing=required-set(df.columns)
    if missing: raise ValueError(f"Prediction table missing {sorted(missing)}")
    if len(df)==0: return {"n":0,"nll":float("nan"),"normalized_nll":float("nan"),"perplexity":float("nan"),"recovery":float("nan")}
    nll=float(-df["native_log_probability"].mean())
    return {"n":int(len(df)),"nll":nll,"normalized_nll":nll/math.log(alphabet_size),"perplexity":float(math.exp(min(nll,50.0))),"recovery":float((df["native_token"]==df["predicted_token"]).mean())}


def macro_by_complex(df: pd.DataFrame, alphabet_size: int) -> pd.DataFrame:
    rows=[]
    for sample_id,part in df.groupby("sample_id",sort=True):
        metrics=token_metrics(part,alphabet_size); metrics["sample_id"]=sample_id; rows.append(metrics)
    return pd.DataFrame(rows)


def expected_calibration_error(correct: np.ndarray, confidence: np.ndarray, bins: int = 15) -> float:
    correct=np.asarray(correct,float); confidence=np.asarray(confidence,float)
    if correct.shape!=confidence.shape: raise ValueError("correct/confidence shape mismatch")
    edges=np.linspace(0,1,bins+1); ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(confidence>=lo)&(confidence<(hi if hi<1 else hi+1e-12))
        if mask.any(): ece+=mask.mean()*abs(correct[mask].mean()-confidence[mask].mean())
    return float(ece)


def brier_multiclass(native_probability: np.ndarray, max_probability: np.ndarray, correct: np.ndarray) -> float:
    """Confidence Brier proxy when complete external-baseline logits are unavailable."""
    correct=np.asarray(correct,float); max_probability=np.asarray(max_probability,float)
    return float(np.mean((max_probability-correct)**2))


def native_probability_brier(native_probability: np.ndarray) -> float:
    p=np.asarray(native_probability,float); return float(np.mean((1.0-p)**2))


def paired_bootstrap(a: pd.Series,b: pd.Series,resamples:int=10000,seed:int=20260905,statistic:Callable[[np.ndarray],float]=np.mean)->dict[str,float]:
    aligned=pd.concat([a.rename("a"),b.rename("b")],axis=1).dropna(); diff=(aligned["a"]-aligned["b"]).to_numpy(float)
    if len(diff)<2: raise ValueError("Need >=2 paired targets")
    rng=np.random.default_rng(seed); n=len(diff); boot=np.empty(resamples)
    for i in range(resamples): boot[i]=statistic(diff[rng.integers(0,n,size=n)])
    return {"n":n,"effect":float(statistic(diff)),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975)),"bootstrap_p_two_sided":float(min(1.0,2*min((boot<=0).mean(),(boot>=0).mean())))}


def paired_wilcoxon(a: pd.Series,b: pd.Series)->dict[str,float]:
    aligned=pd.concat([a.rename("a"),b.rename("b")],axis=1).dropna(); result=stats.wilcoxon(aligned["a"],aligned["b"],alternative="two-sided",zero_method="wilcox")
    return {"n":int(len(aligned)),"statistic":float(result.statistic),"p":float(result.pvalue)}


def holm_adjust(pvalues:dict[str,float])->dict[str,float]:
    items=sorted(pvalues.items(),key=lambda kv:kv[1]); m=len(items); adjusted={}; running=0.0
    for rank,(name,p) in enumerate(items):
        value=min(1.0,(m-rank)*p); running=max(running,value); adjusted[name]=running
    return adjusted


def bh_adjust(pvalues:dict[str,float])->dict[str,float]:
    items=sorted(pvalues.items(),key=lambda kv:kv[1],reverse=True); m=len(items); adjusted={}; running=1.0
    for reverse_rank,(name,p) in enumerate(items,start=1):
        rank=m-reverse_rank+1; value=min(1.0,p*m/rank); running=min(running,value); adjusted[name]=running
    return adjusted


def partner_scramble_delta(native_nll:pd.Series,scrambled_nll:pd.Series)->pd.Series:
    aligned=pd.concat([native_nll.rename("native"),scrambled_nll.rename("scrambled")],axis=1).dropna(); return aligned["scrambled"]-aligned["native"]


def empirical_pmi(counts_20x4:np.ndarray,pseudocount:float=.5)->np.ndarray:
    counts=np.asarray(counts_20x4,float)
    if counts.shape!=(20,4): raise ValueError("Expected 20x4 counts")
    counts=counts+pseudocount; p=counts/counts.sum(); pa=p.sum(1,keepdims=True); pb=p.sum(0,keepdims=True); return np.log(p/(pa*pb))


def matrix_correlations(C:np.ndarray,pmi:np.ndarray)->dict[str,float]:
    c=np.asarray(C,float).ravel(); p=np.asarray(pmi,float).ravel(); pearson=stats.pearsonr(c,p); spearman=stats.spearmanr(c,p)
    return {"pearson_r":float(pearson.statistic),"pearson_p":float(pearson.pvalue),"spearman_rho":float(spearman.statistic),"spearman_p":float(spearman.pvalue)}


def effective_neighbor_number(alpha:np.ndarray)->float:
    a=np.asarray(alpha,float); a=a/a.sum(); entropy=-(a*np.log(np.clip(a,1e-12,1))).sum(); return float(np.exp(entropy))


def mandatory_test_registry()->pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in MANDATORY_TESTS])
