"""Robustness and interventional evaluation on frozen complex targets."""
from __future__ import annotations

from dataclasses import replace
import copy
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F

from pr_pilot.evaluation.battery import expected_calibration_error, native_probability_brier
from pr_pilot.evaluation.runner import score_conditional
from pr_pilot.model.dmicf import JointPriorAndFieldModel, PRBatch
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample


def subset_pr(pr: PRBatch, keep: Tensor) -> PRBatch:
    keep=keep.bool()
    return PRBatch(pr.protein_index[keep],pr.rna_index[keep],pr.edge_features[keep],pr.effective_distance[keep],None if pr.edge_batch is None else pr.edge_batch[keep])


def sample_with_pr(sample: ComplexTensorSample, pr: PRBatch) -> ComplexTensorSample:
    s=copy.copy(sample); s.pr=pr; return s


def deterministic_edge_keep(n:int,drop_fraction:float,seed:int)->Tensor:
    if not 0<=drop_fraction<1: raise ValueError("drop_fraction must be in [0,1)")
    rng=np.random.default_rng(seed); keep=np.ones(n,dtype=bool); n_drop=int(round(n*drop_fraction))
    if n_drop: keep[rng.choice(n,size=n_drop,replace=False)]=False
    if n>0 and not keep.any(): keep[rng.integers(0,n)]=True
    return torch.from_numpy(keep)


def evaluate_pr_edge_removal(model:JointPriorAndFieldModel,s:ComplexTensorSample,levels:list[float],seed:int)->pd.DataFrame:
    rows=[]; n=len(s.pr.protein_index)
    for level in levels:
        keep=deterministic_edge_keep(n,float(level),seed+int(level*10000)).to(s.pr.protein_index.device)
        ss=sample_with_pr(s,subset_pr(s.pr,keep))
        for direction,alphabet in [("protein",20),("rna",4)]:
            df=score_conditional(model,ss,direction,interface_only=True,seed=seed)
            rows.append({"sample_id":s.sample_id,"direction":direction,"edge_drop":float(level),"n_edges":int(keep.sum()),"nll":float(-df.native_log_probability.mean()),"recovery":float((df.native_token==df.predicted_token).mean())})
    return pd.DataFrame(rows)


def evaluate_partner_hiding(model:JointPriorAndFieldModel,s:ComplexTensorSample,levels:list[float],seed:int)->pd.DataFrame:
    rows=[]
    for level in levels:
        for direction in ["protein","rna"]:
            df=score_conditional(model,s,direction,interface_only=True,partner_hide=float(level),seed=seed+int(level*10000))
            rows.append({"sample_id":s.sample_id,"direction":direction,"partner_hide":float(level),"nll":float(-df.native_log_probability.mean()),"recovery":float((df.native_token==df.predicted_token).mean())})
    return pd.DataFrame(rows)


def evaluate_geometry_permutation(model:JointPriorAndFieldModel,s:ComplexTensorSample,repeats:int,seed:int)->pd.DataFrame:
    """Permute rich e_ij among existing PR edges while preserving topology and distance prior."""
    rows=[]; n=len(s.pr.protein_index)
    native={d:score_conditional(model,s,d,interface_only=True,seed=seed) for d in ["protein","rna"]}
    for k in range(repeats):
        rng=np.random.default_rng(seed+k); perm=torch.as_tensor(rng.permutation(n),device=s.pr.edge_features.device,dtype=torch.long)
        pr=PRBatch(s.pr.protein_index,s.pr.rna_index,s.pr.edge_features[perm],s.pr.effective_distance,s.pr.edge_batch)
        ss=sample_with_pr(s,pr)
        for direction in ["protein","rna"]:
            d=score_conditional(model,ss,direction,interface_only=True,seed=seed+k)
            rows.append({"sample_id":s.sample_id,"direction":direction,"repeat":k,"native_nll":float(-native[direction].native_log_probability.mean()),"permuted_geometry_nll":float(-d.native_log_probability.mean()),"delta_nll":float(-d.native_log_probability.mean()+native[direction].native_log_probability.mean())})
    return pd.DataFrame(rows)


def _target_local_nll(model:JointPriorAndFieldModel,s:ComplexTensorSample,direction:str,pos:int)->float:
    pt=s.protein.sequence.clone(); rt=s.rna.sequence.clone(); pk=s.protein.valid.clone().bool(); rk=s.rna.valid.clone().bool()
    if direction=="protein": pk[pos]=False
    else: rk[pos]=False
    out=model(s.protein.node_x,s.protein.edge_index,s.protein.edge_x,s.rna.node_x,s.rna.edge_index,s.rna.edge_x,s.pr,pt,rt,pk,rk)
    if direction=="protein":
        lp=F.log_softmax(out["protein_logits"][pos].float(),-1); native=int(s.protein.sequence[pos])
    else:
        lp=F.log_softmax(out["rna_logits"][pos].float(),-1); native=int(s.rna.sequence[pos])
    return float(-lp[native].cpu())


@torch.no_grad()
def alpha_edge_removal(model:JointPriorAndFieldModel,s:ComplexTensorSample,max_targets:int=32)->pd.DataFrame:
    """Intervene on edges instead of interpreting alpha as causal from a heat map.

    For each target with >=2 partner edges, compare deletion of the highest-alpha
    edge with a lower-alpha edge whose distance is closest to the top edge.
    """
    hp,hr=model.encode_backbones(s.protein.node_x,s.protein.edge_index,s.protein.edge_x,s.rna.node_x,s.rna.edge_index,s.rna.edge_x); fld=model.dmicf.field(hp,hr,s.pr); rows=[]
    for direction,groups,alpha in [("protein",s.pr.protein_index,fld["alpha_p"]),("rna",s.pr.rna_index,fld["alpha_r"])]:
        unique=torch.unique(groups)[:max_targets]
        for target in unique:
            idx=torch.where(groups==target)[0]
            if len(idx)<2: continue
            top=idx[torch.argmax(alpha[idx])]; candidates=idx[idx!=top]
            dist_top=s.pr.effective_distance[top]; matched=candidates[torch.argmin(torch.abs(s.pr.effective_distance[candidates]-dist_top))]
            base=_target_local_nll(model,s,direction,int(target))
            records=[]
            for label,edge in [("top_alpha",top),("distance_matched_other",matched)]:
                keep=torch.ones(len(groups),dtype=torch.bool,device=groups.device); keep[edge]=False; ss=sample_with_pr(s,subset_pr(s.pr,keep)); nll=_target_local_nll(model,ss,direction,int(target)); records.append((label,edge,nll))
            for label,edge,nll in records:
                rows.append({"sample_id":s.sample_id,"direction":direction,"target_position":int(target),"removal":label,"edge_index":int(edge),"alpha":float(alpha[edge].cpu()),"distance":float(s.pr.effective_distance[edge].cpu()),"baseline_nll":base,"removed_nll":nll,"delta_nll":nll-base})
    return pd.DataFrame(rows)


def calibration_table(predictions:pd.DataFrame,bins:int=15)->dict[str,float]:
    if len(predictions)==0: return {"n":0,"ece":float("nan"),"native_brier":float("nan")}
    correct=(predictions.native_token.to_numpy()==predictions.predicted_token.to_numpy()).astype(float); confidence=predictions.max_probability.to_numpy(float); native_prob=np.exp(predictions.native_log_probability.to_numpy(float))
    return {"n":int(len(predictions)),"ece":expected_calibration_error(correct,confidence,bins),"native_brier":native_probability_brier(native_prob),"mean_confidence":float(confidence.mean()),"accuracy":float(correct.mean())}
