"""Executable held-out evaluation for the frozen 100 complexes."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F

from pr_pilot.evaluation.battery import empirical_pmi, matrix_correlations, token_metrics
from pr_pilot.model.dmicf import JointPriorAndFieldModel, PRBatch, double_center
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample, PolymerGraph
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row
from pr_pilot.training.engine import build_model_from_config


def _move(s: ComplexTensorSample, device: torch.device) -> ComplexTensorSample:
    for g in [s.protein,s.rna]:
        for name in ["node_x","edge_index","edge_x","sequence","interface","valid","fixed","reference_xyz","chain_index"]:
            setattr(g,name,getattr(g,name).to(device))
    s.pr=PRBatch(s.pr.protein_index.to(device),s.pr.rna_index.to(device),s.pr.edge_features.to(device),s.pr.effective_distance.to(device))
    return s


def _adapter(cfg: dict, noise: float = 0.0, rich: bool | None = None, seed_offset: int = 0) -> GemmiStructureAdapter:
    g=cfg["geometry"]
    return GemmiStructureAdapter(int(g["rbf_bins"]),int(g["intra_max_neighbors"]),float(g["pr_cutoff_angstrom"]),int(g["pr_max_neighbors"]),noise,int(cfg["experiment"]["pilot_seed"])+seed_offset,bool(g["rich_pr_geometry"] if rich is None else rich))


def load_model(checkpoint: Path, cfg: dict, device: torch.device) -> JointPriorAndFieldModel:
    model=build_model_from_config(cfg).to(device)
    payload=torch.load(checkpoint,map_location="cpu")
    model.load_state_dict(payload["model"]); model.eval(); return model


def _forward(model: JointPriorAndFieldModel,s: ComplexTensorSample,pt:Tensor,rt:Tensor,pk:Tensor,rk:Tensor)->dict[str,Tensor]:
    return model(s.protein.node_x,s.protein.edge_index,s.protein.edge_x,s.rna.node_x,s.rna.edge_index,s.rna.edge_x,s.pr,pt,rt,pk,rk)


def _rows_from_logits(sample_id:str,polymer:str,logits:Tensor,native:Tensor,mask:Tensor,interface:Tensor,model_name:str,seed:int)->list[dict]:
    logp=F.log_softmax(logits.float(),-1); prob=logp.exp(); pred=logits.argmax(-1)
    rows=[]
    for idx in torch.where(mask)[0]:
        i=int(idx); n=int(native[i]); p=int(pred[i])
        rows.append({"sample_id":sample_id,"polymer":polymer,"position":i,"native_token":n,"predicted_token":p,"native_log_probability":float(logp[i,n].cpu()),"max_probability":float(prob[i].max().cpu()),"is_interface":bool(interface[i]),"model":model_name,"seed":seed})
    return rows


@torch.no_grad()
def score_conditional(model:JointPriorAndFieldModel,s:ComplexTensorSample,target:str,interface_only:bool=False,partner_hide:float=0.0,seed:int=0,model_name:str="DMICF")->pd.DataFrame:
    rng=np.random.default_rng(seed)
    pt=s.protein.sequence.clone(); rt=s.rna.sequence.clone()
    if target=="protein":
        mask=(s.protein.interface if interface_only else s.protein.valid) & s.protein.valid & ~s.protein.fixed
        pk=s.protein.valid.clone().bool(); pk[mask]=False; rk=s.rna.valid.clone().bool()
        if partner_hide>0:
            cand=torch.where(rk & ~s.rna.fixed)[0].cpu().numpy(); n=int(round(len(cand)*partner_hide))
            if n: rk[torch.tensor(rng.choice(cand,n,replace=False),device=rk.device)]=False
        out=_forward(model,s,pt,rt,pk,rk)
        return pd.DataFrame(_rows_from_logits(s.sample_id,"protein",out["protein_logits"],s.protein.sequence,mask,s.protein.interface,model_name,seed))
    if target=="rna":
        mask=(s.rna.interface if interface_only else s.rna.valid) & s.rna.valid & ~s.rna.fixed
        rk=s.rna.valid.clone().bool(); rk[mask]=False; pk=s.protein.valid.clone().bool()
        if partner_hide>0:
            cand=torch.where(pk & ~s.protein.fixed)[0].cpu().numpy(); n=int(round(len(cand)*partner_hide))
            if n: pk[torch.tensor(rng.choice(cand,n,replace=False),device=pk.device)]=False
        out=_forward(model,s,pt,rt,pk,rk)
        return pd.DataFrame(_rows_from_logits(s.sample_id,"rna",out["rna_logits"],s.rna.sequence,mask,s.rna.interface,model_name,seed))
    raise ValueError(target)


@torch.no_grad()
def score_joint_teacher_forced(model:JointPriorAndFieldModel,s:ComplexTensorSample,orders:int=5,seed:int=0,model_name:str="DMICF")->pd.DataFrame:
    """Random mixed-order native log-likelihood without exposing the current target token."""
    rows=[]; design=[("protein",int(i)) for i in torch.where(s.protein.valid & ~s.protein.fixed)[0]]+[("rna",int(i)) for i in torch.where(s.rna.valid & ~s.rna.fixed)[0]]
    for order_idx in range(orders):
        rng=random.Random(seed+104729*order_idx); order=design.copy(); rng.shuffle(order)
        pk=s.protein.fixed & s.protein.valid; rk=s.rna.fixed & s.rna.valid
        pt=s.protein.sequence.clone(); rt=s.rna.sequence.clone()
        for polymer,i in order:
            out=_forward(model,s,pt,rt,pk,rk)
            logits=out["protein_logits"][i] if polymer=="protein" else out["rna_logits"][i]
            lp=F.log_softmax(logits.float(),-1); native=int(pt[i] if polymer=="protein" else rt[i]); pred=int(logits.argmax())
            inter=bool(s.protein.interface[i] if polymer=="protein" else s.rna.interface[i])
            rows.append({"sample_id":s.sample_id,"polymer":polymer,"position":i,"order":order_idx,"native_token":native,"predicted_token":pred,"native_log_probability":float(lp[native].cpu()),"max_probability":float(lp.exp().max().cpu()),"is_interface":inter,"model":model_name,"seed":seed})
            if polymer=="protein": pk[i]=True
            else: rk[i]=True
    return pd.DataFrame(rows)


def _permute_tensor(x:Tensor,valid:Tensor,seed:int)->Tensor:
    out=x.clone(); idx=torch.where(valid)[0].cpu().numpy(); rng=np.random.default_rng(seed); perm=idx.copy(); rng.shuffle(perm); out[torch.tensor(idx,device=x.device)]=x[torch.tensor(perm,device=x.device)]; return out


@torch.no_grad()
def partner_scramble(model:JointPriorAndFieldModel,s:ComplexTensorSample,repeats:int=20,seed:int=0)->pd.DataFrame:
    rows=[]
    for direction in ["protein","rna"]:
        native=score_conditional(model,s,direction,interface_only=True,seed=seed)
        native_nll=-native.native_log_probability.mean()
        for k in range(repeats):
            pt=s.protein.sequence.clone(); rt=s.rna.sequence.clone()
            if direction=="protein": rt=_permute_tensor(rt,s.rna.valid,seed+1000+k)
            else: pt=_permute_tensor(pt,s.protein.valid,seed+2000+k)
            if direction=="protein":
                mask=s.protein.interface & s.protein.valid & ~s.protein.fixed; pk=s.protein.valid.clone(); pk[mask]=False; rk=s.rna.valid.clone()
                out=_forward(model,s,pt,rt,pk,rk); logp=F.log_softmax(out["protein_logits"].float(),-1); nll=float(-logp[mask,s.protein.sequence[mask]].mean().cpu())
            else:
                mask=s.rna.interface & s.rna.valid & ~s.rna.fixed; rk=s.rna.valid.clone(); rk[mask]=False; pk=s.protein.valid.clone()
                out=_forward(model,s,pt,rt,pk,rk); logp=F.log_softmax(out["rna_logits"].float(),-1); nll=float(-logp[mask,s.rna.sequence[mask]].mean().cpu())
            rows.append({"sample_id":s.sample_id,"direction":direction,"repeat":k,"native_nll":float(native_nll),"scrambled_nll":nll,"delta_nll":nll-float(native_nll)})
    return pd.DataFrame(rows)


@torch.no_grad()
def counterfactual_partner_mutation(model:JointPriorAndFieldModel,s:ComplexTensorSample,seed:int=0,max_partner_sites:int=16)->pd.DataFrame:
    """Symmetric single-token counterfactuals; target token itself remains masked."""
    rng=np.random.default_rng(seed); rows=[]
    # RNA mutation -> protein distribution response (all 3 alternatives).
    rsites=torch.unique(s.pr.rna_index).cpu().numpy(); rng.shuffle(rsites); rsites=rsites[:max_partner_sites]
    base_pk=s.protein.valid.clone(); base_pk[s.protein.interface & ~s.protein.fixed]=False; rk=s.rna.valid.clone()
    base=_forward(model,s,s.protein.sequence,s.rna.sequence,base_pk,rk); p0=F.softmax(base["protein_logits"].float(),-1)
    for j in rsites:
        native=int(s.rna.sequence[j])
        for alt in [x for x in range(4) if x!=native]:
            rt=s.rna.sequence.clone(); rt[j]=alt; out=_forward(model,s,s.protein.sequence,rt,base_pk,rk); p1=F.softmax(out["protein_logits"].float(),-1)
            kl=(p0*(p0.clamp_min(1e-12).log()-p1.clamp_min(1e-12).log())).sum(-1)
            for i in torch.where(s.protein.interface)[0]:
                edge=(s.pr.protein_index==i)&(s.pr.rna_index==int(j)); dist=float(s.pr.effective_distance[edge].min().cpu()) if edge.any() else float(torch.linalg.vector_norm(s.protein.reference_xyz[i]-s.rna.reference_xyz[j]).cpu())
                rows.append({"sample_id":s.sample_id,"mutated_polymer":"rna","partner_position":int(j),"native":native,"alternative":alt,"responding_polymer":"protein","position":int(i),"distance":dist,"kl":float(kl[i].cpu())})
    # Protein mutation -> RNA response. Sample four alternatives per site for cost control.
    psites=torch.unique(s.pr.protein_index).cpu().numpy(); rng.shuffle(psites); psites=psites[:max_partner_sites]
    pk=s.protein.valid.clone(); base_rk=s.rna.valid.clone(); base_rk[s.rna.interface & ~s.rna.fixed]=False
    base=_forward(model,s,s.protein.sequence,s.rna.sequence,pk,base_rk); r0=F.softmax(base["rna_logits"].float(),-1)
    for i in psites:
        native=int(s.protein.sequence[i]); alts=[x for x in range(20) if x!=native]; rng.shuffle(alts)
        for alt in alts[:4]:
            pt=s.protein.sequence.clone(); pt[i]=alt; out=_forward(model,s,pt,s.rna.sequence,pk,base_rk); r1=F.softmax(out["rna_logits"].float(),-1)
            kl=(r0*(r0.clamp_min(1e-12).log()-r1.clamp_min(1e-12).log())).sum(-1)
            for j in torch.where(s.rna.interface)[0]:
                edge=(s.pr.protein_index==int(i))&(s.pr.rna_index==j); dist=float(s.pr.effective_distance[edge].min().cpu()) if edge.any() else float(torch.linalg.vector_norm(s.protein.reference_xyz[i]-s.rna.reference_xyz[j]).cpu())
                rows.append({"sample_id":s.sample_id,"mutated_polymer":"protein","partner_position":int(i),"native":native,"alternative":alt,"responding_polymer":"rna","position":int(j),"distance":dist,"kl":float(kl[j].cpu())})
    return pd.DataFrame(rows)


@torch.no_grad()
def field_tables(model:JointPriorAndFieldModel,s:ComplexTensorSample)->tuple[pd.DataFrame,np.ndarray]:
    hp,hr=model.encode_backbones(s.protein.node_x,s.protein.edge_index,s.protein.edge_x,s.rna.node_x,s.rna.edge_index,s.rna.edge_x); fld=model.dmicf.field(hp,hr,s.pr)
    cnorm=float(torch.linalg.vector_norm(fld["C"]).cpu()); rows=[]; counts=np.zeros((20,4),dtype=np.int64)
    for e in range(len(s.pr.protein_index)):
        i=int(s.pr.protein_index[e]); j=int(s.pr.rna_index[e]); a=int(s.protein.sequence[i]); b=int(s.rna.sequence[j]); d=float(s.pr.effective_distance[e].cpu())
        if d<=5.0: counts[a,b]+=1
        rows.append({"sample_id":s.sample_id,"edge":e,"protein_position":i,"rna_position":j,"distance":d,"aa":a,"base":b,"delta_fro":float(torch.linalg.vector_norm(fld["DeltaC"][e]).cpu()),"delta_over_c":float(torch.linalg.vector_norm(fld["DeltaC"][e]).cpu()/(cnorm+1e-12)),"alpha_p":float(fld["alpha_p"][e].cpu()),"alpha_r":float(fld["alpha_r"][e].cpu()),"score":float(fld["scores"][e].cpu())})
    return pd.DataFrame(rows),counts


@torch.no_grad()
def evaluate_holdout(cfg:dict,checkpoint:Path,manifest_path:Path,out_dir:Path,device:str|None=None,model_name:str="DMICF")->dict:
    dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); model=load_model(checkpoint,cfg,dev); adapter=_adapter(cfg); table=ManifestTable(manifest_path); out_dir.mkdir(parents=True,exist_ok=True)
    cond_p=[]; cond_r=[]; joint=[]; scr=[]; cf=[]; fields=[]; counts=np.zeros((20,4),dtype=np.int64)
    seed=int(cfg["experiment"]["pilot_seed"])
    for row in table.rows():
        s=_move(load_complex_row(adapter,row),dev)
        cond_p.append(score_conditional(model,s,"protein",False,seed=seed,model_name=model_name)); cond_r.append(score_conditional(model,s,"rna",False,seed=seed,model_name=model_name)); joint.append(score_joint_teacher_forced(model,s,int(cfg["evaluation"].get("joint_teacher_forced_orders",5)),seed,model_name)); scr.append(partner_scramble(model,s,int(cfg["evaluation"].get("scramble_repeats",20)),seed)); cf.append(counterfactual_partner_mutation(model,s,seed,int(cfg["evaluation"].get("counterfactual_max_partner_sites",16)))); ft,c=field_tables(model,s); fields.append(ft); counts+=c
    outputs={"conditional_protein":pd.concat(cond_p,ignore_index=True),"conditional_rna":pd.concat(cond_r,ignore_index=True),"joint_teacher_forced":pd.concat(joint,ignore_index=True),"partner_scramble":pd.concat(scr,ignore_index=True),"counterfactual":pd.concat(cf,ignore_index=True),"field_edges":pd.concat(fields,ignore_index=True)}
    for name,df in outputs.items(): df.to_csv(out_dir/f"{name}.tsv",sep="\t",index=False)
    C=model.dmicf.global_c().detach().cpu().numpy(); pmi=empirical_pmi(counts); np.save(out_dir/"C.npy",C); np.save(out_dir/"C_interaction_only.npy",double_center(torch.from_numpy(C)).numpy()); np.save(out_dir/"heldout_pmi.npy",pmi); np.save(out_dir/"heldout_counts_20x4.npy",counts)
    summary={"protein":token_metrics(outputs["conditional_protein"],20),"rna":token_metrics(outputs["conditional_rna"],4),"joint_protein":token_metrics(outputs["joint_teacher_forced"].query("polymer=='protein'"),20),"joint_rna":token_metrics(outputs["joint_teacher_forced"].query("polymer=='rna'"),4),"C_vs_PMI_interaction_only":matrix_correlations(double_center(torch.from_numpy(C)).numpy(),pmi)}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8"); return summary
