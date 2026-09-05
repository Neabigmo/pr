"""One-command primary evaluation suite for the frozen 100-complex OOD holdout."""
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import torch

from pr_pilot.evaluation.battery import empirical_pmi, matrix_correlations, mandatory_test_registry
from pr_pilot.evaluation.runner import evaluate_holdout, load_model, _move, score_conditional
from pr_pilot.evaluation.robustness import alpha_edge_removal, calibration_table, evaluate_geometry_permutation, evaluate_partner_hiding, evaluate_pr_edge_removal
from pr_pilot.inference.sampler import sample_joint
from pr_pilot.model.dmicf import double_center
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row


def _adapter(cfg:dict,noise:float=0.0,seed_offset:int=0)->GemmiStructureAdapter:
    g=cfg["geometry"]
    return GemmiStructureAdapter(int(g["rbf_bins"]),int(g["intra_max_neighbors"]),float(g["pr_cutoff_angstrom"]),int(g["pr_max_neighbors"]),noise,int(cfg["experiment"]["pilot_seed"])+seed_offset,bool(g["rich_pr_geometry"]))


def _candidate_metrics(cands,sample)->pd.DataFrame:
    rows=[]
    native_p=sample.protein.sequence.cpu().numpy(); native_r=sample.rna.sequence.cpu().numpy()
    pairs=[]
    for c in cands:
        p=c.protein_tokens.cpu().numpy(); r=c.rna_tokens.cpu().numpy(); pre_p=c.pre_spir_protein.cpu().numpy(); pre_r=c.pre_spir_rna.cpu().numpy(); pairs.append((p.copy(),r.copy()))
        pi=sample.protein.interface.cpu().numpy(); ri=sample.rna.interface.cpu().numpy()
        rows.append({"sample_id":sample.sample_id,"candidate_id":c.candidate_id,"spir_cycles":c.spir_cycles,"spir_direction":c.spir_direction,"protein_recovery":float((p==native_p).mean()),"rna_recovery":float((r==native_r).mean()),"protein_interface_recovery":float((p[pi]==native_p[pi]).mean()) if pi.any() else np.nan,"rna_interface_recovery":float((r[ri]==native_r[ri]).mean()) if ri.any() else np.nan,"pre_to_post_protein_change":float((p!=pre_p).mean()),"pre_to_post_rna_change":float((r!=pre_r).mean())})
    # Diversity is a target-level property; repeat it for convenient grouping.
    if len(pairs)>1:
        pdist=[]; rdist=[]
        for i in range(len(pairs)):
            for j in range(i+1,len(pairs)):
                pdist.append(float((pairs[i][0]!=pairs[j][0]).mean())); rdist.append(float((pairs[i][1]!=pairs[j][1]).mean()))
        pdiv=float(np.mean(pdist)); rdiv=float(np.mean(rdist))
    else: pdiv=rdiv=0.0
    uniq=len({(tuple(p),tuple(r)) for p,r in pairs})/max(1,len(pairs))
    for row in rows: row.update({"protein_pairwise_diversity":pdiv,"rna_pairwise_diversity":rdiv,"unique_pair_fraction":uniq})
    return pd.DataFrame(rows)


def _permutation_null(C_interaction:np.ndarray,edge_table:pd.DataFrame,repeats:int,seed:int)->dict:
    observed_counts=np.zeros((20,4),dtype=int)
    for a,b in zip(edge_table.aa.astype(int),edge_table.base.astype(int)): observed_counts[a,b]+=1
    observed=matrix_correlations(C_interaction,empirical_pmi(observed_counts))["spearman_rho"]
    rng=np.random.default_rng(seed); vals=[]
    # Preserve AA/base marginals globally. Stratum-specific extensions can be added
    # when contact-surface labels are available in the manifest.
    aa=edge_table.aa.to_numpy(int); base=edge_table.base.to_numpy(int)
    for _ in range(repeats):
        bp=base.copy(); rng.shuffle(bp); counts=np.zeros((20,4),dtype=int)
        for a,b in zip(aa,bp): counts[a,b]+=1
        vals.append(matrix_correlations(C_interaction,empirical_pmi(counts))["spearman_rho"])
    vals=np.asarray(vals); p=(1+np.sum(np.abs(vals)>=abs(observed)))/(len(vals)+1)
    return {"observed_spearman":float(observed),"null_mean":float(vals.mean()),"null_sd":float(vals.std(ddof=1)),"empirical_two_sided_p":float(p),"repeats":int(repeats)}


def _numeric_shift(dev:pd.DataFrame,test:pd.DataFrame,column:str)->dict|None:
    if column not in dev or column not in test: return None
    a=pd.to_numeric(dev[column],errors="coerce").dropna().to_numpy(float); b=pd.to_numeric(test[column],errors="coerce").dropna().to_numpy(float)
    if len(a)<2 or len(b)<2: return None
    pooled=math.sqrt((a.var(ddof=1)+b.var(ddof=1))/2.0) if a.var(ddof=1)+b.var(ddof=1)>0 else 0.0
    return {"dev_mean":float(a.mean()),"test_mean":float(b.mean()),"standardized_mean_difference":float((b.mean()-a.mean())/pooled) if pooled else 0.0,"dev_n":len(a),"test_n":len(b)}


def dataset_shift_audit(dev_path:Path,test_path:Path)->dict:
    dev=pd.read_csv(dev_path,sep="\t"); test=pd.read_csv(test_path,sep="\t"); out={}
    for col in ["protein_length","rna_length","total_tokens","resolution","interface_protein_residues","interface_rna_nucleotides","pr_edge_count"]:
        x=_numeric_shift(dev,test,col)
        if x is not None: out[col]=x
    for col in ["experimental_method","rna_type","origin","source"]:
        if col in dev and col in test:
            cats=sorted(set(dev[col].astype(str))|set(test[col].astype(str))); p=np.array([(dev[col].astype(str)==c).mean() for c in cats])+1e-8; q=np.array([(test[col].astype(str)==c).mean() for c in cats])+1e-8; p/=p.sum(); q/=q.sum(); m=(p+q)/2; js=.5*np.sum(p*np.log(p/m))+.5*np.sum(q*np.log(q/m)); out[col]={"jensen_shannon_divergence_nats":float(js),"categories":cats}
    return out


@torch.no_grad()
def run_full_suite(cfg:dict,checkpoint:Path,test_manifest:Path,out_dir:Path,dev_manifest:Path|None=None,device:str|None=None)->dict:
    out_dir.mkdir(parents=True,exist_ok=True); mandatory_test_registry().to_csv(out_dir/"test_registry.tsv",sep="\t",index=False)
    # Core tables first.
    core_dir=out_dir/"core"; summary=evaluate_holdout(cfg,checkpoint,test_manifest,core_dir,device,"DMICF")
    dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); model=load_model(checkpoint,cfg,dev); table=ManifestTable(test_manifest); base_adapter=_adapter(cfg); seed=int(cfg["experiment"]["pilot_seed"]); ecfg=cfg["evaluation"]
    edge=[]; hide=[]; geom=[]; alpha=[]; noise=[]; spir=[]
    noise_levels=[float(x) for x in ecfg["noise_levels_angstrom"]]
    for row_idx,row in enumerate(table.rows()):
        s=_move(load_complex_row(base_adapter,row),dev)
        edge.append(evaluate_pr_edge_removal(model,s,[float(x) for x in ecfg["pr_edge_drop_levels"]],seed+row_idx))
        hide.append(evaluate_partner_hiding(model,s,[float(x) for x in ecfg["partner_hide_levels"]],seed+row_idx))
        geom.append(evaluate_geometry_permutation(model,s,int(ecfg.get("geometry_permutation_repeats",10)),seed+row_idx))
        alpha.append(alpha_edge_removal(model,s,int(ecfg.get("alpha_ablation_max_targets",32))))
        for sigma in noise_levels:
            ns=_move(load_complex_row(_adapter(cfg,sigma,10000+row_idx),row),dev)
            for direction in ["protein","rna"]:
                df=score_conditional(model,ns,direction,interface_only=True,seed=seed+row_idx)
                noise.append({"sample_id":s.sample_id,"direction":direction,"noise_angstrom":sigma,"nll":float(-df.native_log_probability.mean()),"recovery":float((df.native_token==df.predicted_token).mean())})
        icfg=cfg["inference"]; base_spir=icfg["spir"]; n_cand=int(icfg["candidates_per_complex"])
        for cycles in [0,1,int(ecfg.get("repeated_spir_cycles",3))]:
            cands=sample_joint(model,s,n_cand,float(icfg["initial_temperature"]),seed+row_idx,spir_enabled=cycles>0,spir_reopen_fraction=float(base_spir["reopen_fraction"]),spir_temperature=float(base_spir["temperature"]),spir_cycles=cycles,reverse_direction_fraction=float(base_spir["reverse_direction_fraction"]))
            spir.append(_candidate_metrics(cands,s))
    tables={"edge_removal":pd.concat(edge,ignore_index=True),"partner_hiding":pd.concat(hide,ignore_index=True),"geometry_permutation":pd.concat(geom,ignore_index=True),"alpha_edge_removal":pd.concat(alpha,ignore_index=True) if any(len(x) for x in alpha) else pd.DataFrame(),"coordinate_noise":pd.DataFrame(noise),"spir_candidates":pd.concat(spir,ignore_index=True)}
    for name,df in tables.items(): df.to_csv(out_dir/f"{name}.tsv",sep="\t",index=False)
    cp=pd.read_csv(core_dir/"conditional_protein.tsv",sep="\t"); cr=pd.read_csv(core_dir/"conditional_rna.tsv",sep="\t")
    calibration={"protein":calibration_table(cp),"rna":calibration_table(cr)}; (out_dir/"calibration.json").write_text(json.dumps(calibration,indent=2),encoding="utf-8")
    edge_table=pd.read_csv(core_dir/"field_edges.tsv",sep="\t"); C=np.load(core_dir/"C_interaction_only.npy"); perm=_permutation_null(C,edge_table,int(ecfg.get("pmi_permutation_repeats",1000)),seed); (out_dir/"C_PMI_permutation_null.json").write_text(json.dumps(perm,indent=2),encoding="utf-8")
    if dev_manifest is not None:
        shift=dataset_shift_audit(dev_manifest,test_manifest); (out_dir/"strict_ood_shift.json").write_text(json.dumps(shift,indent=2),encoding="utf-8")
    final={"core":summary,"calibration":calibration,"C_PMI_permutation":perm,"status":"primary single-checkpoint battery complete; cross-model/seed statistics require compare_runs.py"}
    (out_dir/"suite_summary.json").write_text(json.dumps(final,indent=2,sort_keys=True),encoding="utf-8"); return final
