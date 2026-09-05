#!/usr/bin/env python3
"""Run two interpretable controls around the global 20x4 compatibility matrix.

A. C-backbone-only-context control
   Target-chain *all* designable tokens are hidden while the loss is evaluated on
   interface positions. This tests whether the primary C stage depends strongly on
   known same-chain sequence context outside the interface.

B. Fixed empirical-PMI control
   Build a 20x4 amino-acid/base PMI from experimental complex-development heavy-
   atom contacts only, place it directly in C, and combine it with the full-1000
   dual structural priors. No final-test contact contributes to this matrix and the
   primary DM-ICF model is still randomly initialized; this is a separate classical
   statistical-potential reference.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from pr_pilot.evaluation.empirical_contacts import empirical_contact_tables
from pr_pilot.evaluation.runner import partner_scramble, score_conditional, score_joint_teacher_forced
from pr_pilot.model.dmicf import global_center
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row
from pr_pilot.training.engine import _adapter, _autocast, _cosine_schedule, build_model_from_config
from pr_pilot.training.losses import balanced_sequence_loss
from pr_pilot.training.refit import selected_epoch_count
from pr_pilot.training.stages import Stage, build_optimizer, configure_stage


def _move(sample, device):
    for graph in [sample.protein, sample.rna]:
        for name in ["node_x", "edge_index", "edge_x", "sequence", "interface", "valid", "fixed", "reference_xyz", "chain_index"]:
            setattr(graph, name, getattr(graph, name).to(device))
    sample.pr.protein_index = sample.pr.protein_index.to(device)
    sample.pr.rna_index = sample.pr.rna_index.to(device)
    sample.pr.edge_features = sample.pr.edge_features.to(device)
    sample.pr.effective_distance = sample.pr.effective_distance.to(device)
    return sample


def _optimizer(model, cfg):
    opt = cfg["optimization"]
    return build_optimizer(
        model, Stage.GLOBAL_C,
        float(opt["lr_heads"]), float(opt["lr_projections"]),
        float(opt["lr_encoder_top"]), float(opt["lr_encoder_bottom"]),
        float(opt["lr_global_c_joint"]), float(opt["weight_decay"]),
        float(opt["layerwise_lr_decay"]),
    )


def _c_control_loss(model, sample, protein_target: bool):
    if protein_target:
        pt = sample.protein.sequence.clone()
        pk = sample.protein.fixed & sample.protein.valid
        rt = sample.rna.sequence.clone()
        rk = sample.rna.valid.clone().bool()
        mask = sample.protein.interface & sample.protein.valid & ~sample.protein.fixed
    else:
        pt = sample.protein.sequence.clone()
        pk = sample.protein.valid.clone().bool()
        rt = sample.rna.sequence.clone()
        rk = sample.rna.fixed & sample.rna.valid
        mask = sample.rna.interface & sample.rna.valid & ~sample.rna.fixed
    out = model(
        sample.protein.node_x, sample.protein.edge_index, sample.protein.edge_x,
        sample.rna.node_x, sample.rna.edge_index, sample.rna.edge_x, sample.pr,
        pt, rt, pk, rk, use_delta=False, learned_alpha=False,
    )
    if protein_target:
        return balanced_sequence_loss(
            out["protein_logits"], sample.protein.sequence, mask, sample.protein.interface,
            None, None, None, None, "protein", 0.0, 0.0,
        ).total
    return balanced_sequence_loss(
        None, None, None, None,
        out["rna_logits"], sample.rna.sequence, mask, sample.rna.interface,
        "rna", 0.0, 0.0,
    ).total


@torch.no_grad()
def _validate_c_control(model, manifest, cfg, device):
    model.eval()
    adapter = _adapter(cfg, 0, training=False)
    values = []
    for row in manifest.rows():
        sample = _move(load_complex_row(adapter, row), device)
        values.append(float(_c_control_loss(model, sample, True)))
        values.append(float(_c_control_loss(model, sample, False)))
    return float(np.mean(values))


def train_c_backbone_only(
    cfg: dict,
    init_checkpoint: Path,
    train_manifest: Path,
    val_manifest: Path,
    out_dir: Path,
    device: str | None,
) -> Path:
    seed = int(cfg["experiment"]["pilot_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(dev)
    model.load_state_dict(torch.load(init_checkpoint, map_location="cpu")["model"])
    configure_stage(model, Stage.GLOBAL_C)
    optimizer = _optimizer(model, cfg)
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    train = ManifestTable(train_manifest); val = ManifestTable(val_manifest)
    max_epochs = int(cfg["training_stages"]["global_c"]["max_epochs"])
    patience = int(cfg["optimization"]["early_stopping_patience"])
    total_steps = max_epochs * len(train)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"; best = float("inf"); bad = 0; step = 0
    for epoch in range(max_epochs):
        model.train(); adapter = _adapter(cfg, epoch, training=True)
        rows = list(train.rows()); random.Random(seed + epoch).shuffle(rows)
        for idx, row in enumerate(rows):
            sample = _move(load_complex_row(adapter, row), dev)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, dev):
                loss = _c_control_loss(model, sample, protein_target=((idx + epoch) % 2 == 0))
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg["optimization"]["grad_clip_norm"]))
            optimizer.step(); step += 1
            _cosine_schedule(optimizer, step, total_steps, float(cfg["optimization"]["warmup_fraction"]), base_lrs)
        score = _validate_c_control(model, val, cfg, dev)
        if score < best - 1e-6:
            best = score; bad = 0
            torch.save({"model": model.state_dict(), "stage": "global_c_backbone_only", "epoch": epoch, "val_metric": score, "config": cfg}, best_path)
        else:
            bad += 1
            if bad >= patience: break
    return best_path


def refit_c_backbone_only(cfg, init_checkpoint, selected_dev_checkpoint, full_manifest, out_dir, device=None):
    epochs = selected_epoch_count(selected_dev_checkpoint)
    seed = int(cfg["experiment"]["pilot_seed"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(dev)
    model.load_state_dict(torch.load(init_checkpoint, map_location="cpu")["model"])
    configure_stage(model, Stage.GLOBAL_C)
    optimizer = _optimizer(model, cfg); base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    table = ManifestTable(full_manifest); total_steps = epochs * len(table); step = 0
    for epoch in range(epochs):
        model.train(); adapter = _adapter(cfg, epoch, training=True)
        rows = list(table.rows()); random.Random(seed + epoch).shuffle(rows)
        for idx, row in enumerate(rows):
            sample = _move(load_complex_row(adapter, row), dev)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, dev): loss = _c_control_loss(model, sample, ((idx + epoch) % 2 == 0))
            loss.backward(); optimizer.step(); step += 1
            _cosine_schedule(optimizer, step, total_steps, float(cfg["optimization"]["warmup_fraction"]), base_lrs)
    out_dir.mkdir(parents=True, exist_ok=True); path = out_dir / "refit.pt"
    torch.save({"model": model.state_dict(), "stage": "global_c_backbone_only", "epoch": epochs - 1, "refit": True, "config": cfg}, path)
    return path


def _pmi_from_dev(dev_manifest: Path, cutoff: float = 5.0, pseudocount: float = 0.5):
    _, counts = empirical_contact_tables(dev_manifest, cutoff=cutoff)
    matrix = counts["any"].astype(np.float64) + pseudocount
    joint = matrix / matrix.sum(); pa = joint.sum(1, keepdims=True); pb = joint.sum(0, keepdims=True)
    pmi = np.log(joint / (pa @ pb))
    return pmi.astype(np.float32), counts["any"]


def fixed_pmi_checkpoint(cfg, prior_checkpoint: Path, dev_manifest: Path, out_dir: Path):
    model = build_model_from_config(cfg)
    model.load_state_dict(torch.load(prior_checkpoint, map_location="cpu")["model"])
    pmi, counts = _pmi_from_dev(dev_manifest)
    with torch.no_grad(): model.dmicf.global_c.raw.copy_(torch.from_numpy(pmi))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "development_empirical_PMI.npy", pmi); np.save(out_dir / "development_contact_counts.npy", counts)
    path = out_dir / "fixed_pmi.pt"
    torch.save({"model": model.state_dict(), "stage": "fixed_empirical_pmi", "pmi_source": str(dev_manifest), "config": cfg}, path)
    return path


def evaluate_simple(cfg, checkpoint, test_manifest, out_dir, model_name, device=None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(dev); model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"]); model.eval()
    g = cfg["geometry"]; adapter = GemmiStructureAdapter(int(g["rbf_bins"]), int(g["intra_max_neighbors"]), float(g["pr_cutoff_angstrom"]), int(g["pr_max_neighbors"]), 0.0, int(cfg["experiment"]["pilot_seed"]), bool(g["rich_pr_geometry"]))
    p, r, j, s = [], [], [], []
    for row in ManifestTable(test_manifest).rows():
        sample = _move(load_complex_row(adapter, row), dev)
        p.append(score_conditional(model, sample, "protein", False, seed=int(cfg["experiment"]["pilot_seed"]), model_name=model_name))
        r.append(score_conditional(model, sample, "rna", False, seed=int(cfg["experiment"]["pilot_seed"]), model_name=model_name))
        j.append(score_joint_teacher_forced(model, sample, int(cfg["evaluation"].get("joint_teacher_forced_orders", 5)), int(cfg["experiment"]["pilot_seed"]), model_name))
        s.append(partner_scramble(model, sample, int(cfg["evaluation"].get("scramble_repeats", 20)), int(cfg["experiment"]["pilot_seed"])))
    core = out_dir / "core"; core.mkdir(parents=True, exist_ok=True)
    for name, frames in [("conditional_protein",p),("conditional_rna",r),("joint_teacher_forced",j),("partner_scramble",s)]: pd.concat(frames, ignore_index=True).to_csv(core / f"{name}.tsv", sep="\t", index=False)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=Path("configs/pilot.yaml")); parser.add_argument("--manifest-root",type=Path,default=Path("manifests/pilot_v1")); parser.add_argument("--primary-root",type=Path,default=Path("artifacts/pilot_experiments/training")); parser.add_argument("--out",type=Path,default=Path("artifacts/statistical_controls")); parser.add_argument("--device"); parser.add_argument("--seeds",type=int,nargs="+"); args=parser.parse_args()
    base=yaml.safe_load(args.config.read_text(encoding="utf-8")); seeds=args.seeds or [int(x) for x in base["experiment"]["primary_training_seeds"]]
    run_rows=[]
    for seed in seeds:
        cfg=json.loads(json.dumps(base)); cfg["experiment"]["pilot_seed"]=seed
        dev_prior=args.primary_root/"primary_development"/f"seed{seed}"/"rna_prior"/"best.pt"; refit_prior=args.primary_root/"primary_refit_full1000"/f"seed{seed}"/"rna_prior"/"refit.pt"
        c_dev=train_c_backbone_only(cfg,dev_prior,args.manifest_root/"complex_train.tsv",args.manifest_root/"complex_val.tsv",args.out/"training"/"c_backbone_only"/f"seed{seed}"/"development",args.device)
        c_refit=refit_c_backbone_only(cfg,refit_prior,c_dev,args.manifest_root/"complex_dev.tsv",args.out/"training"/"c_backbone_only"/f"seed{seed}"/"refit_full1000",args.device)
        c_eval=args.out/"evaluation"/"c_backbone_only"/f"seed{seed}"; evaluate_simple(cfg,c_refit,args.manifest_root/"complex_test.tsv",c_eval,"C_backbone_only",args.device); run_rows.append({"model":"C_backbone_only","seed":seed,"run_dir":str(c_eval.resolve())})
        pmi_ckpt=fixed_pmi_checkpoint(cfg,refit_prior,args.manifest_root/"complex_dev.tsv",args.out/"training"/"fixed_PMI"/f"seed{seed}")
        pmi_eval=args.out/"evaluation"/"fixed_PMI"/f"seed{seed}"; evaluate_simple(cfg,pmi_ckpt,args.manifest_root/"complex_test.tsv",pmi_eval,"fixed_empirical_PMI",args.device); run_rows.append({"model":"fixed_empirical_PMI","seed":seed,"run_dir":str(pmi_eval.resolve())})
    pd.DataFrame(run_rows).to_csv(args.out/"statistical_control_runs.tsv",sep="\t",index=False)
    print(json.dumps({"runs":len(run_rows),"out":str(args.out)},indent=2))

if __name__=="__main__": main()
