"""Executable trainer for the six-stage 1k/1k/1k pilot.

The implementation is intentionally sample-wise for correctness and auditability.
The pilot is small enough that this is acceptable; later optimization can add
packed batches without changing objectives.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from pr_pilot.model.dmicf import JointPriorAndFieldModel, PRBatch
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row, load_protein_row, load_rna_row
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample, PolymerGraph
from pr_pilot.training.corruption import generate_corruption, hide_known_partner_tokens
from pr_pilot.training.losses import balanced_sequence_loss, alpha_entropy_regularizer, global_c_regularizer
from pr_pilot.training.stages import Stage, TaskRatioSchedule, apply_joint_unfreezing, build_optimizer, configure_stage, trainable_parameter_report


def _feature_dims(rbf_bins: int, rich_pr: bool) -> dict[str, int]:
    # Must match runtime/gemmi_adapter.py exactly.
    return {
        "protein_node": 15,
        "rna_node": 21,
        "protein_edge": rbf_bins + 15,
        "rna_edge": rbf_bins + 15,
        "pr_edge": (35 * rbf_bins + 35 + 3 + 3 + 9) if rich_pr else rbf_bins,
    }


def build_model_from_config(cfg: dict) -> JointPriorAndFieldModel:
    g = cfg["geometry"]
    dims = _feature_dims(int(g["rbf_bins"]), bool(g["rich_pr_geometry"]))
    return JointPriorAndFieldModel(
        dims["protein_node"], dims["protein_edge"], dims["rna_node"], dims["rna_edge"], dims["pr_edge"],
        hidden=int(g["hidden_dim"]), protein_layers=int(g["protein_layers"]), rna_layers=int(g["rna_layers"]),
        decoder_layers=int(g.get("decoder_layers", 2)),
    )


def _adapter(cfg: dict, epoch: int, training: bool) -> GemmiStructureAdapter:
    g = cfg["geometry"]
    sigma = float(g["coordinate_noise_angstrom"]) if training else 0.0
    return GemmiStructureAdapter(
        rbf_bins=int(g["rbf_bins"]), intra_max_neighbors=int(g["intra_max_neighbors"]),
        pr_cutoff_angstrom=float(g["pr_cutoff_angstrom"]), pr_max_neighbors=int(g["pr_max_neighbors"]),
        coordinate_noise_angstrom=sigma, seed=int(cfg["experiment"]["pilot_seed"]) + 1009 * epoch,
        rich_pr_geometry=bool(g["rich_pr_geometry"]),
    )


def _move_graph(g: PolymerGraph, device: torch.device) -> PolymerGraph:
    for name in ["node_x","edge_index","edge_x","sequence","interface","valid","fixed","reference_xyz","chain_index"]:
        setattr(g, name, getattr(g, name).to(device))
    return g


def _move_complex(s: ComplexTensorSample, device: torch.device) -> ComplexTensorSample:
    _move_graph(s.protein, device); _move_graph(s.rna, device)
    s.pr = PRBatch(
        s.pr.protein_index.to(device), s.pr.rna_index.to(device), s.pr.edge_features.to(device),
        s.pr.effective_distance.to(device), None if s.pr.edge_batch is None else s.pr.edge_batch.to(device),
    )
    return s


def _stable_uniform(seed: int, *parts: object) -> float:
    d = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(d[:8], "little") / float(2**64)


def _stage_flags(stage: Stage) -> tuple[bool, bool]:
    if stage == Stage.GLOBAL_C: return False, False
    if stage == Stage.DELTA_C: return True, False
    return True, True


def _all_interface_corruption(graph: PolymerGraph) -> tuple[Tensor, Tensor, Tensor]:
    target = graph.interface & graph.valid & ~graph.fixed
    if not target.any():
        raise ValueError("Complex contains no designable interface position")
    tokens = graph.sequence.clone()
    known = graph.valid.clone().bool()
    known[target] = False
    known[graph.fixed & graph.valid] = True
    return tokens, known, target


def _complex_forward(model: JointPriorAndFieldModel, s: ComplexTensorSample, ptok: Tensor, rtok: Tensor, pknown: Tensor, rknown: Tensor, stage: Stage) -> dict[str, Tensor]:
    use_delta, learned_alpha = _stage_flags(stage)
    return model(
        s.protein.node_x, s.protein.edge_index, s.protein.edge_x,
        s.rna.node_x, s.rna.edge_index, s.rna.edge_x, s.pr,
        ptok, rtok, pknown, rknown, use_delta=use_delta, learned_alpha=learned_alpha,
    )


def _one_training_loss(model: JointPriorAndFieldModel, row, adapter: GemmiStructureAdapter, stage: Stage, cfg: dict, epoch: int, progress: float, device: torch.device) -> tuple[Tensor, dict]:
    seed = int(cfg["experiment"]["pilot_seed"])
    mcfg, lcfg = cfg["masking"], cfg["loss"]

    if stage == Stage.PROTEIN_PRIOR:
        g = _move_graph(load_protein_row(adapter, row), device)
        c = generate_corruption(g, 20, row.sample_id, epoch, seed, float(mcfg["min_fraction"]), float(mcfg["max_fraction"]), float(mcfg["random_mask_probability"]), float(mcfg["local_patch_probability"]), float(mcfg["wrong_token_fraction_within_corruption"]), float(mcfg["interface_mask_multiplier"]))
        logits, _ = model.protein_prior_logits(g.node_x, g.edge_index, g.edge_x, c.input_tokens, c.known)
        b = balanced_sequence_loss(logits, g.sequence, c.target_mask, g.interface, None, None, None, None, "protein", float(lcfg["protein_label_smoothing"]), 0.0)
        return b.total, b.detached_scalars()

    if stage == Stage.RNA_PRIOR:
        g = _move_graph(load_rna_row(adapter, row), device)
        c = generate_corruption(g, 4, row.sample_id, epoch, seed, float(mcfg["min_fraction"]), float(mcfg["max_fraction"]), float(mcfg["random_mask_probability"]), float(mcfg["local_patch_probability"]), float(mcfg["wrong_token_fraction_within_corruption"]), float(mcfg["interface_mask_multiplier"]))
        logits, _ = model.rna_prior_logits(g.node_x, g.edge_index, g.edge_x, c.input_tokens, c.known)
        b = balanced_sequence_loss(None, None, None, None, logits, g.sequence, c.target_mask, g.interface, "rna", 0.0, float(lcfg["rna_label_smoothing"]))
        return b.total, b.detached_scalars()

    s = _move_complex(load_complex_row(adapter, row), device)
    if stage in {Stage.GLOBAL_C, Stage.DELTA_C, Stage.ALPHA}:
        protein_target = _stable_uniform(seed, row.sample_id, epoch, stage.value) < 0.5
        if protein_target:
            ptok, pknown, pmask = _all_interface_corruption(s.protein)
            rtok, rknown, rmask = s.rna.sequence.clone(), s.rna.valid.clone().bool(), torch.zeros_like(s.rna.valid)
            task = "protein"
        else:
            rtok, rknown, rmask = _all_interface_corruption(s.rna)
            ptok, pknown, pmask = s.protein.sequence.clone(), s.protein.valid.clone().bool(), torch.zeros_like(s.protein.valid)
            task = "rna"
        out = _complex_forward(model, s, ptok, rtok, pknown, rknown, stage)
        smooth_p = 0.0 if stage == Stage.GLOBAL_C else float(lcfg["protein_label_smoothing"])
        smooth_r = 0.0 if stage == Stage.GLOBAL_C else float(lcfg["rna_label_smoothing"])
        b = balanced_sequence_loss(out["protein_logits"] if task == "protein" else None, s.protein.sequence if task == "protein" else None, pmask if task == "protein" else None, s.protein.interface if task == "protein" else None, out["rna_logits"] if task == "rna" else None, s.rna.sequence if task == "rna" else None, rmask if task == "rna" else None, s.rna.interface if task == "rna" else None, task, smooth_p, smooth_r)
        loss = b.total
        if stage == Stage.GLOBAL_C and bool(lcfg.get("c_l2_enabled", False)):
            penalty, _ = global_c_regularizer(out["C"], b.total, float(lcfg["c_l2_target_fraction"]))
            loss = loss + penalty
        if stage == Stage.ALPHA and int(lcfg.get("alpha_entropy_warmup_steps", 0)) > 0 and progress < float(lcfg.get("alpha_entropy_warmup_fraction", 0.10)):
            alpha = out["alpha_p"] if task == "protein" else out["alpha_r"]
            groups = s.pr.protein_index if task == "protein" else s.pr.rna_index
            ng = s.protein.node_x.shape[0] if task == "protein" else s.rna.node_x.shape[0]
            loss = loss + float(lcfg["alpha_entropy_warmup_weight"]) * alpha_entropy_regularizer(alpha, groups, ng)
        return loss, {**b.detached_scalars(), "task": task}

    # Joint coordination: deterministic task sampling from the scheduled weights.
    sched = TaskRatioSchedule(tuple(cfg["training_stages"]["joint"]["task_ratio_start"]), tuple(cfg["training_stages"]["joint"]["task_ratio_end"]))
    weights = sched.weights(progress)
    u = _stable_uniform(seed, row.sample_id, epoch, "joint-task")
    if u < weights["protein_conditional"]:
        task_name = "protein_conditional"
    elif u < weights["protein_conditional"] + weights["rna_conditional"]:
        task_name = "rna_conditional"
    else:
        task_name = "joint"

    if task_name == "protein_conditional":
        pc = generate_corruption(s.protein, 20, row.sample_id, epoch, seed, float(mcfg["min_fraction"]), float(mcfg["max_fraction"]), float(mcfg["random_mask_probability"]), float(mcfg["local_patch_probability"]), float(mcfg["wrong_token_fraction_within_corruption"]), float(mcfg["interface_mask_multiplier"]))
        ptok, pknown, pmask = pc.input_tokens, pc.known, pc.target_mask
        rtok, rknown = s.rna.sequence.clone(), s.rna.valid.clone().bool()
        rknown = hide_known_partner_tokens(rknown, s.rna.fixed, row.sample_id, epoch, seed, float(mcfg["partner_token_dropout"]))
        rmask = torch.zeros_like(s.rna.valid)
        loss_task = "protein"
    elif task_name == "rna_conditional":
        rc = generate_corruption(s.rna, 4, row.sample_id, epoch, seed, float(mcfg["min_fraction"]), float(mcfg["max_fraction"]), float(mcfg["random_mask_probability"]), float(mcfg["local_patch_probability"]), float(mcfg["wrong_token_fraction_within_corruption"]), float(mcfg["interface_mask_multiplier"]))
        rtok, rknown, rmask = rc.input_tokens, rc.known, rc.target_mask
        ptok, pknown = s.protein.sequence.clone(), s.protein.valid.clone().bool()
        pknown = hide_known_partner_tokens(pknown, s.protein.fixed, row.sample_id, epoch, seed, float(mcfg["partner_token_dropout"]))
        pmask = torch.zeros_like(s.protein.valid)
        loss_task = "rna"
    else:
        pc = generate_corruption(s.protein, 20, row.sample_id, epoch, seed, float(mcfg["min_fraction"]), float(mcfg["max_fraction"]), float(mcfg["random_mask_probability"]), float(mcfg["local_patch_probability"]), float(mcfg["wrong_token_fraction_within_corruption"]), float(mcfg["interface_mask_multiplier"]))
        rc = generate_corruption(s.rna, 4, row.sample_id, epoch, seed + 17, float(mcfg["min_fraction"]), float(mcfg["max_fraction"]), float(mcfg["random_mask_probability"]), float(mcfg["local_patch_probability"]), float(mcfg["wrong_token_fraction_within_corruption"]), float(mcfg["interface_mask_multiplier"]))
        ptok, pknown, pmask = pc.input_tokens, pc.known, pc.target_mask
        rtok, rknown, rmask = rc.input_tokens, rc.known, rc.target_mask
        loss_task = "joint"

    out = _complex_forward(model, s, ptok, rtok, pknown, rknown, Stage.JOINT)
    b = balanced_sequence_loss(out["protein_logits"] if loss_task in {"protein","joint"} else None, s.protein.sequence if loss_task in {"protein","joint"} else None, pmask if loss_task in {"protein","joint"} else None, s.protein.interface if loss_task in {"protein","joint"} else None, out["rna_logits"] if loss_task in {"rna","joint"} else None, s.rna.sequence if loss_task in {"rna","joint"} else None, rmask if loss_task in {"rna","joint"} else None, s.rna.interface if loss_task in {"rna","joint"} else None, loss_task, float(lcfg["protein_label_smoothing"]), float(lcfg["rna_label_smoothing"]))
    return b.total, {**b.detached_scalars(), "task": task_name}


@torch.no_grad()
def validate_stage(model: JointPriorAndFieldModel, stage: Stage, manifest: ManifestTable, cfg: dict, device: torch.device) -> float:
    """Pre-registered normalized NLL validation; no stochastic augmentation."""
    model.eval(); adapter = _adapter(cfg, 0, training=False); values = []
    for row in manifest.rows():
        if stage == Stage.PROTEIN_PRIOR:
            g = _move_graph(load_protein_row(adapter, row), device)
            known = g.fixed & g.valid
            logits, _ = model.protein_prior_logits(g.node_x,g.edge_index,g.edge_x,g.sequence,known)
            b = balanced_sequence_loss(logits,g.sequence,g.valid & ~g.fixed,g.interface,None,None,None,None,"protein",0.0,0.0)
            values.append(float(b.total))
        elif stage == Stage.RNA_PRIOR:
            g = _move_graph(load_rna_row(adapter, row), device)
            known = g.fixed & g.valid
            logits, _ = model.rna_prior_logits(g.node_x,g.edge_index,g.edge_x,g.sequence,known)
            b = balanced_sequence_loss(None,None,None,None,logits,g.sequence,g.valid & ~g.fixed,g.interface,"rna",0.0,0.0)
            values.append(float(b.total))
        else:
            s = _move_complex(load_complex_row(adapter,row), device)
            # Conditional interface metrics in both directions.
            ptok,pknown,pmask = _all_interface_corruption(s.protein)
            rtok,rknown,rmask = _all_interface_corruption(s.rna)
            use_delta, learned_alpha = _stage_flags(stage)
            outp = model(s.protein.node_x,s.protein.edge_index,s.protein.edge_x,s.rna.node_x,s.rna.edge_index,s.rna.edge_x,s.pr,ptok,s.rna.sequence,pknown,s.rna.valid,use_delta=use_delta,learned_alpha=learned_alpha)
            bp = balanced_sequence_loss(outp["protein_logits"],s.protein.sequence,pmask,s.protein.interface,None,None,None,None,"protein",0.0,0.0)
            outr = model(s.protein.node_x,s.protein.edge_index,s.protein.edge_x,s.rna.node_x,s.rna.edge_index,s.rna.edge_x,s.pr,s.protein.sequence,rtok,s.protein.valid,rknown,use_delta=use_delta,learned_alpha=learned_alpha)
            br = balanced_sequence_loss(None,None,None,None,outr["rna_logits"],s.rna.sequence,rmask,s.rna.interface,"rna",0.0,0.0)
            if stage == Stage.JOINT:
                # Full-mask joint term, averaged with the two interface-conditionals.
                pnone = s.protein.fixed & s.protein.valid; rnone = s.rna.fixed & s.rna.valid
                outj = model(s.protein.node_x,s.protein.edge_index,s.protein.edge_x,s.rna.node_x,s.rna.edge_index,s.rna.edge_x,s.pr,s.protein.sequence,s.rna.sequence,pnone,rnone)
                bj = balanced_sequence_loss(outj["protein_logits"],s.protein.sequence,s.protein.valid & ~s.protein.fixed,s.protein.interface,outj["rna_logits"],s.rna.sequence,s.rna.valid & ~s.rna.fixed,s.rna.interface,"joint",0.0,0.0)
                values.append((float(bp.total)+float(br.total)+float(bj.total))/3.0)
            else:
                values.append((float(bp.total)+float(br.total))/2.0)
    if not values:
        raise ValueError("Empty validation manifest")
    return float(np.mean(values))


def _cosine_schedule(optimizer: torch.optim.Optimizer, step: int, total: int, warmup_fraction: float, base_lrs: list[float]) -> None:
    warm = max(1, int(total * warmup_fraction))
    if step < warm:
        scale = (step + 1) / warm
    else:
        x = (step - warm) / max(1, total - warm)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(max(x,0.0),1.0)))
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale


def train_stage(
    cfg: dict,
    stage: Stage,
    train_manifest: Path,
    val_manifest: Path,
    out_dir: Path,
    init_checkpoint: Path | None = None,
    device: str | None = None,
) -> Path:
    """Train one stage and return best checkpoint path."""
    seed = int(cfg["experiment"]["pilot_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(dev)
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(payload["model"])
    configure_stage(model, stage)
    optcfg = cfg["optimization"]
    optimizer = build_optimizer(model, stage, float(optcfg["lr_heads"]), float(optcfg["lr_projections"]), float(optcfg["lr_encoder_top"]), float(optcfg["lr_encoder_bottom"]), float(optcfg["lr_global_c_joint"]), float(optcfg["weight_decay"]), float(optcfg["layerwise_lr_decay"]))
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    train_table, val_table = ManifestTable(train_manifest), ManifestTable(val_manifest)
    max_epochs = int(cfg["training_stages"][stage.value].get("max_epochs", cfg["optimization"].get("max_epochs_default", 100)))
    patience = int(optcfg["early_stopping_patience"])
    total_steps = max_epochs * max(1, len(train_table))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    best_path = out_dir / "best.pt"
    best = float("inf"); bad_epochs = 0; global_step = 0

    for epoch in range(max_epochs):
        model.train(); progress = epoch / max(1, max_epochs - 1)
        if stage == Stage.JOINT:
            apply_joint_unfreezing(model, progress)
        adapter = _adapter(cfg, epoch, training=True)
        rows = list(train_table.rows())
        rng = random.Random(seed + epoch * 104729); rng.shuffle(rows)
        running = []
        for row in rows:
            optimizer.zero_grad(set_to_none=True)
            loss, detail = _one_training_loss(model,row,adapter,stage,cfg,epoch,progress,dev)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at {stage.value} epoch={epoch} sample={row.sample_id}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(optcfg["grad_clip_norm"]))
            optimizer.step()
            global_step += 1
            _cosine_schedule(optimizer, global_step, total_steps, float(optcfg["warmup_fraction"]), base_lrs)
            running.append(float(loss.detach().cpu()))
        val = validate_stage(model,stage,val_table,cfg,dev)
        record = {"stage":stage.value,"epoch":epoch,"train_loss":float(np.mean(running)),"val_metric":val,"trainable":trainable_parameter_report(model)}
        with metrics_path.open("a",encoding="utf-8") as f: f.write(json.dumps(record,sort_keys=True)+"\n")
        if val < best - 1e-6:
            best = val; bad_epochs = 0
            torch.save({"model":model.state_dict(),"stage":stage.value,"epoch":epoch,"val_metric":val,"config":cfg},best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience: break
    if not best_path.exists(): raise RuntimeError("No checkpoint was saved")
    return best_path
