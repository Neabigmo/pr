"""Same-data internal fairness controls for DM-ICF.

``partner_blind`` forces the cross-molecular correction to zero.
``geometry_only`` retains q/DeltaC/alpha capacity and partner geometry but removes
specific partner-token identity by averaging compatibility across the partner
alphabet.

Controls share frozen manifests, structural-prior checkpoints, stage ownership,
validation rules and full-1000 schedule-prefix refit semantics with the primary
model.
"""
from __future__ import annotations

from pathlib import Path
import json
import random
import types

import numpy as np
import torch

from pr_pilot.model.dmicf import JointPriorAndFieldModel
from pr_pilot.runtime.manifest_dataset import ManifestTable
from pr_pilot.training.engine import (
    _adapter,
    _autocast,
    _cosine_schedule,
    _one_training_loss,
    build_model_from_config,
    validate_stage,
)
from pr_pilot.training.refit import (
    development_schedule_horizon,
    schedule_progress,
    selected_epoch_count,
)
from pr_pilot.training.stages import (
    Stage,
    apply_joint_unfreezing,
    build_optimizer,
    configure_stage,
    trainable_parameter_report,
)


CONTROL_STAGES = [Stage.GLOBAL_C, Stage.DELTA_C, Stage.ALPHA, Stage.JOINT]


def install_control_mode(model: JointPriorAndFieldModel, mode: str) -> None:
    """Replace only cross-molecular output semantics; leave architecture intact."""
    if mode not in {"partner_blind", "geometry_only"}:
        raise ValueError("mode must be partner_blind or geometry_only")
    original = model.forward

    def controlled_forward(self, *args, **kwargs):
        out = original(*args, **kwargs)
        if mode == "partner_blind":
            out["protein_delta_logits"] = torch.zeros_like(out["protein_struct_logits"])
            out["rna_delta_logits"] = torch.zeros_like(out["rna_struct_logits"])
            out["protein_logits"] = out["protein_struct_logits"]
            out["rna_logits"] = out["rna_struct_logits"]
            return out

        if len(args) < 7:
            raise ValueError(
                "Controlled forward requires the standard positional JointPriorAndFieldModel call"
            )
        pr = args[6]
        cedge = out["C"].unsqueeze(0) + out["DeltaC"]

        p_edge = cedge.mean(dim=-1) * out["alpha_p"][:, None]
        p_delta = torch.zeros_like(out["protein_struct_logits"])
        p_delta.index_add_(0, pr.protein_index, p_edge)
        p_delta = self.dmicf.lambda_p * p_delta

        r_edge = cedge.mean(dim=-2) * out["alpha_r"][:, None]
        r_delta = torch.zeros_like(out["rna_struct_logits"])
        r_delta.index_add_(0, pr.rna_index, r_edge)
        r_delta = self.dmicf.lambda_r * r_delta

        out["protein_delta_logits"] = p_delta
        out["rna_delta_logits"] = r_delta
        out["protein_logits"] = out["protein_struct_logits"] + p_delta
        out["rna_logits"] = out["rna_struct_logits"] + r_delta
        return out

    model.forward = types.MethodType(controlled_forward, model)
    model.control_mode = mode


def _optimizer(model: JointPriorAndFieldModel, stage: Stage, cfg: dict):
    opt = cfg["optimization"]
    return build_optimizer(
        model,
        stage,
        float(opt["lr_heads"]),
        float(opt["lr_projections"]),
        float(opt["lr_encoder_top"]),
        float(opt["lr_encoder_bottom"]),
        float(opt["lr_global_c_joint"]),
        float(opt["weight_decay"]),
        float(opt["layerwise_lr_decay"]),
    )


def _load_control_model(
    cfg: dict, checkpoint: Path, stage: Stage, mode: str, device: torch.device
):
    model = build_model_from_config(cfg).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    configure_stage(model, stage)
    install_control_mode(model, mode)
    return model


def train_control_stage(
    cfg: dict,
    mode: str,
    stage: Stage,
    train_manifest: Path,
    val_manifest: Path,
    init_checkpoint: Path,
    out_dir: Path,
    device: str | None = None,
) -> Path:
    if stage not in CONTROL_STAGES:
        raise ValueError(f"Control mode applies only to {CONTROL_STAGES}")
    seed = int(cfg["experiment"]["pilot_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_control_model(cfg, init_checkpoint, stage, mode, dev)
    optimizer = _optimizer(model, stage, cfg)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    train_table = ManifestTable(train_manifest)
    val_table = ManifestTable(val_manifest)
    max_epochs = development_schedule_horizon(cfg, stage)
    patience = int(cfg["optimization"]["early_stopping_patience"])
    total_steps = max_epochs * max(1, len(train_table))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    best_path = out_dir / "best.pt"
    best = float("inf")
    bad_epochs = 0
    global_step = 0

    for epoch in range(max_epochs):
        model.train()
        progress = schedule_progress(epoch, max_epochs)
        if stage == Stage.JOINT:
            apply_joint_unfreezing(model, progress)
        adapter = _adapter(cfg, epoch, training=True)
        rows = list(train_table.rows())
        random.Random(seed + epoch * 104729).shuffle(rows)
        losses = []
        for row in rows:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, dev):
                loss, _ = _one_training_loss(
                    model, row, adapter, stage, cfg, epoch, progress, dev
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {mode} loss at {stage.value} {row.sample_id}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                float(cfg["optimization"]["grad_clip_norm"]),
            )
            optimizer.step()
            global_step += 1
            _cosine_schedule(
                optimizer,
                global_step,
                total_steps,
                float(cfg["optimization"]["warmup_fraction"]),
                base_lrs,
            )
            losses.append(float(loss.detach().cpu()))
        val = validate_stage(model, stage, val_table, cfg, dev)
        record = {
            "control_mode": mode,
            "stage": stage.value,
            "epoch": epoch,
            "schedule_horizon_epochs": max_epochs,
            "schedule_progress": progress,
            "train_loss": float(np.mean(losses)),
            "val_metric": val,
            "trainable": trainable_parameter_report(model),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if val < best - 1e-6:
            best = val
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "stage": stage.value,
                    "control_mode": mode,
                    "epoch": epoch,
                    "schedule_horizon_epochs": max_epochs,
                    "schedule_progress": progress,
                    "val_metric": val,
                    "config": cfg,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if not best_path.exists():
        raise RuntimeError("Control stage saved no checkpoint")
    return best_path


def refit_control_stage(
    cfg: dict,
    mode: str,
    stage: Stage,
    full_manifest: Path,
    selected_dev_checkpoint: Path,
    init_checkpoint: Path,
    out_dir: Path,
    device: str | None = None,
) -> Path:
    """Validation-free full-1000 refit with the development schedule prefix."""
    epochs = selected_epoch_count(selected_dev_checkpoint)
    horizon = development_schedule_horizon(cfg, stage)
    if epochs > horizon:
        raise ValueError("Selected control epoch exceeds schedule horizon")
    seed = int(cfg["experiment"]["pilot_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_control_model(cfg, init_checkpoint, stage, mode, dev)
    optimizer = _optimizer(model, stage, cfg)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    table = ManifestTable(full_manifest)
    total_steps = horizon * max(1, len(table))
    global_step = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        progress = schedule_progress(epoch, horizon)
        if stage == Stage.JOINT:
            apply_joint_unfreezing(model, progress)
        adapter = _adapter(cfg, epoch, training=True)
        rows = list(table.rows())
        random.Random(seed + epoch * 104729).shuffle(rows)
        for row in rows:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, dev):
                loss, _ = _one_training_loss(
                    model, row, adapter, stage, cfg, epoch, progress, dev
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                float(cfg["optimization"]["grad_clip_norm"]),
            )
            optimizer.step()
            global_step += 1
            _cosine_schedule(
                optimizer,
                global_step,
                total_steps,
                float(cfg["optimization"]["warmup_fraction"]),
                base_lrs,
            )

    checkpoint = out_dir / "refit.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "stage": stage.value,
            "control_mode": mode,
            "epoch": epochs - 1,
            "selected_epoch_count": epochs,
            "schedule_horizon_epochs": horizon,
            "schedule_progress_at_stop": schedule_progress(epochs - 1, horizon),
            "refit": True,
            "validation_used": False,
            "config": cfg,
        },
        checkpoint,
    )
    return checkpoint


def run_control_pipeline(
    cfg: dict,
    mode: str,
    manifests: Path,
    primary_development_prior: Path,
    primary_refit_prior: Path,
    out_dir: Path,
    device: str | None = None,
) -> Path:
    dev_prev = primary_development_prior
    refit_prev = primary_refit_prior
    for stage in CONTROL_STAGES:
        dev_stage = out_dir / "development" / stage.value
        dev_checkpoint = train_control_stage(
            cfg,
            mode,
            stage,
            manifests / "complex_train.tsv",
            manifests / "complex_val.tsv",
            dev_prev,
            dev_stage,
            device,
        )
        refit_stage_dir = out_dir / "refit_full1000" / stage.value
        refit_checkpoint = refit_control_stage(
            cfg,
            mode,
            stage,
            manifests / "complex_dev.tsv",
            dev_checkpoint,
            refit_prev,
            refit_stage_dir,
            device,
        )
        dev_prev = dev_checkpoint
        refit_prev = refit_checkpoint
    (out_dir / "FINAL_REFIT_CHECKPOINT.txt").write_text(
        str(refit_prev), encoding="utf-8"
    )
    return refit_prev
