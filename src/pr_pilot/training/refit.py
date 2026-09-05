"""Validation-free final refit on all 1,000 frozen development structures.

The 900/100 development split is used only to choose each stage's epoch count and
other hyperparameters. Once those choices are frozen, this module retrains the
whole six-stage pipeline from scratch using:

- all 1,000 protein-prior structures;
- all 1,000 RNA-prior structures;
- all 1,000 complex-development structures.

No validation metric, early stopping or final-test information is consulted during
refit. This is the model that should be used for the primary final-100 report if
``sampling.final_refit_on_full_dev`` is true.
"""
from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import torch

from pr_pilot.runtime.manifest_dataset import ManifestTable
from pr_pilot.training.engine import (
    _adapter,
    _autocast,
    _cosine_schedule,
    _one_training_loss,
    build_model_from_config,
)
from pr_pilot.training.stages import (
    Stage,
    apply_joint_unfreezing,
    build_optimizer,
    configure_stage,
    trainable_parameter_report,
)


STAGE_ORDER = [
    Stage.PROTEIN_PRIOR,
    Stage.RNA_PRIOR,
    Stage.GLOBAL_C,
    Stage.DELTA_C,
    Stage.ALPHA,
    Stage.JOINT,
]


def selected_epoch_count(best_checkpoint: Path) -> int:
    """Return one-based selected epoch count from a validation-selected checkpoint."""
    payload = torch.load(best_checkpoint, map_location="cpu")
    if "epoch" not in payload:
        raise ValueError(f"Checkpoint {best_checkpoint} lacks selected epoch")
    return int(payload["epoch"]) + 1


def _optimizer(model, stage: Stage, cfg: dict):
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


def refit_stage(
    cfg: dict,
    stage: Stage,
    manifest_path: Path,
    epochs: int,
    out_dir: Path,
    init_checkpoint: Path | None = None,
    device: str | None = None,
) -> Path:
    """Train exactly ``epochs`` with no validation/early stopping."""
    if epochs <= 0:
        raise ValueError("Refit epochs must be positive")
    seed = int(cfg["experiment"]["pilot_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model_from_config(cfg).to(dev)
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(payload["model"])
    configure_stage(model, stage)
    optimizer = _optimizer(model, stage, cfg)
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    table = ManifestTable(manifest_path)
    total_steps = epochs * max(1, len(table))
    global_step = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "refit_metrics.jsonl"

    for epoch in range(epochs):
        model.train()
        progress = epoch / max(1, epochs - 1)
        if stage == Stage.JOINT:
            apply_joint_unfreezing(model, progress)
        adapter = _adapter(cfg, epoch, training=True)
        rows = list(table.rows())
        rng = random.Random(seed + epoch * 104729)
        rng.shuffle(rows)
        losses = []
        for row in rows:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, dev):
                loss, _ = _one_training_loss(model, row, adapter, stage, cfg, epoch, progress, dev)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite refit loss stage={stage.value} epoch={epoch} sample={row.sample_id}"
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
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "stage": stage.value,
                        "epoch": epoch,
                        "fixed_epoch_count": epochs,
                        "train_loss": float(np.mean(losses)),
                        "validation_used": False,
                        "trainable": trainable_parameter_report(model),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    checkpoint = out_dir / "refit.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "stage": stage.value,
            "epoch": epochs - 1,
            "refit": True,
            "validation_used": False,
            "manifest": str(manifest_path),
            "config": cfg,
        },
        checkpoint,
    )
    return checkpoint


def refit_full_pipeline(
    cfg: dict,
    manifest_root: Path,
    selected_training_root: Path,
    out_dir: Path,
    device: str | None = None,
) -> Path:
    """Refit all six stages using epoch counts selected on the 900/100 development split."""
    stage_manifests = {
        Stage.PROTEIN_PRIOR: manifest_root / "protein_pool.tsv",
        Stage.RNA_PRIOR: manifest_root / "rna_pool.tsv",
        Stage.GLOBAL_C: manifest_root / "complex_dev.tsv",
        Stage.DELTA_C: manifest_root / "complex_dev.tsv",
        Stage.ALPHA: manifest_root / "complex_dev.tsv",
        Stage.JOINT: manifest_root / "complex_dev.tsv",
    }
    selected_counts = {}
    for stage in STAGE_ORDER:
        selected = selected_training_root / stage.value / "best.pt"
        if not selected.exists():
            raise FileNotFoundError(f"Missing validation-selected checkpoint for refit: {selected}")
        selected_counts[stage.value] = selected_epoch_count(selected)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_epoch_counts.json").write_text(
        json.dumps(selected_counts, indent=2, sort_keys=True), encoding="utf-8"
    )
    previous = None
    for stage in STAGE_ORDER:
        previous = refit_stage(
            cfg,
            stage,
            stage_manifests[stage],
            selected_counts[stage.value],
            out_dir / stage.value,
            init_checkpoint=previous,
            device=device,
        )
    if previous is None:
        raise RuntimeError("Refit pipeline produced no checkpoint")
    (out_dir / "FINAL_REFIT_CHECKPOINT.txt").write_text(str(previous), encoding="utf-8")
    return previous
