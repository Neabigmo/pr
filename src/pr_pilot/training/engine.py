"""Executable trainer for the six-stage 1k/1k/1k pilot.

Scientific invariants implemented here:
- canonical interface labels come from model-independent full-heavy-atom contacts;
- C -> DeltaC -> alpha are separable stages;
- primary joint training gradually unfreezes pretrained encoders while keeping C
  fixed;
- a joint run with no initialization checkpoint is treated as a true scratch
  control and all parameters are trainable from step 0;
- joint checkpoint selection uses deterministic sequential teacher-forced
  pseudo-NLL rather than a simultaneous full-mask forward that would suppress
  partner-dependent corrections.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
import multiprocessing as mp
from pathlib import Path
import hashlib
import json
import math
import random

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from pr_pilot.model.dmicf import JointPriorAndFieldModel, PRBatch
from pr_pilot.runtime.dataset_adapter import ComplexTensorSample, PolymerGraph
from pr_pilot.runtime.gemmi_adapter import GemmiStructureAdapter, feature_dimensions
from pr_pilot.runtime.manifest_dataset import ManifestTable, load_complex_row, load_protein_row, load_rna_row
from pr_pilot.training.corruption import generate_corruption, hide_known_partner_tokens
from pr_pilot.training.losses import alpha_entropy_regularizer, balanced_sequence_loss, global_c_regularizer
from pr_pilot.training.stages import (
    Stage,
    TaskRatioSchedule,
    apply_joint_unfreezing,
    build_optimizer,
    configure_stage,
    make_joint_fully_trainable,
    trainable_parameter_report,
)


def build_model_from_config(cfg: dict) -> JointPriorAndFieldModel:
    g = cfg["geometry"]
    dims = feature_dimensions(int(g["rbf_bins"]), bool(g["rich_pr_geometry"]))
    return JointPriorAndFieldModel(
        dims["protein_node"],
        dims["protein_edge"],
        dims["rna_node"],
        dims["rna_edge"],
        dims["pr_edge"],
        hidden=int(g["hidden_dim"]),
        protein_layers=int(g["protein_layers"]),
        rna_layers=int(g["rna_layers"]),
        decoder_layers=int(g.get("decoder_layers", 2)),
        interaction_layers=int(g.get("interaction_layers", 3)),
        drop_path=float(g.get("drop_path", 0.0)),
    )


def _adapter(cfg: dict, epoch: int, training: bool) -> GemmiStructureAdapter:
    g = cfg["geometry"]
    sigma = float(g["coordinate_noise_angstrom"]) if training else 0.0
    return GemmiStructureAdapter(
        rbf_bins=int(g["rbf_bins"]),
        intra_max_neighbors=int(g["intra_max_neighbors"]),
        pr_cutoff_angstrom=float(g["pr_cutoff_angstrom"]),
        pr_max_neighbors=int(g["pr_max_neighbors"]),
        coordinate_noise_angstrom=sigma,
        seed=int(cfg["experiment"]["pilot_seed"]) + 1009 * epoch,
        rich_pr_geometry=bool(g["rich_pr_geometry"]),
    )


def _move_graph(graph: PolymerGraph, device: torch.device) -> PolymerGraph:
    for name in ["node_x", "edge_index", "edge_x", "sequence", "interface", "valid", "fixed", "reference_xyz", "chain_index"]:
        setattr(graph, name, getattr(graph, name).to(device))
    return graph


def _move_complex(sample: ComplexTensorSample, device: torch.device) -> ComplexTensorSample:
    _move_graph(sample.protein, device)
    _move_graph(sample.rna, device)
    sample.pr = PRBatch(
        sample.pr.protein_index.to(device),
        sample.pr.rna_index.to(device),
        sample.pr.edge_features.to(device),
        sample.pr.effective_distance.to(device),
        None if sample.pr.edge_batch is None else sample.pr.edge_batch.to(device),
    )
    return sample


def _stable_seed(seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _stable_uniform(seed: int, *parts: object) -> float:
    return _stable_seed(seed, *parts) / float(2**32)


def _drop_intra_edges(graph: PolymerGraph, probability: float, seed: int) -> None:
    p = float(probability)
    if p <= 0 or graph.edge_index.shape[1] == 0:
        return
    if not 0 <= p < 1:
        raise ValueError("intra edge-drop probability must be in [0,1)")
    e = graph.edge_index.shape[1]
    rng = np.random.default_rng(seed)
    covalent = graph.edge_x[:, -1].detach().cpu().numpy() > 0.5
    keep = covalent | (rng.random(e) >= p)
    src = graph.edge_index[0].detach().cpu().numpy()
    for node in np.unique(src):
        idx = np.flatnonzero(src == node)
        if len(idx) and not keep[idx].any():
            keep[idx[0]] = True
    keep_t = torch.as_tensor(keep, dtype=torch.bool, device=graph.edge_index.device)
    graph.edge_index = graph.edge_index[:, keep_t]
    graph.edge_x = graph.edge_x[keep_t]
    graph.validate()


def _drop_pr_edges(sample: ComplexTensorSample, probability: float, seed: int) -> None:
    p = float(probability)
    e = int(sample.pr.protein_index.numel())
    if p <= 0 or e == 0:
        return
    if not 0 <= p < 1:
        raise ValueError("PR edge-drop probability must be in [0,1)")
    rng = np.random.default_rng(seed)
    keep = rng.random(e) >= p
    pi = sample.pr.protein_index.detach().cpu().numpy()
    ri = sample.pr.rna_index.detach().cpu().numpy()
    dist = sample.pr.effective_distance.detach().cpu().numpy()
    for labels in (pi, ri):
        for node in np.unique(labels):
            idx = np.flatnonzero(labels == node)
            if len(idx):
                keep[idx[np.argmin(dist[idx])]] = True
    mask = torch.as_tensor(keep, dtype=torch.bool, device=sample.pr.protein_index.device)
    sample.pr = PRBatch(
        sample.pr.protein_index[mask],
        sample.pr.rna_index[mask],
        sample.pr.edge_features[mask],
        sample.pr.effective_distance[mask],
        None if sample.pr.edge_batch is None else sample.pr.edge_batch[mask],
    )
    sample.validate()


def _augment_graphs(sample, cfg: dict, stage: Stage, sample_id: str, epoch: int) -> None:
    gcfg = cfg["geometry"]
    seed = int(cfg["experiment"]["pilot_seed"])
    if isinstance(sample, PolymerGraph):
        _drop_intra_edges(sample, float(gcfg.get("edge_dropout", 0.0)), _stable_seed(seed, sample_id, epoch, "intra"))
        return
    _drop_intra_edges(sample.protein, float(gcfg.get("edge_dropout", 0.0)), _stable_seed(seed, sample_id, epoch, "p-intra"))
    _drop_intra_edges(sample.rna, float(gcfg.get("edge_dropout", 0.0)), _stable_seed(seed, sample_id, epoch, "r-intra"))
    if stage in {Stage.DELTA_C, Stage.ALPHA, Stage.JOINT}:
        _drop_pr_edges(sample, float(gcfg.get("pr_edge_dropout", 0.0)), _stable_seed(seed, sample_id, epoch, "pr"))


def _stage_flags(stage: Stage) -> tuple[bool, bool]:
    if stage == Stage.GLOBAL_C:
        return False, False
    if stage == Stage.DELTA_C:
        return True, False
    return True, True


def _joint_unfreezing_mode(cfg: dict) -> str:
    mode = str(cfg["training_stages"]["joint"].get("unfreezing_mode", "gradual"))
    if mode not in {"gradual", "all_trainable_from_start"}:
        raise ValueError(f"Unknown joint unfreezing_mode={mode!r}")
    return mode


def _load_training_sample(adapter: GemmiStructureAdapter, stage: Stage, row):
    if stage == Stage.PROTEIN_PRIOR:
        return load_protein_row(adapter, row)
    if stage == Stage.RNA_PRIOR:
        return load_rna_row(adapter, row)
    return load_complex_row(adapter, row)


def _adapter_prefetch_spec(adapter: GemmiStructureAdapter) -> tuple:
    """Return the CPU-only adapter settings needed by a worker process."""
    return (
        adapter.rbf_bins,
        adapter.intra_max_neighbors,
        adapter.pr_cutoff,
        adapter.pr_max_neighbors,
        adapter.noise,
        adapter.seed,
        adapter.rich_pr_geometry,
    )


_PROCESS_ADAPTER: GemmiStructureAdapter | None = None
_PROCESS_ADAPTER_SPEC: tuple | None = None


def _load_training_sample_in_process(payload: tuple):
    """Load one sample in a persistent CPU worker process.

    The adapter is reconstructed lazily and cached per worker.  Passing only
    immutable settings and the manifest row keeps the parent process from
    serializing a live adapter or sharing parser state across processes.
    """
    global _PROCESS_ADAPTER, _PROCESS_ADAPTER_SPEC
    adapter_spec, stage_value, row = payload
    if _PROCESS_ADAPTER is None or _PROCESS_ADAPTER_SPEC != adapter_spec:
        _PROCESS_ADAPTER = GemmiStructureAdapter(
            rbf_bins=adapter_spec[0],
            intra_max_neighbors=adapter_spec[1],
            pr_cutoff_angstrom=adapter_spec[2],
            pr_max_neighbors=adapter_spec[3],
            coordinate_noise_angstrom=adapter_spec[4],
            seed=adapter_spec[5],
            rich_pr_geometry=adapter_spec[6],
        )
        _PROCESS_ADAPTER_SPEC = adapter_spec
    return _prefetch_numpy(_load_training_sample(_PROCESS_ADAPTER, Stage(stage_value), row))


def _prefetch_numpy(value):
    """Convert a CPU graph to NumPy before crossing a process boundary.

    PyTorch's multiprocessing reducer may transfer each tensor through shared
    memory/file descriptors.  A bounded queue of variable-size graphs can then
    stall behind those transfers.  Plain NumPy buffers use ordinary bounded
    pickle payloads and are restored to CPU tensors in the consumer process.
    """
    fields = [
        "node_x",
        "edge_index",
        "edge_x",
        "sequence",
        "interface",
        "valid",
        "fixed",
        "reference_xyz",
        "chain_index",
    ]

    def graph_to_numpy(graph: PolymerGraph) -> PolymerGraph:
        for name in fields:
            setattr(graph, name, getattr(graph, name).detach().cpu().numpy())
        return graph

    if isinstance(value, PolymerGraph):
        return graph_to_numpy(value)
    value.protein = graph_to_numpy(value.protein)
    value.rna = graph_to_numpy(value.rna)
    value.pr = PRBatch(
        value.pr.protein_index.detach().cpu().numpy(),
        value.pr.rna_index.detach().cpu().numpy(),
        value.pr.edge_features.detach().cpu().numpy(),
        value.pr.effective_distance.detach().cpu().numpy(),
        None if value.pr.edge_batch is None else value.pr.edge_batch.detach().cpu().numpy(),
    )
    return value


def _restore_prefetched_tensors(value):
    """Restore NumPy-backed process results to CPU tensors in the trainer."""
    fields = [
        "node_x",
        "edge_index",
        "edge_x",
        "sequence",
        "interface",
        "valid",
        "fixed",
        "reference_xyz",
        "chain_index",
    ]

    def graph_to_tensors(graph: PolymerGraph) -> PolymerGraph:
        for name in fields:
            tensor = getattr(graph, name)
            if not torch.is_tensor(tensor):
                setattr(graph, name, torch.from_numpy(tensor))
        return graph

    if isinstance(value, PolymerGraph):
        return graph_to_tensors(value)
    value.protein = graph_to_tensors(value.protein)
    value.rna = graph_to_tensors(value.rna)
    value.pr = PRBatch(
        torch.from_numpy(value.pr.protein_index) if not torch.is_tensor(value.pr.protein_index) else value.pr.protein_index,
        torch.from_numpy(value.pr.rna_index) if not torch.is_tensor(value.pr.rna_index) else value.pr.rna_index,
        torch.from_numpy(value.pr.edge_features) if not torch.is_tensor(value.pr.edge_features) else value.pr.edge_features,
        torch.from_numpy(value.pr.effective_distance) if not torch.is_tensor(value.pr.effective_distance) else value.pr.effective_distance,
        None if value.pr.edge_batch is None else (torch.from_numpy(value.pr.edge_batch) if not torch.is_tensor(value.pr.edge_batch) else value.pr.edge_batch),
    )
    return value


def _prefetch_training_samples(
    rows: list,
    adapter: GemmiStructureAdapter,
    stage: Stage,
    workers: int,
    process_pool: ProcessPoolExecutor | None = None,
    backend: str = "thread",
):
    """Yield rows with bounded CPU-side preparation ahead of model execution.

    The model still receives one variable-size graph at a time, so this does not
    change optimizer steps, batching semantics, data order, or metrics.  It only
    overlaps structure parsing/graph construction with the previous GPU step.

    ``thread`` is the portable default.  ``process`` is intended for Linux
    training hosts where Gemmi/feature construction is Python-heavy and a
    thread pool cannot use the available CPU cores effectively.  A supplied
    process pool is reused across epochs and validation to avoid respawn cost.
    """
    workers = max(0, int(workers))
    if workers == 0:
        for row in rows:
            yield row, _load_training_sample(adapter, stage, row)
        return

    if backend not in {"thread", "process"}:
        raise ValueError(f"Unknown data_prefetch_backend={backend!r}")

    pending = deque()
    row_iter = iter(rows)
    owned_pool = None
    if backend == "process":
        pool = process_pool
        if pool is None:
            # Spawn avoids inheriting a live CUDA context into CPU workers.
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
            owned_pool = pool
        submit = lambda item: pool.submit(_load_training_sample_in_process, item)
        adapter_spec = _adapter_prefetch_spec(adapter)
        make_item = lambda row: (adapter_spec, stage.value, row)
    else:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="structure-prefetch")
        owned_pool = pool
        submit = lambda item: pool.submit(_load_training_sample, adapter, stage, item)
        make_item = lambda row: row

    try:
        for _ in range(min(len(rows), workers * 2)):
            row = next(row_iter, None)
            if row is None:
                break
            pending.append((row, submit(make_item(row))))
        while pending:
            row, future = pending.popleft()
            prepared = future.result()
            if backend == "process":
                prepared = _restore_prefetched_tensors(prepared)
            next_row = next(row_iter, None)
            if next_row is not None:
                pending.append((next_row, submit(make_item(next_row))))
            yield row, prepared
    finally:
        if owned_pool is not None:
            owned_pool.shutdown(wait=True)


def _all_interface_corruption(graph: PolymerGraph) -> tuple[Tensor, Tensor, Tensor]:
    target = graph.interface & graph.valid & ~graph.fixed
    if not target.any():
        raise ValueError("Complex contains no designable canonical interface position")
    tokens = graph.sequence.clone()
    known = graph.valid.clone().bool()
    known[target] = False
    known[graph.fixed & graph.valid] = True
    return tokens, known, target


def _complex_forward(
    model: JointPriorAndFieldModel,
    sample: ComplexTensorSample,
    ptok: Tensor,
    rtok: Tensor,
    pknown: Tensor,
    rknown: Tensor,
    stage: Stage,
) -> dict[str, Tensor]:
    use_delta, learned_alpha = _stage_flags(stage)
    return model(
        sample.protein.node_x,
        sample.protein.edge_index,
        sample.protein.edge_x,
        sample.rna.node_x,
        sample.rna.edge_index,
        sample.rna.edge_x,
        sample.pr,
        ptok,
        rtok,
        pknown,
        rknown,
        use_delta=use_delta,
        learned_alpha=learned_alpha,
    )


def _corrupt(graph: PolymerGraph, alphabet: int, row, epoch: int, seed: int, cfg: dict, progress: float, joint: bool = False):
    mcfg = cfg["masking"]
    min_fraction = max(0.20, float(mcfg["min_fraction"])) if joint else float(mcfg["min_fraction"])
    return generate_corruption(
        graph,
        alphabet,
        row.sample_id,
        epoch,
        seed,
        min_fraction,
        float(mcfg["max_fraction"]),
        float(mcfg["random_mask_probability"]),
        float(mcfg["local_patch_probability"]),
        float(mcfg["wrong_token_fraction_within_corruption"]),
        float(mcfg["interface_mask_multiplier"]),
        progress=progress,
        use_curriculum=bool(mcfg.get("curriculum", True)),
        full_mask_probability=float(mcfg.get("full_mask_probability", 0.15)),
    )


def _maybe_hard_mask(model, graph, tokens, known, mask, polymer: str, fraction: float) -> Tensor:
    if fraction <= 0 or mask.sum() <= 1:
        return mask
    with torch.no_grad():
        if polymer == "protein":
            logits, _ = model.protein_prior_logits(graph.node_x, graph.edge_index, graph.edge_x, tokens, known)
        else:
            logits, _ = model.rna_prior_logits(graph.node_x, graph.edge_index, graph.edge_x, tokens, known)
        prob = torch.softmax(logits.float(), dim=-1)
        native = graph.sequence
        score = prob.max(dim=-1).values - prob.gather(-1, native[:, None]).squeeze(-1)
        idx = torch.where(mask)[0]
        n = max(1, int(round(len(idx) * min(max(fraction, 0.05), 1.0))))
        chosen = idx[torch.topk(score[idx], k=n, largest=True).indices]
        out = torch.zeros_like(mask)
        out[chosen] = True
        return out


def _one_training_loss(
    model: JointPriorAndFieldModel,
    row,
    adapter: GemmiStructureAdapter,
    stage: Stage,
    cfg: dict,
    epoch: int,
    progress: float,
    device: torch.device,
    prepared=None,
) -> tuple[Tensor, dict]:
    seed = int(cfg["experiment"]["pilot_seed"])
    mcfg, lcfg = cfg["masking"], cfg["loss"]

    if stage == Stage.PROTEIN_PRIOR:
        graph = _move_graph(prepared if prepared is not None else load_protein_row(adapter, row), device)
        _augment_graphs(graph, cfg, stage, row.sample_id, epoch)
        corruption = _corrupt(graph, 20, row, epoch, seed, cfg, progress)
        logits, _ = model.protein_prior_logits(graph.node_x, graph.edge_index, graph.edge_x, corruption.input_tokens, corruption.known)
        breakdown = balanced_sequence_loss(logits, graph.sequence, corruption.target_mask, graph.interface, None, None, None, None, "protein", float(lcfg["protein_label_smoothing"]), 0.0)
        return breakdown.total, {**breakdown.detached_scalars(), "mask_fraction": corruption.sampled_fraction, "mask_mode": corruption.mode}

    if stage == Stage.RNA_PRIOR:
        graph = _move_graph(prepared if prepared is not None else load_rna_row(adapter, row), device)
        _augment_graphs(graph, cfg, stage, row.sample_id, epoch)
        corruption = _corrupt(graph, 4, row, epoch, seed, cfg, progress)
        logits, _ = model.rna_prior_logits(graph.node_x, graph.edge_index, graph.edge_x, corruption.input_tokens, corruption.known)
        breakdown = balanced_sequence_loss(None, None, None, None, logits, graph.sequence, corruption.target_mask, graph.interface, "rna", 0.0, float(lcfg["rna_label_smoothing"]))
        return breakdown.total, {**breakdown.detached_scalars(), "mask_fraction": corruption.sampled_fraction, "mask_mode": corruption.mode}

    sample = _move_complex(prepared if prepared is not None else load_complex_row(adapter, row), device)
    _augment_graphs(sample, cfg, stage, row.sample_id, epoch)

    if stage in {Stage.GLOBAL_C, Stage.DELTA_C, Stage.ALPHA}:
        protein_target = _stable_uniform(seed, row.sample_id, epoch, stage.value) < 0.5
        if protein_target:
            ptok, pknown, pmask = _all_interface_corruption(sample.protein)
            rtok = sample.rna.sequence.clone(); rknown = sample.rna.valid.clone().bool(); rmask = torch.zeros_like(sample.rna.valid)
            task = "protein"; target_graph = sample.protein; target_tokens, target_known, target_mask = ptok, pknown, pmask
        else:
            rtok, rknown, rmask = _all_interface_corruption(sample.rna)
            ptok = sample.protein.sequence.clone(); pknown = sample.protein.valid.clone().bool(); pmask = torch.zeros_like(sample.protein.valid)
            task = "rna"; target_graph = sample.rna; target_tokens, target_known, target_mask = rtok, rknown, rmask

        if stage == Stage.ALPHA:
            if task == "protein":
                rknown = hide_known_partner_tokens(rknown, sample.rna.fixed, row.sample_id, epoch, seed, float(mcfg["partner_token_dropout"]))
            else:
                pknown = hide_known_partner_tokens(pknown, sample.protein.fixed, row.sample_id, epoch, seed, float(mcfg["partner_token_dropout"]))

        hard_fraction = float(cfg["training_stages"][stage.value].get("hard_context_fraction_late", 0.0))
        use_hard = progress >= 0.50 and hard_fraction > 0 and _stable_uniform(seed, row.sample_id, epoch, "hard") < hard_fraction
        if use_hard:
            hard_mask = _maybe_hard_mask(model, target_graph, target_tokens, target_known, target_mask, task, fraction=0.50)
            if task == "protein": pmask = hard_mask
            else: rmask = hard_mask

        out = _complex_forward(model, sample, ptok, rtok, pknown, rknown, stage)
        smooth_p = 0.0 if stage == Stage.GLOBAL_C else float(lcfg["protein_label_smoothing"])
        smooth_r = 0.0 if stage == Stage.GLOBAL_C else float(lcfg["rna_label_smoothing"])
        breakdown = balanced_sequence_loss(
            out["protein_logits"] if task == "protein" else None,
            sample.protein.sequence if task == "protein" else None,
            pmask if task == "protein" else None,
            sample.protein.interface if task == "protein" else None,
            out["rna_logits"] if task == "rna" else None,
            sample.rna.sequence if task == "rna" else None,
            rmask if task == "rna" else None,
            sample.rna.interface if task == "rna" else None,
            task, smooth_p, smooth_r,
        )
        loss = breakdown.total
        if stage == Stage.GLOBAL_C and bool(lcfg.get("c_l2_enabled", False)):
            penalty, _ = global_c_regularizer(out["C"], breakdown.total, float(lcfg["c_l2_target_fraction"]))
            loss = loss + penalty
        if stage == Stage.ALPHA and progress < float(lcfg.get("alpha_entropy_warmup_fraction", 0.10)):
            alpha = out["alpha_p"] if task == "protein" else out["alpha_r"]
            groups = sample.pr.protein_index if task == "protein" else sample.pr.rna_index
            n_groups = sample.protein.node_x.shape[0] if task == "protein" else sample.rna.node_x.shape[0]
            loss = loss + float(lcfg["alpha_entropy_warmup_weight"]) * alpha_entropy_regularizer(alpha, groups, n_groups)
        return loss, {**breakdown.detached_scalars(), "task": task, "hard_context": bool(use_hard)}

    schedule = TaskRatioSchedule(tuple(cfg["training_stages"]["joint"]["task_ratio_start"]), tuple(cfg["training_stages"]["joint"]["task_ratio_end"]))
    weights = schedule.weights(progress)
    u = _stable_uniform(seed, row.sample_id, epoch, "joint-task")
    if u < weights["protein_conditional"]: task_name = "protein_conditional"
    elif u < weights["protein_conditional"] + weights["rna_conditional"]: task_name = "rna_conditional"
    else: task_name = "joint"

    if task_name == "protein_conditional":
        pc = _corrupt(sample.protein, 20, row, epoch, seed, cfg, progress)
        ptok, pknown, pmask = pc.input_tokens, pc.known, pc.target_mask
        rtok = sample.rna.sequence.clone()
        rknown = hide_known_partner_tokens(sample.rna.valid.clone().bool(), sample.rna.fixed, row.sample_id, epoch, seed, float(mcfg["partner_token_dropout"]))
        rmask = torch.zeros_like(sample.rna.valid); loss_task = "protein"
    elif task_name == "rna_conditional":
        rc = _corrupt(sample.rna, 4, row, epoch, seed, cfg, progress)
        rtok, rknown, rmask = rc.input_tokens, rc.known, rc.target_mask
        ptok = sample.protein.sequence.clone()
        pknown = hide_known_partner_tokens(sample.protein.valid.clone().bool(), sample.protein.fixed, row.sample_id, epoch, seed, float(mcfg["partner_token_dropout"]))
        pmask = torch.zeros_like(sample.protein.valid); loss_task = "rna"
    else:
        pc = _corrupt(sample.protein, 20, row, epoch, seed, cfg, progress, joint=True)
        rc = _corrupt(sample.rna, 4, row, epoch, seed + 17, cfg, progress, joint=True)
        ptok, pknown, pmask = pc.input_tokens, pc.known, pc.target_mask
        rtok, rknown, rmask = rc.input_tokens, rc.known, rc.target_mask
        loss_task = "joint"

    out = _complex_forward(model, sample, ptok, rtok, pknown, rknown, Stage.JOINT)
    breakdown = balanced_sequence_loss(
        out["protein_logits"] if loss_task in {"protein", "joint"} else None,
        sample.protein.sequence if loss_task in {"protein", "joint"} else None,
        pmask if loss_task in {"protein", "joint"} else None,
        sample.protein.interface if loss_task in {"protein", "joint"} else None,
        out["rna_logits"] if loss_task in {"rna", "joint"} else None,
        sample.rna.sequence if loss_task in {"rna", "joint"} else None,
        rmask if loss_task in {"rna", "joint"} else None,
        sample.rna.interface if loss_task in {"rna", "joint"} else None,
        loss_task, float(lcfg["protein_label_smoothing"]), float(lcfg["rna_label_smoothing"]),
    )
    return breakdown.total, {**breakdown.detached_scalars(), "task": task_name}


def _autocast(cfg: dict, device: torch.device):
    precision = str(cfg["optimization"].get("precision", "fp32")).lower()
    enabled = device.type == "cuda" and precision == "bf16" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def _joint_order(sample: ComplexTensorSample, kind: str, seed: int) -> list[tuple[str, int]]:
    p = [("protein", int(i)) for i in torch.where(sample.protein.valid & ~sample.protein.fixed)[0]]
    r = [("rna", int(i)) for i in torch.where(sample.rna.valid & ~sample.rna.fixed)[0]]
    if kind == "protein_first": return p + r
    if kind == "rna_first": return r + p
    if kind != "mixed": raise ValueError(kind)
    order = p + r
    random.Random(seed).shuffle(order)
    return order


@torch.no_grad()
def sequential_joint_pseudonll(model: JointPriorAndFieldModel, sample: ComplexTensorSample, cfg: dict, seed: int) -> float:
    n_orders = max(3, int(cfg.get("evaluation", {}).get("joint_teacher_forced_orders", 3)))
    kinds = ["mixed", "protein_first", "rna_first"] + ["mixed"] * max(0, n_orders - 3)
    values: list[float] = []
    for oi, kind in enumerate(kinds):
        order = _joint_order(sample, kind, _stable_seed(seed, sample.sample_id, "joint-val", oi))
        pk = sample.protein.fixed & sample.protein.valid
        rk = sample.rna.fixed & sample.rna.valid
        pt = sample.protein.sequence.clone(); rt = sample.rna.sequence.clone()
        p_losses: list[float] = []; r_losses: list[float] = []
        for polymer, index in order:
            out = _complex_forward(model, sample, pt, rt, pk, rk, Stage.JOINT)
            if polymer == "protein":
                logp = F.log_softmax(out["protein_logits"][index].float(), dim=-1)
                p_losses.append(float((-logp[int(pt[index].item())] / math.log(20.0)).cpu()))
                pk[index] = True
            else:
                logp = F.log_softmax(out["rna_logits"][index].float(), dim=-1)
                r_losses.append(float((-logp[int(rt[index].item())] / math.log(4.0)).cpu()))
                rk[index] = True
        if not p_losses or not r_losses:
            raise ValueError(f"Joint validation requires designable positions on both polymers: {sample.sample_id}")
        values.append(0.5 * (float(np.mean(p_losses)) + float(np.mean(r_losses))))
    return float(np.mean(values))


@torch.no_grad()
def validate_stage(
    model: JointPriorAndFieldModel,
    stage: Stage,
    manifest: ManifestTable,
    cfg: dict,
    device: torch.device,
    process_pool: ProcessPoolExecutor | None = None,
) -> float:
    model.eval()
    adapter = _adapter(cfg, 0, training=False)
    rows = list(manifest.rows())
    workers = int(cfg.get("optimization", {}).get("data_workers", 0))
    backend = str(cfg.get("optimization", {}).get("data_prefetch_backend", "thread"))
    if not rows:
        raise ValueError("Empty validation manifest")

    if stage in {Stage.PROTEIN_PRIOR, Stage.RNA_PRIOR}:
        values = []
        for row, graph in _prefetch_training_samples(rows, adapter, stage, workers, process_pool, backend):
            with _autocast(cfg, device):
                if stage == Stage.PROTEIN_PRIOR:
                    graph = _move_graph(graph, device)
                    known = graph.fixed & graph.valid
                    logits, _ = model.protein_prior_logits(graph.node_x, graph.edge_index, graph.edge_x, graph.sequence, known)
                    b = balanced_sequence_loss(logits, graph.sequence, graph.valid & ~graph.fixed, graph.interface, None, None, None, None, "protein", 0.0, 0.0)
                else:
                    graph = _move_graph(graph, device)
                    known = graph.fixed & graph.valid
                    logits, _ = model.rna_prior_logits(graph.node_x, graph.edge_index, graph.edge_x, graph.sequence, known)
                    b = balanced_sequence_loss(None, None, None, None, logits, graph.sequence, graph.valid & ~graph.fixed, graph.interface, "rna", 0.0, 0.0)
            values.append(float(b.total))
        return float(np.mean(values))

    p_values: list[float] = []; r_values: list[float] = []; joint_values: list[float] = []
    joint_limit = max(1, int(cfg.get("evaluation", {}).get("joint_validation_max_complexes", 20)))
    seed = int(cfg["experiment"]["pilot_seed"])
    joint_ids = {row.sample_id for row in sorted(rows, key=lambda x: _stable_seed(seed, x.sample_id, "joint-validation-subset"))[:joint_limit]}

    for row, sample in _prefetch_training_samples(rows, adapter, stage, workers, process_pool, backend):
        with _autocast(cfg, device):
            sample = _move_complex(sample, device)
            ptok, pknown, pmask = _all_interface_corruption(sample.protein)
            rtok, rknown, rmask = _all_interface_corruption(sample.rna)
            use_delta, learned_alpha = _stage_flags(stage)
            outp = model(sample.protein.node_x, sample.protein.edge_index, sample.protein.edge_x, sample.rna.node_x, sample.rna.edge_index, sample.rna.edge_x, sample.pr, ptok, sample.rna.sequence, pknown, sample.rna.valid, use_delta=use_delta, learned_alpha=learned_alpha)
            bp = balanced_sequence_loss(outp["protein_logits"], sample.protein.sequence, pmask, sample.protein.interface, None, None, None, None, "protein", 0.0, 0.0)
            outr = model(sample.protein.node_x, sample.protein.edge_index, sample.protein.edge_x, sample.rna.node_x, sample.rna.edge_index, sample.rna.edge_x, sample.pr, sample.protein.sequence, rtok, sample.protein.valid, rknown, use_delta=use_delta, learned_alpha=learned_alpha)
            br = balanced_sequence_loss(None, None, None, None, outr["rna_logits"], sample.rna.sequence, rmask, sample.rna.interface, "rna", 0.0, 0.0)
        p_values.append(float(bp.total)); r_values.append(float(br.total))
        if stage == Stage.JOINT and row.sample_id in joint_ids:
            joint_values.append(sequential_joint_pseudonll(model, sample, cfg, seed))

    if stage == Stage.JOINT:
        if not joint_values: raise RuntimeError("No sequential joint-validation values were computed")
        return float((np.mean(p_values) + np.mean(r_values) + np.mean(joint_values)) / 3.0)
    return float((np.mean(p_values) + np.mean(r_values)) / 2.0)


def _cosine_schedule(optimizer: torch.optim.Optimizer, step: int, total: int, warmup_fraction: float, base_lrs: list[float]) -> None:
    warm = max(1, int(total * warmup_fraction))
    if step < warm: scale = (step + 1) / warm
    else:
        x = (step - warm) / max(1, total - warm)
        scale = 0.5 * (1.0 + math.cos(math.pi * min(max(x, 0.0), 1.0)))
    for group, base in zip(optimizer.param_groups, base_lrs): group["lr"] = base * scale


def train_stage(cfg: dict, stage: Stage, train_manifest: Path, val_manifest: Path, out_dir: Path, init_checkpoint: Path | None = None, device: str | None = None) -> Path:
    seed = int(cfg["experiment"]["pilot_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model_from_config(cfg).to(dev)
    if init_checkpoint is not None:
        payload = torch.load(init_checkpoint, map_location="cpu"); model.load_state_dict(payload["model"])
    configure_stage(model, stage)
    joint_mode = _joint_unfreezing_mode(cfg) if stage == Stage.JOINT else "not_applicable"
    if stage == Stage.JOINT and init_checkpoint is None:
        joint_mode = "all_trainable_from_start"
    if stage == Stage.JOINT and joint_mode == "all_trainable_from_start":
        make_joint_fully_trainable(model, include_global_c=True)

    optcfg = cfg["optimization"]
    optimizer = build_optimizer(model, stage, float(optcfg["lr_heads"]), float(optcfg["lr_projections"]), float(optcfg["lr_encoder_top"]), float(optcfg["lr_encoder_bottom"]), float(optcfg["lr_global_c_joint"]), float(optcfg["weight_decay"]), float(optcfg["layerwise_lr_decay"]))
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    train_table = ManifestTable(train_manifest); val_table = ManifestTable(val_manifest)
    max_epochs = int(cfg["training_stages"][stage.value].get("max_epochs", cfg["optimization"].get("max_epochs_default", 100)))
    patience = int(optcfg["early_stopping_patience"]); total_steps = max_epochs * max(1, len(train_table))
    out_dir.mkdir(parents=True, exist_ok=True); metrics_path = out_dir / "metrics.jsonl"; best_path = out_dir / "best.pt"
    best = float("inf"); bad_epochs = 0; global_step = 0
    workers = int(optcfg.get("data_workers", 0))
    backend = str(optcfg.get("data_prefetch_backend", "thread"))
    process_pool = None
    if workers > 0 and backend == "process":
        # Keep one pool alive for the complete stage.  Recreating twelve
        # interpreter processes every epoch would erase much of the gain.
        process_pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))

    try:
        for epoch in range(max_epochs):
            model.train(); progress = epoch / max(1, max_epochs - 1)
            if stage == Stage.JOINT and joint_mode == "gradual": apply_joint_unfreezing(model, progress)
            adapter = _adapter(cfg, epoch, training=True)
            rows = list(train_table.rows()); random.Random(seed + epoch * 104729).shuffle(rows)
            running = []; mask_fractions = []; hard_count = 0; task_counts: dict[str, int] = {}
            for row, prepared in _prefetch_training_samples(rows, adapter, stage, workers, process_pool, backend):
                optimizer.zero_grad(set_to_none=True)
                with _autocast(cfg, dev): loss, detail = _one_training_loss(model, row, adapter, stage, cfg, epoch, progress, dev, prepared)
                if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite loss at {stage.value} epoch={epoch} sample={row.sample_id}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad and p.grad is not None], float(optcfg["grad_clip_norm"]))
                optimizer.step(); global_step += 1
                _cosine_schedule(optimizer, global_step, total_steps, float(optcfg["warmup_fraction"]), base_lrs)
                running.append(float(loss.detach().cpu()))
                if "mask_fraction" in detail: mask_fractions.append(float(detail["mask_fraction"]))
                if detail.get("hard_context"): hard_count += 1
                if "task" in detail: task_counts[str(detail["task"])] = task_counts.get(str(detail["task"]), 0) + 1

            val = validate_stage(model, stage, val_table, cfg, dev, process_pool)
            record = {"stage": stage.value, "epoch": epoch, "train_loss": float(np.mean(running)), "val_metric": val, "mean_mask_fraction": float(np.mean(mask_fractions)) if mask_fractions else None, "hard_context_samples": hard_count, "task_counts": task_counts, "joint_unfreezing_mode": joint_mode, "trainable": trainable_parameter_report(model), "precision": "bf16" if dev.type == "cuda" and str(optcfg.get("precision", "")).lower() == "bf16" and torch.cuda.is_bf16_supported() else "fp32"}
            with metrics_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, sort_keys=True) + "\n")
            if val < best - 1e-6:
                best = val; bad_epochs = 0
                torch.save({"model": model.state_dict(), "stage": stage.value, "epoch": epoch, "val_metric": val, "joint_unfreezing_mode": joint_mode, "config": cfg}, best_path)
            else:
                bad_epochs += 1
                if bad_epochs >= patience: break
    finally:
        if process_pool is not None:
            process_pool.shutdown(wait=True)
    if not best_path.exists(): raise RuntimeError("No checkpoint was saved")
    return best_path
