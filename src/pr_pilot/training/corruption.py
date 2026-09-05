"""Deterministic corruption/masking used by prior and joint training.

The masking policy implements the agreed light curriculum rather than sampling
10--100% uniformly from the first epoch:
- early: ~10--40%;
- middle: ~20--70%;
- late: configured full range (normally 10--100%);
- explicit full-mask examples are injected late with configurable probability.
Random and local/spatial patch masks share the same target-count schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import Tensor

from pr_pilot.runtime.dataset_adapter import PolymerGraph


@dataclass
class Corruption:
    input_tokens: Tensor
    known: Tensor
    target_mask: Tensor
    wrong_token_mask: Tensor
    sampled_fraction: float
    mode: str


def _rng(seed: int, sample_id: str, epoch: int, tag: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{sample_id}|{epoch}|{tag}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))


def curriculum_bounds(progress: float, min_fraction: float, max_fraction: float) -> tuple[float, float]:
    """Three-band light masking curriculum from easy completion to full design."""
    progress = min(max(float(progress), 0.0), 1.0)
    if progress < 0.30:
        lo = min_fraction
        hi = min(max_fraction, 0.40)
    elif progress < 0.70:
        lo = max(min_fraction, 0.20)
        hi = min(max_fraction, 0.70)
    else:
        lo = min_fraction
        hi = max_fraction
    if hi < lo:
        hi = lo
    return float(lo), float(hi)


def _weighted_random_indices(
    rng: np.random.Generator,
    eligible: np.ndarray,
    interface: np.ndarray,
    n: int,
    interface_multiplier: float,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    weights = np.ones(len(eligible), dtype=np.float64)
    weights[interface[eligible]] *= max(float(interface_multiplier), 0.0)
    weights /= weights.sum()
    return rng.choice(eligible, size=min(n, len(eligible)), replace=False, p=weights)


def generate_corruption(
    graph: PolymerGraph,
    alphabet_size: int,
    sample_id: str,
    epoch: int,
    seed: int,
    min_fraction: float = 0.10,
    max_fraction: float = 1.00,
    random_mask_probability: float = 0.70,
    local_patch_probability: float = 0.30,
    wrong_token_fraction: float = 0.10,
    interface_multiplier: float = 2.0,
    force_fraction: float | None = None,
    *,
    progress: float = 1.0,
    use_curriculum: bool = True,
    full_mask_probability: float = 0.15,
) -> Corruption:
    """Create a reproducible target mask and corrupted input sequence.

    Fixed/invalid positions are never targets. ``wrong_token_fraction`` converts
    a subset of targets into *visible incorrect tokens*; these positions remain
    prediction targets, which teaches recovery from imperfect intermediate design
    states instead of leaking the native target.
    """
    graph.validate()
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be >=2")
    if not 0 <= full_mask_probability <= 1:
        raise ValueError("full_mask_probability must be in [0,1]")
    rng = _rng(seed, sample_id, epoch, "corrupt")
    designable = (graph.valid & ~graph.fixed).cpu().numpy().astype(bool)
    eligible = np.flatnonzero(designable)
    if len(eligible) == 0:
        raise ValueError("No designable positions")

    if force_fraction is not None:
        frac = float(force_fraction)
        mode_fraction = "forced"
    else:
        lo, hi = curriculum_bounds(progress, min_fraction, max_fraction) if use_curriculum else (min_fraction, max_fraction)
        late_full_prob = full_mask_probability if progress >= 0.70 and max_fraction >= 1.0 else 0.0
        if rng.random() < late_full_prob:
            frac = 1.0
            mode_fraction = "full"
        else:
            frac = float(rng.uniform(lo, hi))
            mode_fraction = "sampled"
    frac = min(max(frac, 0.0), 1.0)
    n_target = max(1, int(round(frac * len(eligible))))
    interface = graph.interface.cpu().numpy().astype(bool)

    mode_draw = rng.random()
    if mode_draw < random_mask_probability or local_patch_probability <= 0:
        target = _weighted_random_indices(rng, eligible, interface, n_target, interface_multiplier)
        mask_mode = "random"
    else:
        center = int(rng.choice(eligible))
        if rng.random() < 0.5:
            chain = int(graph.chain_index[center])
            same = eligible[graph.chain_index[eligible].cpu().numpy() == chain]
            target = same[np.argsort(np.abs(same - center))[:n_target]]
            if len(target) < n_target:
                extra_pool = np.setdiff1d(eligible, target)
                extra = _weighted_random_indices(
                    rng,
                    extra_pool,
                    interface,
                    n_target - len(target),
                    interface_multiplier,
                )
                target = np.concatenate([target, extra])
            mask_mode = "sequence_patch"
        else:
            xyz = graph.reference_xyz.cpu().numpy()
            d = np.linalg.norm(xyz[eligible] - xyz[center], axis=1)
            target = eligible[np.argsort(d)[:n_target]]
            mask_mode = "spatial_patch"

    target_mask = torch.zeros_like(graph.valid, dtype=torch.bool)
    target_mask[torch.as_tensor(target, dtype=torch.long, device=target_mask.device)] = True
    input_tokens = graph.sequence.clone()
    known = graph.valid.clone().bool()
    known[target_mask] = False

    wrong_token_mask = torch.zeros_like(target_mask)
    n_wrong = int(round(float(wrong_token_fraction) * int(target_mask.sum())))
    if n_wrong > 0:
        wrong_idx = rng.choice(target, size=min(n_wrong, len(target)), replace=False)
        for idx in wrong_idx:
            native = int(graph.sequence[idx])
            choices = [x for x in range(alphabet_size) if x != native]
            input_tokens[idx] = int(rng.choice(choices))
            known[idx] = True
            wrong_token_mask[idx] = True

    target_mask[graph.fixed] = False
    known[graph.fixed & graph.valid] = True
    input_tokens[graph.fixed & graph.valid] = graph.sequence[graph.fixed & graph.valid]
    actual_fraction = float(target_mask.sum().item()) / max(1, int((graph.valid & ~graph.fixed).sum().item()))
    return Corruption(
        input_tokens=input_tokens,
        known=known,
        target_mask=target_mask,
        wrong_token_mask=wrong_token_mask,
        sampled_fraction=actual_fraction,
        mode=f"{mode_fraction}:{mask_mode}",
    )


def hide_known_partner_tokens(
    known: Tensor,
    fixed: Tensor,
    sample_id: str,
    epoch: int,
    seed: int,
    fraction: float,
) -> Tensor:
    """Randomly hide a fraction of otherwise-known partner tokens."""
    if fraction <= 0:
        return known.clone()
    rng = _rng(seed, sample_id, epoch, "partner-hide")
    candidates = torch.where(known & ~fixed)[0].cpu().numpy()
    out = known.clone()
    if len(candidates):
        n = min(len(candidates), int(round(len(candidates) * fraction)))
        if n:
            chosen = rng.choice(candidates, size=n, replace=False)
            out[torch.as_tensor(chosen, dtype=torch.long, device=out.device)] = False
    return out
