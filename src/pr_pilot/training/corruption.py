"""Deterministic corruption/masking used by prior and joint training."""
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


def _rng(seed: int, sample_id: str, epoch: int, tag: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{sample_id}|{epoch}|{tag}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little") % (2**32))


def _weighted_random_indices(rng: np.random.Generator, eligible: np.ndarray, interface: np.ndarray, n: int, interface_multiplier: float) -> np.ndarray:
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
) -> Corruption:
    """Create a reproducible target mask and corrupted input sequence.

    Local-patch corruption alternates between sequence-local and 3D-nearest patches.
    Fixed or invalid positions are never training targets.
    """
    graph.validate()
    rng = _rng(seed, sample_id, epoch, "corrupt")
    designable = (graph.valid & ~graph.fixed).cpu().numpy().astype(bool)
    eligible = np.flatnonzero(designable)
    if len(eligible) == 0:
        raise ValueError("No designable positions")
    frac = float(force_fraction) if force_fraction is not None else float(rng.uniform(min_fraction, max_fraction))
    frac = min(max(frac, 0.0), 1.0)
    n_target = max(1, int(round(frac * len(eligible))))
    interface = graph.interface.cpu().numpy().astype(bool)

    mode = rng.random()
    if mode < random_mask_probability or local_patch_probability <= 0:
        target = _weighted_random_indices(rng, eligible, interface, n_target, interface_multiplier)
    else:
        center = int(rng.choice(eligible))
        if rng.random() < 0.5:
            chain = int(graph.chain_index[center])
            same = eligible[graph.chain_index[eligible].cpu().numpy() == chain]
            # residue_ids are ordered; index distance is a robust contiguous proxy.
            target = same[np.argsort(np.abs(same - center))[:n_target]]
            if len(target) < n_target:
                extra_pool = np.setdiff1d(eligible, target)
                extra = _weighted_random_indices(rng, extra_pool, interface, n_target - len(target), interface_multiplier)
                target = np.concatenate([target, extra])
        else:
            xyz = graph.reference_xyz.cpu().numpy()
            d = np.linalg.norm(xyz[eligible] - xyz[center], axis=1)
            target = eligible[np.argsort(d)[:n_target]]

    target_mask = torch.zeros_like(graph.valid, dtype=torch.bool)
    target_mask[torch.as_tensor(target, dtype=torch.long)] = True
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

    # Fixed positions are always visible and never targets.
    target_mask[graph.fixed] = False
    known[graph.fixed & graph.valid] = True
    input_tokens[graph.fixed & graph.valid] = graph.sequence[graph.fixed & graph.valid]
    return Corruption(input_tokens, known, target_mask, wrong_token_mask)


def hide_known_partner_tokens(known: Tensor, fixed: Tensor, sample_id: str, epoch: int, seed: int, fraction: float) -> Tensor:
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
            out[torch.as_tensor(chosen, dtype=torch.long)] = False
    return out
