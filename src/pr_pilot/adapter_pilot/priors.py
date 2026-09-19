"""Thin, state-dict-compatible wrappers around the pinned MPNN priors."""
from __future__ import annotations

import hashlib
import importlib.util
import copy
import sys
from pathlib import Path
from typing import Sequence

import gemmi
import numpy as np
import torch
import torch.nn.functional as F

from pr_pilot.data.residue_vocab import PROTEIN_ALPHABET as PROJECT_PROTEIN_ALPHABET, RNA_ALPHABET as PROJECT_RNA_ALPHABET, classify_residue

PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
RNA_ALPHABET = "AUGC"
RNA_MODEL_COLUMNS = ("DA", "DT", "DG", "DC")  # project order A/U/G/C
PDB_CHAIN_IDS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
UPSTREAM_PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
PROTEIN_CANONICAL_NAMES = dict(zip(PROJECT_PROTEIN_ALPHABET, ("ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR")))
RNA_CANONICAL_NAMES = {token: token for token in PROJECT_RNA_ALPHABET}


def view_chain_ids(chains: Sequence[str]) -> list[str]:
    """Return consecutive one-character ids preserving the supplied order.

    ProteinMPNN's featurizer sorts chain ids.  Consecutive ids make that sort
    identical to the source-structure order even when source ids are I/J/F or
    contain mmCIF suffixes.
    """
    if len(set(chains)) != len(chains):
        raise ValueError(f"duplicate selected chain ids: {chains}")
    if len(chains) > len(PDB_CHAIN_IDS):
        raise ValueError(f"too many chains for one-character PDB view: {len(chains)}")
    return list(PDB_CHAIN_IDS[: len(chains)])


def source_chain_order(structure_path: Path, chains: Sequence[str]) -> list[str]:
    """Selected chain ids in the exact order stored by the source structure."""
    structure = gemmi.read_structure(str(structure_path))
    if len(structure) == 0:
        raise ValueError(f"no model in {structure_path}")
    keep = {str(value) for value in chains}
    order = [str(chain.name) for chain in structure[0] if str(chain.name) in keep]
    if set(order) != keep:
        raise ValueError(f"selected chains missing from {structure_path}: expected={sorted(keep)} found={order}")
    return order


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load upstream module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_pdb_view(structure_path: Path, sample_id: str, chains: Sequence[str], polymer: str, root: Path) -> Path:
    """Create a deterministic single-polymer PDB view from the source CIF."""
    structure_path = Path(structure_path)
    key = hashlib.sha256(("pdb_view_v6|" + str(structure_path.resolve()) + "|" + ",".join(chains) + "|" + polymer).encode()).hexdigest()[:16]
    out = Path(root) / polymer / f"{key}_{sample_id.replace('/', '_')}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    structure = gemmi.read_structure(str(structure_path))
    if len(structure) == 0:
        raise ValueError(f"no model in {structure_path}")
    keep = {str(x) for x in chains}
    order = source_chain_order(structure_path, chains)
    mapped = dict(zip(order, view_chain_ids(order)))
    model = structure[0]
    # Gemmi's remove_chain is PDB-name-oriented and can leave mmCIF ids such
    # as ``B-2`` behind.  Deep-copy selected chains, clear the model by index,
    # and append them in source order so upstream residue order agrees with
    # the runtime records even when the source CIF stores chains differently.
    selected = {str(chain.name): copy.deepcopy(chain) for chain in model if str(chain.name) in keep}
    for index in range(len(model) - 1, -1, -1):
        del model[index]
    for original in order:
        chain = selected[str(original)]
        chain.name = mapped[str(original)]
        # Modified residues such as MSE are valid project residues but are
        # written as HETATM by Gemmi.  The pinned PDB parsers intentionally
        # read polymer ATOM records, so normalize only this selected view.
        for residue in chain:
            cls = classify_residue(str(residue.name))
            if polymer == "protein" and cls.polymer == "protein" and cls.token in PROTEIN_CANONICAL_NAMES:
                residue.name = PROTEIN_CANONICAL_NAMES[cls.token]
                residue.entity_type = gemmi.EntityType.Polymer
                residue.het_flag = " "
            elif polymer == "rna" and cls.polymer == "rna" and cls.token in RNA_CANONICAL_NAMES:
                residue.name = RNA_CANONICAL_NAMES[cls.token]
                residue.entity_type = gemmi.EntityType.Polymer
                residue.het_flag = " "
        model.add_chain(chain)
    if len(model) == 0:
        raise ValueError(f"no selected chains {sorted(keep)} in {structure_path}")
    structure.write_pdb(str(out))
    return out


def _protein_features(module, pdb: Path, chains: Sequence[str], device: torch.device):
    parsed = module.parse_PDB(str(pdb), input_chain_list=list(chains), ca_only=False)
    if len(parsed) != 1:
        raise ValueError(f"ProteinMPNN parser returned {len(parsed)} structures for {pdb}")
    entry = parsed[0]
    chain_dict = {entry["name"]: (list(chains), [])}
    tensors = module.tied_featurize([entry], device, chain_dict, ca_only=False)
    return entry, tensors


def _protein_encoded(model, tensors):
    X, _S, mask, _lengths, _chain_M, chain_encoding, *_ = tensors
    E, E_idx = model.features(X, mask, tensors[12], chain_encoding)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = model.W_e(E)
    mask_attend = model._pilot_gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1) if hasattr(model, "_pilot_gather_nodes") else None
    if mask_attend is None:
        neighbors_flat = E_idx.reshape((E_idx.shape[0], -1)).unsqueeze(-1).expand(-1, -1, mask.shape[2] if mask.ndim == 3 else 1)
        # This branch is not used; the upstream gather helper is attached by
        # the wrapper at construction time to avoid modifying upstream files.
        del neighbors_flat
        raise RuntimeError("upstream gather helper was not attached")
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
    h_EX_encoder = model._pilot_cat_neighbors(torch.zeros_like(h_V), h_E, E_idx)
    h_EXV_encoder = model._pilot_cat_neighbors(h_V, h_EX_encoder, E_idx)
    mask_1d = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_fw = mask_1d
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder
    for layer in model.decoder_layers:
        h_V = layer(h_V, h_EXV_encoder_fw, mask)
    return h_V, model.W_out(h_V), tensors


def _protein_structure_encoded(model, tensors):
    """Return the ProteinMPNN encoder output before any decoder layer.

    This is the representation allowed to enter the explicit interaction
    generator.  It depends on backbone coordinates, residue indices, and chain
    encoding, but never on native sequence tokens or teacher-forced decoder
    states.
    """
    X, _S, mask, _lengths, _chain_M, chain_encoding, *_ = tensors
    E, E_idx = model.features(X, mask, tensors[12], chain_encoding)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = model.W_e(E)
    mask_attend = model._pilot_gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
    return h_V, tensors


class ProteinMPNNPrior:
    """Official ProteinMPNN with its parameters frozen and untouched."""

    def __init__(self, checkout: Path, checkpoint: Path, device: torch.device):
        self.checkout = Path(checkout)
        self.device = device
        self.module = _load_module("pilot_protein_mpnn_utils", self.checkout / "protein_mpnn_utils.py")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hidden = 128
        self.model = self.module.ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=hidden,
            edge_features=hidden,
            hidden_dim=hidden,
            num_encoder_layers=3,
            num_decoder_layers=3,
            # Training noise is metadata, not inference noise.  The official
            # inference command constructs this model with backbone_noise=0.
            augment_eps=0.0,
            k_neighbors=int(payload["num_edges"]),
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(device).eval()
        self.model._pilot_gather_nodes = self.module.gather_nodes
        self.model._pilot_cat_neighbors = self.module.cat_neighbors_nodes
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode_backbone(self, pdb: Path, chains: Sequence[str]) -> dict[str, torch.Tensor | list[str]]:
        entry, tensors = _protein_features(self.module, pdb, chains, self.device)
        h_v, logits, tensors = _protein_encoded(self.model, tensors)
        # Match the historical external evaluator: remove the unknown X column
        # and renormalize over the 20 canonical amino acids.
        base = F.log_softmax(logits[..., :20], dim=-1).squeeze(0)
        valid = tensors[2].squeeze(0).bool()
        tokens = tensors[1].squeeze(0).long()[valid]
        sequence = "".join(UPSTREAM_PROTEIN_ALPHABET[int(token)] for token in tokens)
        return {
            "hidden": h_v.squeeze(0)[valid],
            "logits": logits.squeeze(0)[valid],
            "log_probs": base[valid],
            "tokens": tokens,
            "sequence": sequence,
            "length": int(valid.sum().item()),
        }

    @torch.no_grad()
    def encode_structure_only(self, pdb: Path, chains: Sequence[str]) -> dict[str, torch.Tensor | list[str]]:
        """Encode only the sequence-free ProteinMPNN encoder representation."""
        _entry, tensors = _protein_features(self.module, pdb, chains, self.device)
        h_v, tensors = _protein_structure_encoded(self.model, tensors)
        valid = tensors[2].squeeze(0).bool()
        return {"hidden": h_v.squeeze(0)[valid], "length": int(valid.sum().item())}

    @torch.no_grad()
    def forward_prior(self, pdb: Path, chains: Sequence[str]) -> torch.Tensor:
        return self.encode_backbone(pdb, chains)["log_probs"]

    @torch.no_grad()
    def sample_prior(self, *args, **kwargs):
        """Expose the unmodified upstream sampling routine for final inference."""
        return self.model.sample(*args, **kwargs)

    @torch.no_grad()
    def forward_with_residual(self, pdb: Path, chains: Sequence[str], residual: torch.Tensor) -> torch.Tensor:
        base = self.forward_prior(pdb, chains)
        if tuple(residual.shape) != tuple(base.shape):
            raise ValueError(f"residual shape {tuple(residual.shape)} != {tuple(base.shape)}")
        return base + residual


def _na_config():
    atom_types = [
        "N", "CA", "C", "O", "OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    ]
    atom_dict = dict(zip(atom_types, range(len(atom_types))))
    polytypes = ["PP", "DNA", "RNA", "UNK", "MAS", "PAD"]
    polytype_to_int = dict(zip(polytypes, range(len(polytypes))))
    restypes = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "UNK",
        "DA", "DC", "DG", "DT", "DX", "A", "C", "G", "U", "RX", "MAS", "PAD",
    ]
    restype_to_int = dict(zip(restypes, range(len(restypes))))
    for rna, dna in (("A", "DA"), ("C", "DC"), ("G", "DG"), ("U", "DT"), ("RX", "DX")):
        restype_to_int[rna] = restype_to_int[dna]
    return atom_dict, polytype_to_int, restype_to_int


class NAMPrior:
    """Official NA-MPNN RNA-only branch, reduced to canonical AUGC logits."""

    def __init__(self, checkout: Path, checkpoint: Path, device: torch.device):
        self.checkout = Path(checkout)
        self.device = device
        self.module = _load_module("pilot_na_mpnn_model_utils", self.checkout / "inference" / "model_utils.py")
        self.data = _load_module("pilot_na_mpnn_data_utils", self.checkout / "inference" / "data_utils.py")
        atom_dict, polytype_to_int, restype_to_int = _na_config()
        self.restype_to_int = restype_to_int
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = self.module.ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=32,
            model_type="na_mpnn",
            vocab=33,
            num_letters=33,
            atom_dict=atom_dict,
            restype_to_int=restype_to_int,
            polytype_to_int=polytype_to_int,
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _features(self, pdb: Path, chains: Sequence[str]):
        parsed, _backbone, _other, icodes, _water = self.data.parse_PDB(
            str(pdb), device=str(self.device), chains=list(chains), parse_na_only=True,
            model_type="na_mpnn", na_shared_tokens=True,
        )
        parsed["chain_mask"] = parsed["rna_mask"].clone()
        features = self.data.featurize(parsed)
        features["batch_size"] = 1
        keys = [f"{chain}:{int(index)}:{str(icode).strip()}" for chain, index, icode in zip(parsed["chain_letters"], parsed["R_idx"].tolist(), icodes)]
        return parsed, features, keys

    @torch.no_grad()
    def encode_backbone(self, pdb: Path, chains: Sequence[str]) -> dict[str, torch.Tensor | list[str]]:
        parsed, features, keys = self._features(pdb, chains)
        h_v, _h_e, _e_idx = self.model.encode(features)
        output = self.model.unconditional_probs(features)
        indices = torch.tensor([self.restype_to_int[name] for name in RNA_MODEL_COLUMNS], device=self.device)
        four = output["log_probs"][0, :, indices]
        four = four - torch.logsumexp(four, dim=-1, keepdim=True)
        rna_mask = parsed["rna_mask"].bool()
        return {
            "hidden": h_v.squeeze(0)[rna_mask],
            "logits": output["log_probs"].squeeze(0)[rna_mask],
            "log_probs": four[rna_mask],
            "tokens": parsed["S"][rna_mask].long(),
            "residue_keys": [key for key, keep in zip(keys, rna_mask.tolist()) if keep],
            "length": int(rna_mask.sum().item()),
        }

    @torch.no_grad()
    def forward_prior(self, pdb: Path, chains: Sequence[str]) -> torch.Tensor:
        return self.encode_backbone(pdb, chains)["log_probs"]

    @torch.no_grad()
    def sample_prior(self, feature_dict: dict):
        return self.model.sample(feature_dict)

    @torch.no_grad()
    def forward_with_residual(self, pdb: Path, chains: Sequence[str], residual: torch.Tensor) -> torch.Tensor:
        base = self.forward_prior(pdb, chains)
        if tuple(residual.shape) != tuple(base.shape):
            raise ValueError(f"residual shape {tuple(residual.shape)} != {tuple(base.shape)}")
        return base + residual
