"""Featurisation and a NumPy forward pass for the small message-passing network."""
from __future__ import annotations

import numpy as np
from rdkit import Chem

ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53]
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
]
N_LAYERS = 3
FEATURE_DIM = len(ELEMENTS) + 1 + 6 + 3 + 1 + 3 + 4 + 1


def one_hot(value, choices) -> list[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def atom_features(atom: Chem.Atom) -> list[float]:
    element = atom.GetAtomicNum()
    features = one_hot(element, ELEMENTS) + [0.0 if element in ELEMENTS else 1.0]
    features += one_hot(min(atom.GetDegree(), 5), range(6))
    features += one_hot(atom.GetFormalCharge(), (-1, 0, 1))
    features += [float(atom.GetIsAromatic())]
    features += one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    features += one_hot(min(atom.GetTotalNumHs(), 3), range(4))
    features += [float(atom.IsInRing())]
    return features


def featurize(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [(i, j), (j, i)]
    edge_index = np.array(edges, dtype=np.int64).reshape(-1, 2).T
    return x, edge_index


def neighbour_mean(h: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    total = np.zeros_like(h)
    np.add.at(total, dst, h[src])
    degree = np.bincount(dst, minlength=len(h)).clip(min=1)[:, None]
    return total / degree


def forward(weights: dict[str, np.ndarray], x: np.ndarray, edge_index: np.ndarray):
    """Return (prediction in label units, per-layer atom embeddings)."""
    h = x @ weights["embed_w"] + weights["embed_b"]
    layers = [h]
    for k in range(N_LAYERS):
        h = np.maximum(
            h @ weights[f"self_w{k}"] + neighbour_mean(h, edge_index) @ weights[f"nbr_w{k}"] + weights[f"bias{k}"],
            0.0,
        )
        layers.append(h)
    pooled = np.concatenate([h.mean(axis=0), h.max(axis=0)])
    hidden = np.maximum(pooled @ weights["out_w0"] + weights["out_b0"], 0.0)
    scaled = float(hidden @ weights["out_w1"] + weights["out_b1"])
    return scaled * float(weights["y_std"]) + float(weights["y_mean"]), layers
