"""Train the message-passing network on the MoleculeNet Lipophilicity set (experimental logD 7.4)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch import nn

from gnn import FEATURE_DIM, N_LAYERS, featurize, forward

ROOT = Path(__file__).resolve().parent.parent
HIDDEN = 64
EPOCHS = 120
BATCH = 64
SEED = 0


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Linear(FEATURE_DIM, HIDDEN)
        self.self_lin = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(N_LAYERS))
        self.nbr_lin = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(N_LAYERS))
        self.bias = nn.ParameterList(nn.Parameter(torch.zeros(HIDDEN)) for _ in range(N_LAYERS))
        self.out0 = nn.Linear(2 * HIDDEN, HIDDEN)
        self.out1 = nn.Linear(HIDDEN, 1)

    def forward(self, x, edge_index, batch, n_graphs):
        h = self.embed(x)
        src, dst = edge_index
        degree = torch.zeros(len(x)).index_add_(0, dst, torch.ones(len(dst))).clamp(min=1).unsqueeze(1)
        for k in range(N_LAYERS):
            agg = torch.zeros_like(h).index_add_(0, dst, h[src]) / degree
            h = torch.relu(self.self_lin[k](h) + self.nbr_lin[k](agg) + self.bias[k])
        counts = torch.zeros(n_graphs).index_add_(0, batch, torch.ones(len(batch))).unsqueeze(1)
        mean = torch.zeros(n_graphs, HIDDEN).index_add_(0, batch, h) / counts
        peak = torch.full((n_graphs, HIDDEN), -1e9).scatter_reduce(
            0, batch.unsqueeze(1).expand_as(h), h, reduce="amax", include_self=True
        )
        return self.out1(torch.relu(self.out0(torch.cat([mean, peak], dim=1)))).squeeze(1)


def collate(graphs, idx):
    xs, edges, batch, offset = [], [], [], 0
    for g, i in enumerate(idx):
        x, e = graphs[i]
        xs.append(x)
        edges.append(e + offset)
        batch.append(np.full(len(x), g))
        offset += len(x)
    return (
        torch.tensor(np.concatenate(xs)),
        torch.tensor(np.concatenate(edges, axis=1)),
        torch.tensor(np.concatenate(batch)),
        len(idx),
    )


def predict(net, graphs, idx):
    net.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(idx), 256):
            out.append(net(*collate(graphs, idx[start : start + 256])).numpy())
    return np.concatenate(out)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    frame = pd.read_csv(ROOT / "data" / "Lipophilicity.csv")
    graphs, labels = [], []
    for smiles, y in zip(frame["smiles"], frame["exp"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumBonds() > 0:
            graphs.append(featurize(mol))
            labels.append(y)
    y = np.array(labels, dtype=np.float32)
    order = rng.permutation(len(y))
    n_test = n_val = len(y) // 10
    test, val, train = order[:n_test], order[n_test : n_test + n_val], order[n_test + n_val :]
    y_mean, y_std = float(y[train].mean()), float(y[train].std())
    target = torch.tensor((y - y_mean) / y_std)

    net = Net()
    optimiser = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS)
    best, best_state = float("inf"), None
    for epoch in range(EPOCHS):
        net.train()
        shuffled = rng.permutation(train)
        for start in range(0, len(shuffled), BATCH):
            idx = shuffled[start : start + BATCH]
            loss = nn.functional.mse_loss(net(*collate(graphs, idx)), target[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        scheduler.step()
        val_rmse = float(np.sqrt(np.mean((predict(net, graphs, val) * y_std + y_mean - y[val]) ** 2)))
        if val_rmse < best:
            best, best_state = val_rmse, {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  val RMSE {val_rmse:.3f}")
    net.load_state_dict(best_state)

    pred = predict(net, graphs, test) * y_std + y_mean
    rmse = float(np.sqrt(np.mean((pred - y[test]) ** 2)))
    r2 = float(1 - np.sum((pred - y[test]) ** 2) / np.sum((y[test] - y[test].mean()) ** 2))
    print(f"test RMSE {rmse:.3f}  R2 {r2:.3f}  (n={len(test)}); baseline RMSE {y[test].std():.3f}")

    sd = net.state_dict()
    weights = {"embed_w": sd["embed.weight"].T, "embed_b": sd["embed.bias"]}
    for k in range(N_LAYERS):
        weights[f"self_w{k}"] = sd[f"self_lin.{k}.weight"].T
        weights[f"nbr_w{k}"] = sd[f"nbr_lin.{k}.weight"].T
        weights[f"bias{k}"] = sd[f"bias.{k}"]
    weights |= {
        "out_w0": sd["out0.weight"].T, "out_b0": sd["out0.bias"],
        "out_w1": sd["out1.weight"].T[:, 0], "out_b1": sd["out1.bias"][0],
    }
    arrays = {k: v.numpy().astype(np.float32) for k, v in weights.items()}
    arrays["y_mean"], arrays["y_std"] = np.float32(y_mean), np.float32(y_std)
    np.savez(ROOT / "model" / "gnn_lipophilicity.npz", **arrays)

    check = np.array([forward(arrays, *graphs[i])[0] for i in test[:50]])
    assert np.allclose(check, pred[:50], atol=1e-3), "NumPy forward pass disagrees with PyTorch"
    metrics = {"dataset": "MoleculeNet Lipophilicity (experimental logD 7.4)", "n_molecules": len(y),
               "n_test": int(len(test)), "test_rmse": round(rmse, 3), "test_r2": round(r2, 3),
               "baseline_rmse": round(float(y[test].std()), 3), "epochs": EPOCHS, "seed": SEED}
    (ROOT / "model" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
