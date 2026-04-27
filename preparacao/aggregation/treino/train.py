"""Trains one MLP per granularity (3, 5). Saves best_model_gran{N}.pt."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import AggregatedEngineDataset, WINDOW
from model   import UnivariateMLP

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent / "transformed"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

EPOCHS = 50;  BATCH_SIZE = 128;  LR = 1e-3;  SEED = 42
GRANS = [3, 5]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[AGG train]  device={device}")

results = {}

for N in GRANS:
    torch.manual_seed(SEED)
    print(f"\n=== Granularidade N={N} ===")
    train_ds = AggregatedEngineDataset(DATA_DIR / f"train_gran{N}.csv")
    test_ds  = AggregatedEngineDataset(DATA_DIR / f"test_gran{N}.csv")
    print(f"Train {len(train_ds)}  |  Test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = UnivariateMLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train();  tr = 0.0
        for X, m, y in train_loader:
            X, m, y = X.to(device), m.to(device), y.to(device)
            opt.zero_grad();  loss = crit(model(X, m), y);  loss.backward();  opt.step()
            tr += loss.item() * len(y)
        tr /= len(train_ds)

        model.eval();  va = 0.0
        with torch.no_grad():
            for X, m, y in test_loader:
                X, m, y = X.to(device), m.to(device), y.to(device)
                va += crit(model(X, m), y).item() * len(y)
        va /= len(test_ds)

        train_losses.append(tr);  val_losses.append(va)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), ROOT / f"best_model_gran{N}.pt")
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train={tr:.5f} | val={va:.5f}")

    print(f"Best val (gran{N}): {best_val:.5f}  RMSE_scaled={np.sqrt(best_val):.4f}")
    results[N] = {"best": best_val, "train": train_losses, "val": val_losses}

# Combined loss plot
fig, axes = plt.subplots(1, len(GRANS), figsize=(7 * len(GRANS), 4), sharey=True)
for ax, N in zip(axes, GRANS):
    ax.plot(range(1, EPOCHS + 1), results[N]["train"], label="train", color="steelblue")
    ax.plot(range(1, EPOCHS + 1), results[N]["val"],   label="val",   color="tomato")
    ax.set_title(f"gran={N}  best={results[N]['best']:.4f}")
    ax.set_xlabel("epoch");  ax.legend();  ax.grid(True, alpha=0.3)
axes[0].set_ylabel("MSE")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_loss_curves.png", dpi=130);  plt.close()

print(f"\nSummary:")
for N, r in results.items():
    print(f"  gran={N}: best val MSE={r['best']:.5f}  RMSE_scaled={np.sqrt(r['best']):.4f}")
