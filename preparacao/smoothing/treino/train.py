"""Trains one MLP per alpha (0.1, 0.3, 0.5). Saves best_model_a01.pt etc."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import UnivariateEngineDataset, WINDOW
from model   import UnivariateMLP

ROOT      = Path(__file__).parent
TX_DIR    = ROOT.parent / "transformed"
RAW_TEST  = ROOT.parent.parent.parent / "data" / "test.csv"   # raw test (no smoothing)
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

EPOCHS = 400;  BATCH_SIZE = 128;  LR = 1e-3;  SEED = 42   # 400 epochs * 128 batches ≈ 51k iters
ALPHAS = [0.1, 0.3, 0.5]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SMOOTHING train]  device={device}")

results = {}

for a in ALPHAS:
    tag = f"a{int(a * 10):02d}"
    torch.manual_seed(SEED)
    print(f"\n=== alpha={a}  (tag={tag}) ===")
    train_ds = UnivariateEngineDataset(TX_DIR / f"train_smoothed_{tag}.csv")
    test_ds  = UnivariateEngineDataset(RAW_TEST)   # raw test (no smoothing)
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
            torch.save(model.state_dict(), ROOT / f"best_model_{tag}.pt")
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train={tr:.5f} | val={va:.5f}")

    print(f"Best val (alpha={a}): {best_val:.5f}  RMSE_scaled={np.sqrt(best_val):.4f}")
    results[a] = {"best": best_val, "train": train_losses, "val": val_losses}

# Combined loss plot
fig, axes = plt.subplots(1, len(ALPHAS), figsize=(6 * len(ALPHAS), 4), sharey=True)
for ax, a in zip(axes, ALPHAS):
    ax.plot(range(1, EPOCHS + 1), results[a]["train"], label="train", color="steelblue")
    ax.plot(range(1, EPOCHS + 1), results[a]["val"],   label="val",   color="tomato")
    ax.set_title(f"alpha={a}  best={results[a]['best']:.4f}")
    ax.set_xlabel("epoch");  ax.legend();  ax.grid(True, alpha=0.3)
axes[0].set_ylabel("MSE")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_loss_curves.png", dpi=130);  plt.close()

print(f"\nSummary:")
for a, r in results.items():
    print(f"  alpha={a}: best val MSE={r['best']:.5f}  RMSE_scaled={np.sqrt(r['best']):.4f}")
