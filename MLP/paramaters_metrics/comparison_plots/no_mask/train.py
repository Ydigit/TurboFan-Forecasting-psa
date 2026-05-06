import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from MLP.univariate.comparison_plots.no_mask.dataset import UnivariateEngineDataset, WINDOW
from MLP.univariate.comparison_plots.no_mask.model   import UnivariateMLP

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent.parent.parent / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

EPOCHS     = 50
BATCH_SIZE = 128
LR         = 1e-3
SEED       = 42

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NO_MASK]  Device: {device}  |  Window: {WINDOW}")

train_ds = UnivariateEngineDataset(DATA_DIR / "train.csv")
test_ds  = UnivariateEngineDataset(DATA_DIR / "test.csv")
print(f"Train samples: {len(train_ds)}  |  Test samples: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

model     = UnivariateMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
best_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    tr = 0.0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        tr += loss.item() * len(y)
    tr /= len(train_ds)

    model.eval()
    va = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            va += criterion(model(X), y).item() * len(y)
    va /= len(test_ds)

    train_losses.append(tr); val_losses.append(va)
    if va < best_val:
        best_val = va
        torch.save(model.state_dict(), ROOT / "best_model.pt")

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | train={tr:.5f} | val={va:.5f}")

print(f"\n[NO_MASK] Best val loss: {best_val:.5f}  (RMSE scaled = {np.sqrt(best_val):.4f})")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, EPOCHS + 1), train_losses, label="train", color="steelblue")
ax.plot(range(1, EPOCHS + 1), val_losses,   label="val",   color="tomato")
ax.set_xlabel("epoch");  ax.set_ylabel("MSE loss (scaled)")
ax.set_title(f"Loss curve — NO mask  (best val={best_val:.4f})")
ax.grid(True, alpha=0.3);  ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_loss_curve.png", dpi=130);  plt.close()

model.load_state_dict(torch.load(ROOT / "best_model.pt"));  model.eval()
test_df = pd.read_csv(DATA_DIR / "test.csv")
series  = test_df["sensor_11"].values.astype(np.float32)
mask    = test_df["mask"].values.astype(np.float32)
engines = test_df["engine_id"].values
unique_engines = sorted(pd.unique(engines))[:3]

fig, axes = plt.subplots(len(unique_engines), 1, figsize=(12, 3 * len(unique_engines)))
if len(unique_engines) == 1: axes = [axes]
for ax, eid in zip(axes, unique_engines):
    idx = np.where((engines == eid) & (mask == 1))[0]
    preds, actuals = [], []
    for t in idx[:-1]:
        start = t - WINDOW + 1
        if start < 0:
            pad = -start
            win_vals = np.concatenate([np.zeros(pad, dtype=np.float32), series[0:t + 1]])
        else:
            win_vals = series[start:t + 1]
        x_t = torch.tensor(win_vals).unsqueeze(0).to(device)
        with torch.no_grad():
            preds.append(model(x_t).item())
        actuals.append(series[t + 1])
    ax.plot(actuals, label="actual", color="steelblue", linewidth=1.5)
    ax.plot(preds,   label="predicted", color="tomato", linewidth=1.2, alpha=0.9)
    ax.set_title(f"NO mask — Engine {eid}")
    ax.set_xlabel("cycle");  ax.set_ylabel("sensor_11 (scaled)")
    ax.legend();  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_predicted_vs_actual.png", dpi=130);  plt.close()
print(f"Plots saved in {PLOTS_DIR}")
