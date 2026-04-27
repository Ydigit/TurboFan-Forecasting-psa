import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import DiffEngineDataset, WINDOW
from model   import UnivariateMLP

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent / "transformed"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

EPOCHS = 50;  BATCH_SIZE = 128;  LR = 1e-3;  SEED = 42
ORDERS = [1, 2]

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DIFF train]  device={device}")


def train_one(order: int):
    train_ds = DiffEngineDataset(DATA_DIR / f"train_diff{order}.csv")
    test_ds  = DiffEngineDataset(DATA_DIR / f"test_diff{order}.csv")
    print(f"\n=== diff_{order}  ===  Train {len(train_ds)}  |  Test {len(test_ds)}")

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
            torch.save(model.state_dict(), ROOT / f"best_model_diff{order}.pt")
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={tr:.5f} | val={va:.5f}")

    print(f"  Best val loss diff_{order}: {best_val:.5f}  RMSE_scaled={np.sqrt(best_val):.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, EPOCHS + 1), train_losses, label="train", color="steelblue")
    ax.plot(range(1, EPOCHS + 1), val_losses,   label="val",   color="tomato")
    ax.set_title(f"Differentiation (order {order})  best val={best_val:.4f}")
    ax.set_xlabel("epoch");  ax.set_ylabel("MSE");  ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / f"01_loss_curve_diff{order}.png"
    plt.savefig(out, dpi=130);  plt.close()
    print(f"  Saved {out}")
    return best_val


for k in ORDERS:
    train_one(k)
