"""
Grid search for the multivariate LSTM, primarily varying the window size W.

Features used: sensor_11 (endogenous) + the top-8 exogenous features picked
by data/feature_selection.py = 9 features total.

For each W in W_RANGE (and optionally each hidden_size), a fresh LSTM is
trained for >= MIN_ITERS iterations (per CLAUDE.md spec). The best validation
state is saved as best_model_W{W}_h{hidden}.pt and the result is recorded.

Outputs:
  grid_search_results.csv      RMSE_psia, R2, n_test, train_time per combo
  plots/grid_search_rmse.png   bar chart of RMSE per (W, hidden)
  best_model_W{W}_h{hidden}.pt one checkpoint per combo
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import MultivariateEngineDataset
from model   import MultivariateLSTM

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent.parent.parent / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Grid ─────────────────────────────────────────────────────────────────────
W_RANGE      = [5, 10, 15, 20]
HIDDEN_RANGE = [16]                # add more values if you want to also vary hidden

MIN_ITERS = 50_000
BATCH     = 128
LR        = 1e-3
SEED      = 42

EXOG_FEATURES = (DATA_DIR / "selected_features.txt").read_text().strip().splitlines()
FEATURES      = ["sensor_11"] + EXOG_FEATURES   # 1 endo + 8 exog = 9
F             = len(FEATURES)

scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale    = float(scaler_y.data_range_[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Features ({F}): {FEATURES}")
print(f"Grid: W={W_RANGE} x hidden={HIDDEN_RANGE} = {len(W_RANGE)*len(HIDDEN_RANGE)} combos")


def train_one(W, hidden):
    torch.manual_seed(SEED)

    train_ds = MultivariateEngineDataset(DATA_DIR / "train.csv", window=W, features=FEATURES)
    test_ds  = MultivariateEngineDataset(DATA_DIR / "test.csv",  window=W, features=FEATURES)

    batches_per_epoch = max(1, (len(train_ds) + BATCH - 1) // BATCH)
    epochs = max(1, (MIN_ITERS + batches_per_epoch - 1) // batches_per_epoch)
    iters  = epochs * batches_per_epoch

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    model = MultivariateLSTM(input_size=F, hidden=hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
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

        train_losses.append(tr); val_losses.append(va)
        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), ROOT / f"best_model_W{W}_h{hidden}.pt")

    # Final RMSE/R² on test with best weights
    model.load_state_dict(torch.load(ROOT / f"best_model_W{W}_h{hidden}.pt"))
    model.eval()
    with torch.no_grad():
        preds, actuals = [], []
        for X, m, y in test_loader:
            X, m, y = X.to(device), m.to(device), y.to(device)
            preds.append(model(X, m).cpu().numpy())
            actuals.append(y.cpu().numpy())
    preds   = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    rmse_scaled = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    ss_res = float(np.sum((actuals - preds) ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot

    return {
        "W": W, "hidden": hidden, "epochs": epochs, "iters": iters,
        "rmse_scaled": rmse_scaled, "rmse_psia": rmse_scaled * scale, "r2": r2,
        "n_test": int(len(test_ds)),
        "train_losses": train_losses, "val_losses": val_losses,
    }


# ── Run ──────────────────────────────────────────────────────────────────────
results = []
t_start = time.time()
for W in W_RANGE:
    for hidden in HIDDEN_RANGE:
        t0 = time.time()
        print(f"\n--- W={W}, hidden={hidden} ---")
        res = train_one(W, hidden)
        elapsed = time.time() - t0
        res["elapsed_sec"] = elapsed
        results.append(res)
        print(f"  ep={res['epochs']:>4d}  iter={res['iters']:>6d}  "
              f"RMSE_psia={res['rmse_psia']:.4f}  R2={res['r2']:.4f}  ({elapsed:.1f}s)")

print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

# ── Save results ─────────────────────────────────────────────────────────────
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("train_losses", "val_losses")}
                   for r in results])
df = df.sort_values("rmse_psia").reset_index(drop=True)
df.to_csv(ROOT / "grid_search_results.csv", index=False)

print("\n=== Sorted by RMSE_psia (lower = better) ===")
print(df[["W", "hidden", "iters", "rmse_psia", "r2", "elapsed_sec"]].to_string(index=False))

# ── Bar chart of RMSE per combo ──────────────────────────────────────────────
labels = [f"W={r['W']}, h={r['hidden']}" for _, r in df.iterrows()]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(labels[::-1], df["rmse_psia"].values[::-1], color="steelblue", alpha=0.85)
for i, v in enumerate(df["rmse_psia"].values[::-1]):
    ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)
ax.set_xlabel("RMSE (psia)")
ax.set_title(f"LSTM multivariate — grid search (lower = better)\n"
             f"features = sensor_11 + top {len(EXOG_FEATURES)} exog ({F} total)")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grid_search_rmse.png", dpi=130)
plt.close()
print(f"\nPlot saved to {PLOTS_DIR / 'grid_search_rmse.png'}")

# ── Loss curves overlay ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for r in results:
    label = f"W={r['W']}, h={r['hidden']}"
    ax.plot(r["val_losses"], label=label, alpha=0.85)
ax.set_xlabel("epoch");  ax.set_ylabel("MSE val loss (scaled)")
ax.set_title("LSTM multivariate — validation loss by combo")
ax.grid(True, alpha=0.3);  ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grid_search_loss_curves.png", dpi=130)
plt.close()
print(f"Plot saved to {PLOTS_DIR / 'grid_search_loss_curves.png'}")
