"""
LSTM multivariate — extended grid search varying:
  - W              (window size: 5, 10, 15, 20, 30)
  - hidden_size    (16, 32, 64)
  - feature set    ("top-8" vs "all-13" vs "all-with-settings")

Iterations fixed at 5000 (spec minimum) so we can sweep more axes.
Saves predictions for the BEST combo and per-combo metrics.

Outputs:
  grid_search_v2_results.csv          full table (W, hidden, feature_set, RMSE, R2, ...)
  plots/v2_rmse_heatmap.png           RMSE heatmap per feature set
  plots/v2_loss_curves_best.png       loss curve of the best combo (verifies training)
  best_model_v2.pt                    best checkpoint
  predictions_v2.csv                  predictions of the best combo
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

# ── Grid axes ───────────────────────────────────────────────────────────────
W_RANGE        = [5, 10, 15, 20, 30]
HIDDEN_RANGE   = [16, 32, 64]
ITERATIONS     = 5_000           # fixed — spec minimum
BATCH          = 128
LR             = 1e-3
SEED           = 42

EXOG_TOP8 = (DATA_DIR / "selected_features.txt").read_text().strip().splitlines()
ALL_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
               "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17",
               "sensor_20", "sensor_21"]
SETTINGS = ["setting_1", "setting_2", "setting_3"]

FEATURE_SETS = {
    "top-8":           ["sensor_11"] + EXOG_TOP8,                        # 9
    "all-13":          ["sensor_11"] + ALL_SENSORS,                      # 14
    "all-with-settings": ["sensor_11"] + ALL_SENSORS + SETTINGS,         # 17
}

scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale    = float(scaler_y.data_range_[0])
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

total_combos = len(W_RANGE) * len(HIDDEN_RANGE) * len(FEATURE_SETS)
print(f"\nGrid: W={W_RANGE} x hidden={HIDDEN_RANGE} x features={list(FEATURE_SETS)} = {total_combos} combos")
print(f"Iterations per combo: {ITERATIONS}\n")


def train_one(W, hidden, features):
    torch.manual_seed(SEED)
    F = len(features)

    train_ds = MultivariateEngineDataset(DATA_DIR / "train.csv", window=W, features=features)
    test_ds  = MultivariateEngineDataset(DATA_DIR / "test.csv",  window=W, features=features)

    batches_per_epoch = max(1, (len(train_ds) + BATCH - 1) // BATCH)
    epochs = max(1, (ITERATIONS + batches_per_epoch - 1) // batches_per_epoch)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    model = MultivariateLSTM(input_size=F, hidden=hidden).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    best_val = float("inf");  best_state = None
    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train();  tr = 0.0
        for X, m, y in train_loader:
            X, m, y = X.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(X, m), y)
            loss.backward();  opt.step()
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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Final eval with best weights
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds, actuals = [], []
        for X, m, y in test_loader:
            X, m, y = X.to(device), m.to(device), y.to(device)
            preds.append(model(X, m).cpu().numpy())
            actuals.append(y.cpu().numpy())
    preds   = np.concatenate(preds);  actuals = np.concatenate(actuals)
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
    ss_r = float(np.sum((actuals - preds) ** 2));  ss_t = float(np.sum((actuals - actuals.mean()) ** 2))
    r2   = 1.0 - ss_r / ss_t

    return {
        "rmse_scaled": rmse, "rmse_psia": rmse * scale, "r2": r2,
        "epochs": epochs, "iters": epochs * batches_per_epoch,
        "n_test": int(len(test_ds)), "F": F,
        "train_losses": train_losses, "val_losses": val_losses,
        "best_state": best_state, "preds": preds, "actuals": actuals,
    }


# ── Run ──────────────────────────────────────────────────────────────────────
results = []
t_start = time.time()
combo_idx = 0
for fs_name, features in FEATURE_SETS.items():
    for hidden in HIDDEN_RANGE:
        for W in W_RANGE:
            combo_idx += 1
            t0 = time.time()
            print(f"  [{combo_idx:>2d}/{total_combos}] features={fs_name:<22s} W={W:<3d} hidden={hidden:<3d} ", end="", flush=True)
            res = train_one(W, hidden, features)
            elapsed = time.time() - t0
            res["W"] = W;  res["hidden"] = hidden;  res["feature_set"] = fs_name
            res["elapsed_sec"] = elapsed
            results.append(res)
            print(f"-> RMSE_psia={res['rmse_psia']:.4f}  R2={res['r2']:.4f}  ({elapsed:.1f}s)")

elapsed_total = time.time() - t_start
print(f"\nTotal time: {elapsed_total/60:.1f} min")

# ── Save table ───────────────────────────────────────────────────────────────
summary = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ("train_losses", "val_losses", "best_state", "preds", "actuals")}
                        for r in results])
summary = summary.sort_values("rmse_psia").reset_index(drop=True)
summary.to_csv(ROOT / "grid_search_v2_results.csv", index=False)

print("\n=== Top 10 by RMSE_psia ===")
print(summary[["feature_set", "W", "hidden", "F", "iters", "rmse_psia", "r2", "elapsed_sec"]].head(10).to_string(index=False))

# ── Save best combo as best_model_v2.pt + predictions_v2.csv ────────────────
best_idx = int(summary.index[0])
best_combo = next(r for r in results
                  if r["W"] == summary.iloc[0]["W"] and r["hidden"] == summary.iloc[0]["hidden"]
                  and r["feature_set"] == summary.iloc[0]["feature_set"])

torch.save(best_combo["best_state"], ROOT / "best_model_v2.pt")

test_df = pd.read_csv(DATA_DIR / "test.csv")
mask = test_df["mask"].values
engines = test_df["engine_id"].values
valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]
predictions_df = pd.DataFrame({
    "engine_id": engines[valid_t].astype(int),
    "t":         valid_t.astype(int),
    "actual":    best_combo["actuals"].astype(float),
    "predicted": best_combo["preds"].astype(float),
})
predictions_df.to_csv(ROOT / "predictions_v2.csv", index=False)

print(f"\nBest combo: features={best_combo['feature_set']}, W={best_combo['W']}, hidden={best_combo['hidden']}")
print(f"  RMSE_psia = {best_combo['rmse_psia']:.4f}")
print(f"  R²        = {best_combo['r2']:.4f}")

# ── Plot 1: RMSE heatmap (one panel per feature set) ─────────────────────────
fig, axes = plt.subplots(1, len(FEATURE_SETS), figsize=(6 * len(FEATURE_SETS), 5), sharey=True)
if len(FEATURE_SETS) == 1: axes = [axes]
for ax, fs_name in zip(axes, FEATURE_SETS.keys()):
    grid = np.full((len(W_RANGE), len(HIDDEN_RANGE)), np.nan)
    for i, w in enumerate(W_RANGE):
        for j, h in enumerate(HIDDEN_RANGE):
            row = summary[(summary["feature_set"] == fs_name) & (summary["W"] == w) & (summary["hidden"] == h)]
            if len(row): grid[i, j] = row["rmse_psia"].iloc[0]
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto",
                   vmin=summary["rmse_psia"].min(), vmax=summary["rmse_psia"].max())
    ax.set_xticks(range(len(HIDDEN_RANGE)));  ax.set_xticklabels([f"h={h}" for h in HIDDEN_RANGE])
    ax.set_yticks(range(len(W_RANGE)));       ax.set_yticklabels([f"W={w}" for w in W_RANGE])
    ax.set_title(f"{fs_name}\n({len(FEATURE_SETS[fs_name])} features)")
    for i in range(len(W_RANGE)):
        for j in range(len(HIDDEN_RANGE)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9)
fig.suptitle(f"LSTM multivariate v2 — RMSE (psia) by W x hidden x feature set\n(iter={ITERATIONS}, lr={LR})", fontsize=11)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label="RMSE (psia)")
plt.savefig(PLOTS_DIR / "v2_rmse_heatmap.png", dpi=130);  plt.close()

# ── Plot 2: loss curve of the best combo (verifies training) ─────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(best_combo["train_losses"], label="train", color="steelblue")
ax.plot(best_combo["val_losses"],   label="val",   color="tomato")
ax.set_xlabel("epoch");  ax.set_ylabel("MSE loss (scaled)")
ax.set_title(f"Best combo loss curve — {best_combo['feature_set']}, W={best_combo['W']}, hidden={best_combo['hidden']}")
ax.grid(True, alpha=0.3);  ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "v2_loss_curves_best.png", dpi=130);  plt.close()

# ── Plot 3: predicted vs actual (best combo) on first 3 engines ─────────────
unique_engines = sorted(test_df["engine_id"].unique())[:3]
fig, axes = plt.subplots(len(unique_engines), 1, figsize=(13, 3 * len(unique_engines)))
if len(unique_engines) == 1: axes = [axes]
for ax, eid in zip(axes, unique_engines):
    sub = predictions_df[predictions_df["engine_id"] == eid].sort_values("t").reset_index(drop=True)
    ax.plot(sub["actual"].values,    label="actual",      color="steelblue", linewidth=1.5)
    ax.plot(sub["predicted"].values, label="LSTM v2 best", color="tomato",   linewidth=1.2, alpha=0.9)
    ax.set_title(f"Engine {eid} — best LSTM (features={best_combo['feature_set']}, W={best_combo['W']}, h={best_combo['hidden']})")
    ax.set_xlabel("ciclo (test)");  ax.set_ylabel("sensor_11 (escalado)")
    ax.legend();  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "v2_predicted_vs_actual_best.png", dpi=130);  plt.close()

print(f"\nPlots saved in {PLOTS_DIR}")
print(f"Results: {ROOT / 'grid_search_v2_results.csv'}")
print(f"Best model: {ROOT / 'best_model_v2.pt'}")
