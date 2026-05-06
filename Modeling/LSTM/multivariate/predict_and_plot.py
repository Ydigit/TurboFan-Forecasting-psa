"""
Generate predictions and standard plots for the winning LSTM multivariate model
(W=10, hidden=16, 9 features).

Produces the same kind of plots used for the other models:
  plots/01_predicted_vs_actual.png   actual vs predicted overlay on first 3 engines
  plots/02_residuals.png             residuals histogram (psia)
  predictions.csv                    full prediction table
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt

from dataset import MultivariateEngineDataset
from model   import MultivariateLSTM

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent.parent.parent / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

W       = 10
HIDDEN  = 16

EXOG_FEATURES = (DATA_DIR / "selected_features.txt").read_text().strip().splitlines()
FEATURES      = ["sensor_11"] + EXOG_FEATURES
F             = len(FEATURES)

scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale    = float(scaler_y.data_range_[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  W={W}, hidden={HIDDEN}, features={F}")

# ── Load model + dataset ─────────────────────────────────────────────────────
test_ds = MultivariateEngineDataset(DATA_DIR / "test.csv", window=W, features=FEATURES)
X = test_ds.X.to(device); M = test_ds.mask.to(device); y = test_ds.y.cpu().numpy()

model = MultivariateLSTM(input_size=F, hidden=HIDDEN).to(device)
model.load_state_dict(torch.load(ROOT / f"best_model_W{W}_h{HIDDEN}.pt", map_location=device))
model.eval()

with torch.no_grad():
    pred = model(X, M).cpu().numpy()

# ── Reconstruct (engine_id, t) per row ───────────────────────────────────────
test_df = pd.read_csv(DATA_DIR / "test.csv")
mask = test_df["mask"].values
engines = test_df["engine_id"].values
valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]

predictions = pd.DataFrame({
    "engine_id": engines[valid_t].astype(int),
    "t":         valid_t.astype(int),
    "actual":    y.astype(float),
    "predicted": pred.astype(float),
})
predictions.to_csv(ROOT / "predictions.csv", index=False)

# ── Metrics ──────────────────────────────────────────────────────────────────
errors = pred - y
rmse_scaled = float(np.sqrt(np.mean(errors ** 2)))
mae_scaled  = float(np.mean(np.abs(errors)))
ss_res = float(np.sum(errors ** 2))
ss_tot = float(np.sum((y - y.mean()) ** 2))
r2     = 1.0 - ss_res / ss_tot

print(f"\n{'Metric':<15s} {'scaled':>10s} {'psia':>10s}")
print("-" * 38)
print(f"{'RMSE':<15s} {rmse_scaled:10.4f} {rmse_scaled*scale:10.4f}")
print(f"{'MAE':<15s} {mae_scaled:10.4f} {mae_scaled*scale:10.4f}")
print(f"{'R2':<15s} {r2:10.4f}")

# ── Plot 1: predicted vs actual on first 3 engines ───────────────────────────
unique_engines = sorted(test_df["engine_id"].unique())[:3]
fig, axes = plt.subplots(len(unique_engines), 1, figsize=(13, 3 * len(unique_engines)))
if len(unique_engines) == 1: axes = [axes]
for ax, eid in zip(axes, unique_engines):
    sub = predictions[predictions["engine_id"] == eid].sort_values("t").reset_index(drop=True)
    ax.plot(sub["actual"].values,    label="actual",      color="steelblue", linewidth=1.5)
    ax.plot(sub["predicted"].values, label=f"LSTM multi", color="tomato",    linewidth=1.2, alpha=0.9)
    ax.set_title(f"Engine {eid} — LSTM multivariate (W={W}, {F} features) 1-step-ahead")
    ax.set_xlabel("ciclo (test)");  ax.set_ylabel("sensor_11 (escalado)")
    ax.legend();  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_predicted_vs_actual.png", dpi=130);  plt.close()

# ── Plot 2: residuals histogram ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(errors * scale, bins=50, color="steelblue", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("residual (psia)");  ax.set_ylabel("count")
ax.set_title(f"LSTM multivariate residuals — RMSE={rmse_scaled*scale:.4f} psia, R²={r2:.4f}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_residuals.png", dpi=130);  plt.close()

print(f"\nPlots saved in {PLOTS_DIR}")
print(f"Predictions saved in {ROOT / 'predictions.csv'}")
