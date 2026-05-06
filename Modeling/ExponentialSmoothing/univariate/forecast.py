"""
Holt's Exponential Smoothing (additive trend) — 1-step-ahead forecast for sensor_11.

Per the spec, exponential smoothing is univariate-only and is a per-series method.
Strategy: rolling-origin forecast on each test engine.

For each test engine, iterate cycles t = WINDOW .. T-2:
  history = sensor_11[0..t]
  fit ExponentialSmoothing(trend='add') on history
  forecast t+1
  compare with actual

Output: predictions.csv, plots, RMSE/R2 in psia.

Note: ExpSmoothing doesn't have "iterations" like neural networks. The fit uses
MLE (closed form for level/trend states; scipy optimizer for alpha/beta).
"""

import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PROJECT   = ROOT.parent.parent.parent
DATA_DIR  = PROJECT / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW = 10  # min history before first forecast (parity with MLP)

scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale    = float(scaler_y.data_range_[0])   # MinMax: max - min of target in psia

test_df = pd.read_csv(DATA_DIR / "test.csv")
test_df = test_df[test_df["mask"] == 1].copy()  # drop padding
print(f"Test rows: {len(test_df)} ({test_df['engine_id'].nunique()} engines)")
print(f"Forecast horizon: 1 cycle ahead (rolling origin)")

preds, actuals, engines_out, ts_out = [], [], [], []

t_start = time.time()
for eid, grp in test_df.groupby("engine_id", sort=True):
    series = grp["sensor_11"].values
    for t in range(WINDOW, len(series) - 1):
        history = series[: t + 1]
        try:
            model = ExponentialSmoothing(
                history, trend="add", initialization_method="estimated"
            ).fit(optimized=True)
            pred = float(model.forecast(1)[0])
        except Exception:
            pred = float(history[-1])  # naive fallback for unstable fits
        preds.append(pred)
        actuals.append(float(series[t + 1]))
        engines_out.append(int(eid))
        ts_out.append(int(t))
elapsed = time.time() - t_start
print(f"\nFitted {len(preds)} forecasts in {elapsed:.1f}s")

preds   = np.array(preds, dtype=np.float64)
actuals = np.array(actuals, dtype=np.float64)
errors  = preds - actuals

rmse_scaled = float(np.sqrt(np.mean(errors ** 2)))
mae_scaled  = float(np.mean(np.abs(errors)))
ss_res      = float(np.sum(errors ** 2))
ss_tot      = float(np.sum((actuals - actuals.mean()) ** 2))
r2          = 1.0 - ss_res / ss_tot

print(f"\n{'Metric':<15s} {'scaled':>10s} {'psia':>10s}")
print("-" * 38)
print(f"{'RMSE':<15s} {rmse_scaled:10.4f} {rmse_scaled*scale:10.4f}")
print(f"{'MAE':<15s} {mae_scaled:10.4f} {mae_scaled*scale:10.4f}")
print(f"{'R2':<15s} {r2:10.4f}")

pd.DataFrame({
    "engine_id": engines_out, "t": ts_out, "actual": actuals, "predicted": preds,
}).to_csv(ROOT / "predictions.csv", index=False)

# ── Plot 1: predicted vs actual on first 3 engines ───────────────────────────
unique_engines = sorted(set(engines_out))[:3]
fig, axes = plt.subplots(len(unique_engines), 1, figsize=(13, 3 * len(unique_engines)))
if len(unique_engines) == 1: axes = [axes]
for ax, eid in zip(axes, unique_engines):
    idx = np.array([i for i, e in enumerate(engines_out) if e == eid])
    ax.plot(actuals[idx], label="actual",        color="steelblue", linewidth=1.5)
    ax.plot(preds[idx],   label="ExpSmoothing",  color="tomato",    linewidth=1.2, alpha=0.9)
    ax.set_title(f"Engine {eid} — Holt ExpSmoothing 1-step-ahead")
    ax.set_xlabel("ciclo (test)");  ax.set_ylabel("sensor_11 (escalado)")
    ax.legend();  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_predicted_vs_actual.png", dpi=130);  plt.close()

# ── Plot 2: residuals histogram ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(errors * scale, bins=50, color="steelblue", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("residual (psia)");  ax.set_ylabel("count")
ax.set_title(f"ExpSmoothing residuals — RMSE={rmse_scaled*scale:.4f} psia, R²={r2:.4f}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_residuals.png", dpi=130);  plt.close()

print(f"\nPlots saved in {PLOTS_DIR}")
print(f"Predictions saved in {ROOT / 'predictions.csv'}")
