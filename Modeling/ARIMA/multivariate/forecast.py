"""
SARIMAX(1,1,1) — 1-step-ahead forecast for sensor_11 with other sensors as exogenous.

Pipeline:
  endogenous: sensor_11 (Ps30 — the target)
  exogenous:  top 8 informative non-redundant features picked by data/feature_selection.py

Rolling-origin per test engine: for cycle t >= WINDOW, fit SARIMAX on
history[0..t] and forecast t+1 using exog observed at t+1.

WINDOW=30 so each fit has at least 30 history points before estimating
8 exog + 3 ARIMA coefficients (avoids over-determined fits).
"""

import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PROJECT   = ROOT.parent.parent.parent
DATA_DIR  = PROJECT / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW = 30
ORDER  = (0, 1, 1)   # (p, d, q) — chosen by BIC + AIC unanimously in grid search

# Read the top features picked by data/feature_selection.py
EXOG_COLS = (DATA_DIR / "selected_features.txt").read_text().strip().splitlines()

scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale    = float(scaler_y.data_range_[0])   # MinMax: max - min of target in psia

test_df = pd.read_csv(DATA_DIR / "test.csv")
test_df = test_df[test_df["mask"] == 1].copy()
print(f"Test rows: {len(test_df)} ({test_df['engine_id'].nunique()} engines)")
print(f"SARIMAX order: {ORDER}  |  exog: {len(EXOG_COLS)} sensors")

preds, actuals, engines_out, ts_out = [], [], [], []
n_failed = 0
total = 0

t_start = time.time()
for eid, grp in test_df.groupby("engine_id", sort=True):
    grp = grp.reset_index(drop=True)
    series = grp["sensor_11"].values.astype(np.float64)
    exog_full = grp[EXOG_COLS].values.astype(np.float64)

    for t in range(WINDOW, len(series) - 1):
        total += 1
        history      = series[: t + 1]
        history_exog = exog_full[: t + 1]
        future_exog  = exog_full[t + 1: t + 2]

        try:
            model = SARIMAX(
                endog=history,
                exog=history_exog,
                order=ORDER,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, method="lbfgs", maxiter=50)
            pred = float(model.forecast(steps=1, exog=future_exog)[0])
        except Exception:
            n_failed += 1
            pred = float(history[-1])  # naive fallback

        preds.append(pred)
        actuals.append(float(series[t + 1]))
        engines_out.append(int(eid))
        ts_out.append(int(t))

    # progress per engine
    elapsed_so_far = time.time() - t_start
    print(f"  engine {int(eid):>3d} done  |  total fits={total}  |  failures={n_failed}  |  elapsed={elapsed_so_far/60:.1f} min", flush=True)

elapsed = time.time() - t_start
print(f"\nFitted {len(preds)} forecasts in {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Failures (fell back to naive): {n_failed} / {total}")

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
    ax.plot(actuals[idx], label="actual",          color="steelblue", linewidth=1.5)
    ax.plot(preds[idx],   label="SARIMAX (multi)", color="tomato",    linewidth=1.2, alpha=0.9)
    ax.set_title(f"Engine {eid} — SARIMAX{ORDER} multivariate 1-step-ahead")
    ax.set_xlabel("ciclo (test)");  ax.set_ylabel("sensor_11 (escalado)")
    ax.legend();  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_predicted_vs_actual.png", dpi=130);  plt.close()

# ── Plot 2: residuals histogram ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(errors * scale, bins=50, color="steelblue", alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("residual (psia)");  ax.set_ylabel("count")
ax.set_title(f"SARIMAX multivariate residuals — RMSE={rmse_scaled*scale:.4f} psia, R²={r2:.4f}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_residuals.png", dpi=130);  plt.close()

print(f"\nPlots saved in {PLOTS_DIR}")
print(f"Predictions saved in {ROOT / 'predictions.csv'}")
