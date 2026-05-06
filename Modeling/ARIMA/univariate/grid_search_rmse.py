"""
ARIMA univariate grid search by ROLLING-ORIGIN RMSE / R² on the test set.

Alternative to grid_search.py (AIC/BIC). For each ARIMA(p, 1, q) with p, q in {0..3}:
  - Run the same rolling-origin 1-step-ahead forecast as forecast.py on the test set
  - Compute RMSE_psia, R², MAE, etc.

This is the "out-of-sample empirical" approach — no AIC/BIC. The trade-off vs
grid_search.py: ~10x slower (60 000 vs 1 280 fits) and uses test data for selection
(arguably "peeking"), but uses the metric the user actually cares about.

Outputs:
  grid_search_rmse_results.csv     full table
  plots/grid_search_rmse_heatmap.png  heatmap of RMSE_psia per (p, q)
  plots/grid_search_r2_heatmap.png    heatmap of R²
"""

import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent.parent.parent / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW = 10
P_RANGE = [0, 1, 2, 3]
D       = 1
Q_RANGE = [0, 1, 2, 3]

scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale    = float(scaler_y.data_range_[0])

test_df = pd.read_csv(DATA_DIR / "test.csv")
test_df = test_df[test_df["mask"] == 1].copy()
print(f"Test rows: {len(test_df)} ({test_df['engine_id'].nunique()} engines)")
print(f"Grid: {len(P_RANGE) * len(Q_RANGE)} (p,q) combos with d={D}\n")


def rolling_origin_one_order(order):
    """Run rolling-origin 1-step-ahead forecast for a given ARIMA order on the test set.
    Returns (RMSE, MAE, R2, n_used, n_failed, elapsed_sec)."""
    t0 = time.time()
    preds, actuals = [], []
    n_failed = 0
    for eid, grp in test_df.groupby("engine_id", sort=True):
        series = grp["sensor_11"].values.astype(np.float64)
        for t in range(WINDOW, len(series) - 1):
            history = series[: t + 1]
            try:
                m = ARIMA(history, order=order).fit()
                pred = float(m.forecast(steps=1)[0])
            except Exception:
                n_failed += 1
                pred = float(history[-1])  # naive fallback
            preds.append(pred)
            actuals.append(float(series[t + 1]))
    elapsed = time.time() - t0
    preds = np.asarray(preds);  actuals = np.asarray(actuals)
    err  = preds - actuals
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    ss_r = float(np.sum(err ** 2));  ss_t = float(np.sum((actuals - actuals.mean()) ** 2))
    r2   = 1.0 - ss_r / ss_t
    return rmse, mae, r2, len(preds), n_failed, elapsed


# ── Run all combos ──────────────────────────────────────────────────────────
results = []
total = len(P_RANGE) * len(Q_RANGE)
done = 0
t_start = time.time()

for p in P_RANGE:
    for q in Q_RANGE:
        done += 1
        order = (p, D, q)
        print(f"  [{done:>2d}/{total}] ARIMA({p},{D},{q})... ", end="", flush=True)
        rmse, mae, r2, n_used, n_failed, elapsed = rolling_origin_one_order(order)
        results.append({
            "p": p, "d": D, "q": q,
            "n_params": p + q,
            "rmse_scaled": rmse,
            "rmse_psia":   rmse * scale,
            "mae_psia":    mae * scale,
            "r2":          r2,
            "n":           n_used,
            "n_failed":    n_failed,
            "elapsed_sec": elapsed,
        })
        print(f"RMSE_psia={rmse*scale:.4f}  R2={r2:.4f}  ({elapsed/60:.1f} min)  failures={n_failed}")

elapsed_total = time.time() - t_start
print(f"\nTotal time: {elapsed_total/60:.1f} min")

df = pd.DataFrame(results)
df_sorted_rmse = df.sort_values("rmse_psia").reset_index(drop=True)
df_sorted_r2   = df.sort_values("r2", ascending=False).reset_index(drop=True)
df.to_csv(ROOT / "grid_search_rmse_results.csv", index=False)

print("\n=== Top 5 by RMSE_psia (lower = better) ===")
print(df_sorted_rmse[["p", "d", "q", "n_params", "rmse_psia", "r2", "elapsed_sec"]].head(5).to_string(index=False))

print("\n=== Top 5 by R² (higher = better) ===")
print(df_sorted_r2[["p", "d", "q", "n_params", "rmse_psia", "r2", "elapsed_sec"]].head(5).to_string(index=False))

# ── Heatmaps ────────────────────────────────────────────────────────────────
def heatmap(metric_col, title, cmap, lower_is_better, fname):
    grid = np.full((len(P_RANGE), len(Q_RANGE)), np.nan)
    for r in results:
        grid[P_RANGE.index(r["p"]), Q_RANGE.index(r["q"])] = r[metric_col]
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap_eff = cmap if lower_is_better else cmap + "_r"
    # use reversed cmap to make "good" green
    if lower_is_better:
        im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
    else:
        im = ax.imshow(grid, cmap="RdYlGn",   aspect="auto")
    ax.set_xticks(range(len(Q_RANGE)));  ax.set_xticklabels([f"q={q}" for q in Q_RANGE])
    ax.set_yticks(range(len(P_RANGE)));  ax.set_yticklabels([f"p={p}" for p in P_RANGE])
    ax.set_xlabel("MA order (q)");  ax.set_ylabel("AR order (p)")
    ax.set_title(title)
    mid = np.nanmean(grid)
    for i in range(len(P_RANGE)):
        for j in range(len(Q_RANGE)):
            v = grid[i, j]
            if not np.isnan(v):
                color = "white" if (lower_is_better and v > mid) or (not lower_is_better and v < mid) else "black"
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, label=metric_col)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=130);  plt.close()

heatmap("rmse_psia", f"ARIMA(p,{D},q) — RMSE_psia (rolling-origin on test)\n(lower = better)",
        "RdYlGn", lower_is_better=True, fname="grid_search_rmse_heatmap.png")
heatmap("r2",        f"ARIMA(p,{D},q) — R² (rolling-origin on test)\n(higher = better)",
        "RdYlGn", lower_is_better=False, fname="grid_search_r2_heatmap.png")

print(f"\nResults: {ROOT / 'grid_search_rmse_results.csv'}")
print(f"Plots:   {PLOTS_DIR}")
