"""
Grid search over ARIMA(p, 1, q) orders to pick the best for univariate forecasting.

Strategy:
  - For each (p, q) in {0,1,2,3} × {0,1,2,3}, d=1, fit ARIMA(p, 1, q) on each
    of the 80 train engines' full series.
  - Collect AIC and BIC per engine, average across engines.
  - Lower AIC/BIC = better trade-off between fit and complexity.

Why d=1: sensor_11 has clear trend (degradation) → first difference is
stationary. d=0 would leave a non-stationary input; d=2 over-differences.

Outputs:
  grid_search_results.csv     full table (16 rows)
  plots/grid_search_aic.png   heatmap of AIC across the (p, q) grid
"""

import warnings
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent.parent.parent / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

P_RANGE = [0, 1, 2, 3]
D       = 1
Q_RANGE = [0, 1, 2, 3]

train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df = train_df[train_df["mask"] == 1].copy()
engines = sorted(train_df["engine_id"].unique())
print(f"Loaded {len(train_df)} real rows  |  {len(engines)} train engines")

results = []
total = len(P_RANGE) * len(Q_RANGE)
print(f"\nGrid: {total} (p,q) combos x {len(engines)} engines = {total*len(engines)} fits\n")

t_start = time.time()
for p in P_RANGE:
    for q in Q_RANGE:
        order = (p, D, q)
        aics, bics, n_ok = [], [], 0
        for eid in engines:
            series = train_df[train_df["engine_id"] == eid]["sensor_11"].values
            try:
                m = ARIMA(series, order=order).fit()
                aics.append(float(m.aic))
                bics.append(float(m.bic))
                n_ok += 1
            except Exception:
                pass
        if n_ok > 0:
            results.append({
                "p": p, "d": D, "q": q,
                "n_engines_fit": n_ok,
                "aic_mean": float(np.mean(aics)),
                "bic_mean": float(np.mean(bics)),
            })
            print(f"  ARIMA({p},{D},{q})  fit on {n_ok:>2d}/{len(engines)} engines  "
                  f"|  AIC={np.mean(aics):>+9.2f}  BIC={np.mean(bics):>+9.2f}")
        else:
            print(f"  ARIMA({p},{D},{q})  FAILED on all engines")
elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.1f}s")

df = pd.DataFrame(results)
df.to_csv(ROOT / "grid_search_results.csv", index=False)

print(f"\n=== Top 5 by AIC (lower = better) ===")
print(df.sort_values("aic_mean").head(5).to_string(index=False))

print(f"\n=== Top 5 by BIC (lower = better, penalizes complexity more) ===")
print(df.sort_values("bic_mean").head(5).to_string(index=False))

# Heatmap of AIC
grid = np.full((len(P_RANGE), len(Q_RANGE)), np.nan)
for r in results:
    grid[P_RANGE.index(r["p"]), Q_RANGE.index(r["q"])] = r["aic_mean"]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(Q_RANGE)));  ax.set_xticklabels([f"q={q}" for q in Q_RANGE])
ax.set_yticks(range(len(P_RANGE)));  ax.set_yticklabels([f"p={p}" for p in P_RANGE])
ax.set_xlabel("MA order (q)");  ax.set_ylabel("AR order (p)")
ax.set_title(f"AIC mean across {len(engines)} train engines  —  ARIMA(p, {D}, q)")
mid = np.nanmean(grid)
for i in range(len(P_RANGE)):
    for j in range(len(Q_RANGE)):
        v = grid[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10,
                    color="white" if v < mid else "black")
fig.colorbar(im, ax=ax, label="AIC (lower = better)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grid_search_aic.png", dpi=130)
plt.close()
print(f"\nHeatmap saved to {PLOTS_DIR / 'grid_search_aic.png'}")
