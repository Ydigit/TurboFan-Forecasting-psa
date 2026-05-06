"""
Grid search over SARIMAX(p, 1, q) orders for the multivariate ARIMA model.

Same logic as univariate grid_search.py but with the top-8 exogenous features
(picked by data/feature_selection.py).

For each (p, q) in {0,1,2,3} × {0,1,2,3}, d=1:
  fit SARIMAX(p, 1, q) on each train engine's full series + 8 exog
  collect AIC and BIC
  average across engines

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
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent.parent.parent / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

P_RANGE = [0, 1, 2, 3]
D       = 1
Q_RANGE = [0, 1, 2, 3]

EXOG_COLS = (DATA_DIR / "selected_features.txt").read_text().strip().splitlines()
print(f"Exog features ({len(EXOG_COLS)}): {EXOG_COLS}")

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
            grp = train_df[train_df["engine_id"] == eid]
            series = grp["sensor_11"].values.astype(np.float64)
            exog   = grp[EXOG_COLS].values.astype(np.float64)
            try:
                m = SARIMAX(
                    endog=series, exog=exog, order=order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, method="lbfgs", maxiter=50)
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
            print(f"  SARIMAX({p},{D},{q})  fit on {n_ok:>2d}/{len(engines)} engines  "
                  f"|  AIC={np.mean(aics):>+9.2f}  BIC={np.mean(bics):>+9.2f}")
        else:
            print(f"  SARIMAX({p},{D},{q})  FAILED on all engines")
elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")

df = pd.DataFrame(results)
df.to_csv(ROOT / "grid_search_results.csv", index=False)

print(f"\n=== Top 5 by AIC (lower = better) ===")
print(df.sort_values("aic_mean").head(5).to_string(index=False))

print(f"\n=== Top 5 by BIC (lower = better) ===")
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
ax.set_title(f"AIC mean across {len(engines)} train engines  —  SARIMAX(p, {D}, q) + {len(EXOG_COLS)} exog")
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
