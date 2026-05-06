"""
Grid search over Exponential Smoothing configurations for univariate forecasting.

Configurations (5):
  1. SES                       trend=None,   damped=False
  2. Holt additive             trend="add",  damped=False
  3. Holt additive damped      trend="add",  damped=True
  4. Holt multiplicative       trend="mul",  damped=False     (requires positive values)
  5. Holt multiplicative damped trend="mul", damped=True      (requires positive values)

Seasonality skipped — sensor_11 has no obvious cyclic pattern per engine.

For each config, fit on each train engine's full series. Collect:
  - AIC, AICc, BIC (model selection — AICc is the corrected AIC for small samples)
  - in-sample RMSE (residual size on training data)
  - estimated alpha (level smoothing), beta (trend smoothing), phi (damping)

Outputs:
  grid_search_results.csv     full table
  plots/grid_search.png       comparison bar charts (AIC, BIC, in-sample RMSE)
"""

import warnings
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent.parent.parent / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CONFIGS = [
    {"name": "SES",                   "trend": None,  "damped": False},
    {"name": "Holt-add",              "trend": "add", "damped": False},
    {"name": "Holt-add-damped",       "trend": "add", "damped": True},
    {"name": "Holt-mul",              "trend": "mul", "damped": False},
    {"name": "Holt-mul-damped",       "trend": "mul", "damped": True},
]

train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df = train_df[train_df["mask"] == 1].copy()
engines = sorted(train_df["engine_id"].unique())
print(f"Loaded {len(train_df)} real rows  |  {len(engines)} train engines")

s_min = train_df["sensor_11"].min()
s_max = train_df["sensor_11"].max()
print(f"sensor_11 range: [{s_min:.3f}, {s_max:.3f}]")
if s_min <= 0:
    print(f"NOTE: scaled data has non-positive values — mul trend configs will be skipped on those engines\n")

results = []
print(f"Grid: {len(CONFIGS)} configs x {len(engines)} engines = {len(CONFIGS)*len(engines)} fits\n")

t_start = time.time()
for cfg in CONFIGS:
    name = cfg["name"]
    aics, aiccs, bics, rmses, alphas, betas, phis = [], [], [], [], [], [], []
    n_ok, n_skip = 0, 0

    for eid in engines:
        series = train_df[train_df["engine_id"] == eid]["sensor_11"].values
        if cfg["trend"] == "mul" and (series <= 0).any():
            n_skip += 1
            continue
        try:
            m = ExponentialSmoothing(
                series,
                trend=cfg["trend"],
                damped_trend=cfg["damped"],
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)

            aics .append(float(m.aic))
            aiccs.append(float(m.aicc))
            bics .append(float(m.bic))
            rmses.append(float(np.sqrt(np.mean((m.fittedvalues - series) ** 2))))

            p = m.params
            alphas.append(float(p.get("smoothing_level", np.nan)))
            betas .append(float(p.get("smoothing_trend",  np.nan)) if cfg["trend"]   else np.nan)
            phis  .append(float(p.get("damping_trend",    np.nan)) if cfg["damped"] else np.nan)
            n_ok += 1
        except Exception:
            pass

    if n_ok > 0:
        results.append({
            "config":              name,
            "trend":               cfg["trend"],
            "damped":              cfg["damped"],
            "n_engines_fit":       n_ok,
            "n_engines_skipped":   n_skip,
            "aic_mean":            float(np.mean(aics)),
            "aicc_mean":           float(np.mean(aiccs)),
            "bic_mean":            float(np.mean(bics)),
            "rmse_insample_mean":  float(np.mean(rmses)),
            "alpha_mean":          float(np.nanmean(alphas)),
            "beta_mean":           float(np.nanmean(betas)) if not np.all(np.isnan(betas)) else float("nan"),
            "phi_mean":            float(np.nanmean(phis))  if not np.all(np.isnan(phis))  else float("nan"),
        })
        ms = lambda v: f"{v:.3f}" if not np.isnan(v) else "  —  "
        beta_v = np.nanmean(betas) if betas else float("nan")
        phi_v  = np.nanmean(phis)  if phis  else float("nan")
        print(f"  {name:<22s}  n={n_ok:>2d} (skip={n_skip})  AIC={np.mean(aics):>+8.2f}  "
              f"AICc={np.mean(aiccs):>+8.2f}  BIC={np.mean(bics):>+8.2f}  "
              f"RMSE_in={np.mean(rmses):.4f}  alpha={ms(np.nanmean(alphas))}  "
              f"beta={ms(beta_v)}  phi={ms(phi_v)}")
    else:
        print(f"  {name:<22s}  FAILED on all engines (skipped: {n_skip})")

elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.1f}s")

df = pd.DataFrame(results)
df.to_csv(ROOT / "grid_search_results.csv", index=False)

cols = ["config", "aic_mean", "aicc_mean", "bic_mean", "rmse_insample_mean",
        "alpha_mean", "beta_mean", "phi_mean"]
print(f"\n=== Sorted by AIC (lower = better) ===")
print(df.sort_values("aic_mean")[cols].to_string(index=False))

print(f"\n=== Sorted by BIC (lower = better) ===")
print(df.sort_values("bic_mean")[cols].to_string(index=False))

# ── Comparison bar charts ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
specs = [
    ("aic_mean",            "AIC (lower = better)",            "steelblue"),
    ("aicc_mean",           "AICc (lower = better)",           "purple"),
    ("bic_mean",            "BIC (lower = better)",            "tomato"),
    ("rmse_insample_mean",  "In-sample RMSE (lower = better)", "seagreen"),
]
for ax, (col, title, color) in zip(axes, specs):
    s = df.sort_values(col)
    ax.barh(s["config"], s[col], color=color, alpha=0.85)
    for i, v in enumerate(s[col]):
        ax.text(v, i, f"  {v:.3f}", va="center", fontsize=9)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "grid_search.png", dpi=130)
plt.close()
print(f"\nPlot saved to {PLOTS_DIR / 'grid_search.png'}")
