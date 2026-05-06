"""
SARIMAX multivariate grid search by ROLLING-ORIGIN RMSE / R² — full (p, d, q) grid.

For each SARIMAX(p, d, q) with p, d, q in {0, 1, 2}, exog = top-8 features:
  - Rolling-origin 1-step-ahead forecast on test set
  - Compute RMSE_psia, MAE_psia, R²

27 combos total. Includes d=0 (no differencing — exposes non-stationarity)
and d=2 (over-differencing).

Incremental save after each combo so progress is never lost if interrupted.

Outputs:
  grid_search_rmse_v2_results.csv         full table
  plots/grid_v2_rmse_heatmaps.png         3 panels (one per d): RMSE_psia
  plots/grid_v2_r2_heatmaps.png           3 panels (one per d): R²
  plots/grid_v2_top10_bars.png            top 10 by RMSE / R²
  predictions_winner_v2.csv               predictions of the best combo
  plots/01_predicted_vs_actual_winner.png actual vs predicted on first 3 engines
  plots/02_residuals_winner.png           residuals histogram for winner
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
DATA_DIR  = ROOT.parent.parent.parent / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CSV_OUT = ROOT / "grid_search_rmse_v2_results.csv"

WINDOW = 30           # min history before first forecast (multivariate needs more)
P_RANGE = [0, 1, 2]
D_RANGE = [0, 1, 2]
Q_RANGE = [0, 1, 2]

EXOG_COLS = (DATA_DIR / "selected_features.txt").read_text().strip().splitlines()
print(f"Exog ({len(EXOG_COLS)}): {EXOG_COLS}")

scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale    = float(scaler_y.data_range_[0])

test_df = pd.read_csv(DATA_DIR / "test.csv")
test_df = test_df[test_df["mask"] == 1].copy()
print(f"Test rows: {len(test_df)} ({test_df['engine_id'].nunique()} engines)")
print(f"Grid: {len(P_RANGE) * len(D_RANGE) * len(Q_RANGE)} (p,d,q) combos\n")


def rolling_origin_one_order(order):
    t0 = time.time()
    preds, actuals, eng_out, t_out = [], [], [], []
    n_failed = 0
    for eid, grp in test_df.groupby("engine_id", sort=True):
        grp = grp.reset_index(drop=True)
        series = grp["sensor_11"].values.astype(np.float64)
        exog_full = grp[EXOG_COLS].values.astype(np.float64)
        for t in range(WINDOW, len(series) - 1):
            history       = series[: t + 1]
            history_exog  = exog_full[: t + 1]
            future_exog   = exog_full[t + 1: t + 2]
            try:
                m = SARIMAX(
                    endog=history, exog=history_exog, order=order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, method="lbfgs", maxiter=50)
                pred = float(m.forecast(steps=1, exog=future_exog)[0])
                if not np.isfinite(pred):
                    raise ValueError("non-finite forecast")
            except Exception:
                n_failed += 1
                pred = float(history[-1])
            preds.append(pred);  actuals.append(float(series[t + 1]))
            eng_out.append(int(eid));  t_out.append(int(t))
    elapsed = time.time() - t0
    preds = np.asarray(preds);  actuals = np.asarray(actuals)
    err  = preds - actuals
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    ss_r = float(np.sum(err ** 2));  ss_t = float(np.sum((actuals - actuals.mean()) ** 2))
    r2   = 1.0 - ss_r / ss_t
    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "n_used": len(preds), "n_failed": n_failed, "elapsed": elapsed,
        "preds": preds, "actuals": actuals, "engines": eng_out, "ts": t_out,
    }


# ── Resume-aware grid loop ──────────────────────────────────────────────────
existing = pd.read_csv(CSV_OUT) if CSV_OUT.exists() else pd.DataFrame()
done_keys = set()
if len(existing):
    done_keys = set(zip(existing["p"], existing["d"], existing["q"]))
    print(f"Resuming: {len(done_keys)} combos already in {CSV_OUT.name}")

results = existing.to_dict(orient="records") if len(existing) else []
total = len(P_RANGE) * len(D_RANGE) * len(Q_RANGE)
done = len(done_keys)
t_start = time.time()

# We'll keep predictions of the best-so-far combo in memory to write at the end
best_so_far = None
best_rmse_so_far = float("inf")

for p in P_RANGE:
    for d in D_RANGE:
        for q in Q_RANGE:
            if (p, d, q) in done_keys:
                continue
            done += 1
            order = (p, d, q)
            print(f"  [{done:>2d}/{total}] SARIMAX({p},{d},{q})... ", end="", flush=True)
            r = rolling_origin_one_order(order)
            row = {
                "p": p, "d": d, "q": q,
                "n_params": p + q,
                "rmse_scaled": r["rmse"],
                "rmse_psia":   r["rmse"] * scale,
                "mae_psia":    r["mae"] * scale,
                "r2":          r["r2"],
                "n":           r["n_used"],
                "n_failed":    r["n_failed"],
                "elapsed_sec": r["elapsed"],
            }
            results.append(row)
            pd.DataFrame(results).to_csv(CSV_OUT, index=False)
            if r["rmse"] < best_rmse_so_far:
                best_rmse_so_far = r["rmse"]
                best_so_far = {**r, "order": order}
            print(f"RMSE_psia={r['rmse']*scale:.4f}  R2={r['r2']:.4f}  ({r['elapsed']/60:.1f} min)  failures={r['n_failed']}")

elapsed_total = time.time() - t_start
print(f"\nTotal time (this run): {elapsed_total/60:.1f} min")

df = pd.DataFrame(results)
df.to_csv(CSV_OUT, index=False)

print("\n=== Top 5 by RMSE_psia ===")
print(df.sort_values("rmse_psia").head(5)[["p", "d", "q", "rmse_psia", "r2"]].to_string(index=False))

print("\n=== Top 5 by R² ===")
print(df.sort_values("r2", ascending=False).head(5)[["p", "d", "q", "rmse_psia", "r2"]].to_string(index=False))


# ── Heatmaps ────────────────────────────────────────────────────────────────
def heatmaps(metric_col, title, lower_is_better, fname):
    fig, axes = plt.subplots(1, len(D_RANGE), figsize=(5.5 * len(D_RANGE), 5), sharey=True)
    if len(D_RANGE) == 1: axes = [axes]
    all_vals = np.array([r[metric_col] for r in results])
    for ax, d in zip(axes, D_RANGE):
        grid = np.full((len(P_RANGE), len(Q_RANGE)), np.nan)
        for r in results:
            if r["d"] != d: continue
            grid[P_RANGE.index(r["p"]), Q_RANGE.index(r["q"])] = r[metric_col]
        cmap = "RdYlGn_r" if lower_is_better else "RdYlGn"
        im = ax.imshow(grid, cmap=cmap, aspect="auto",
                       vmin=np.nanmin(all_vals), vmax=np.nanmax(all_vals))
        ax.set_xticks(range(len(Q_RANGE)));  ax.set_xticklabels([f"q={q}" for q in Q_RANGE])
        ax.set_yticks(range(len(P_RANGE)));  ax.set_yticklabels([f"p={p}" for p in P_RANGE])
        ax.set_xlabel("MA (q)"); ax.set_ylabel("AR (p)")
        ax.set_title(f"d={d}")
        for i in range(len(P_RANGE)):
            for j in range(len(Q_RANGE)):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label=metric_col)
    plt.savefig(PLOTS_DIR / fname, dpi=130);  plt.close()

heatmaps("rmse_psia",
         "SARIMAX grid — RMSE_psia (rolling-origin on test, lower = better)",
         lower_is_better=True,
         fname="grid_v2_rmse_heatmaps.png")
heatmaps("r2",
         "SARIMAX grid — R² (rolling-origin on test, higher = better)",
         lower_is_better=False,
         fname="grid_v2_r2_heatmaps.png")


# ── Top 10 bars ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top_rmse = df.sort_values("rmse_psia").head(10)
labels_r = [f"({r['p']},{r['d']},{r['q']})" for _, r in top_rmse.iterrows()]
axes[0].barh(labels_r[::-1], top_rmse["rmse_psia"].values[::-1], color="steelblue", alpha=0.85)
for i, v in enumerate(top_rmse["rmse_psia"].values[::-1]):
    axes[0].text(v, i, f" {v:.4f}", va="center", fontsize=9)
axes[0].set_xlabel("RMSE (psia)");  axes[0].set_title("Top 10 by RMSE")
axes[0].grid(True, alpha=0.3, axis="x")

top_r2 = df.sort_values("r2", ascending=False).head(10)
labels_q = [f"({r['p']},{r['d']},{r['q']})" for _, r in top_r2.iterrows()]
axes[1].barh(labels_q[::-1], top_r2["r2"].values[::-1], color="tomato", alpha=0.85)
for i, v in enumerate(top_r2["r2"].values[::-1]):
    axes[1].text(v, i, f" {v:.4f}", va="center", fontsize=9)
axes[1].set_xlabel("R²");  axes[1].set_title("Top 10 by R²")
axes[1].grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "grid_v2_top10_bars.png", dpi=130);  plt.close()


# ── Plot for the winner ──────────────────────────────────────────────────────
if best_so_far is not None:
    pred_df = pd.DataFrame({
        "engine_id": best_so_far["engines"],
        "t":         best_so_far["ts"],
        "actual":    best_so_far["actuals"],
        "predicted": best_so_far["preds"],
    })
    pred_df.to_csv(ROOT / "predictions_winner_v2.csv", index=False)

    order = best_so_far["order"]
    rmse_psia = best_so_far["rmse"] * scale
    r2 = best_so_far["r2"]

    unique_engines = sorted(test_df["engine_id"].unique())[:3]
    fig, axes = plt.subplots(len(unique_engines), 1, figsize=(13, 3 * len(unique_engines)))
    if len(unique_engines) == 1: axes = [axes]
    for ax, eid in zip(axes, unique_engines):
        sub = pred_df[pred_df["engine_id"] == eid].sort_values("t").reset_index(drop=True)
        ax.plot(sub["actual"].values,    label="actual",                color="steelblue", linewidth=1.5)
        ax.plot(sub["predicted"].values, label=f"SARIMAX{order}+8exog", color="tomato",    linewidth=1.2, alpha=0.9)
        ax.set_title(f"Engine {eid} — winner SARIMAX{order} 1-step-ahead (multivariate)")
        ax.set_xlabel("ciclo (test)");  ax.set_ylabel("sensor_11 (escalado)")
        ax.legend();  ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_predicted_vs_actual_winner.png", dpi=130);  plt.close()

    err_psia = (best_so_far["preds"] - best_so_far["actuals"]) * scale
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err_psia, bins=50, color="steelblue", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("residual (psia)");  ax.set_ylabel("count")
    ax.set_title(f"SARIMAX{order} winner residuals — RMSE={rmse_psia:.4f} psia, R²={r2:.4f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_residuals_winner.png", dpi=130);  plt.close()

    print(f"\nWinner predictions saved to {ROOT / 'predictions_winner_v2.csv'}")

print(f"\nResults: {CSV_OUT}")
print(f"Plots:   {PLOTS_DIR}")
