"""
Alpha (and beta) sweep for Holt's Exponential Smoothing.

Goal: show how α (level smoothing) and β (trend smoothing) affect 1-step-ahead
forecast performance — required by the spec ("analysis of how different
parameters affect models' performance").

Strategy: fixed-parameter Holt recursion (no per-step refit), super fast.
For each (α, β) pair:
  - Apply the closed-form Holt recursion to each test engine's series
  - Extract 1-step-ahead forecasts for cycles WINDOW..end
  - Aggregate RMSE / MAE / R² in psia

Outputs:
  alpha_sweep_results.csv          full table of (alpha, beta, RMSE_psia, R²)
  plots/alpha_sweep_curve.png      RMSE vs alpha (one line per beta)
  plots/alpha_sweep_heatmap.png    RMSE heatmap over (alpha, beta) grid
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT.parent.parent.parent / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW = 10
ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BETAS  = [0.05, 0.1, 0.2, 0.4]

scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale    = float(scaler_y.data_range_[0])

test_df = pd.read_csv(DATA_DIR / "test.csv")
test_df = test_df[test_df["mask"] == 1].copy()
print(f"Test rows: {len(test_df)} ({test_df['engine_id'].nunique()} engines)")
print(f"Sweep grid: {len(ALPHAS)} alphas x {len(BETAS)} betas = {len(ALPHAS)*len(BETAS)} combos\n")


def holt_recursion(series, alpha, beta):
    """1-step-ahead Holt forecasts on a 1-D series with FIXED alpha, beta.
    Initialization: ℓ_0 = y_0 ; b_0 = y_1 - y_0.
    Returns array of forecasts ŷ_t for t = 1..len(series)-1.
    """
    n = len(series)
    if n < 2:
        return np.array([], dtype=np.float64)
    L = float(series[0])
    B = float(series[1] - series[0])
    forecasts = np.empty(n - 1, dtype=np.float64)
    for t in range(n - 1):
        forecasts[t] = L + B                               # forecast for t+1
        # update with the actual at t+1 once observed
        y = float(series[t + 1])
        Ln = alpha * y + (1 - alpha) * (L + B)
        Bn = beta  * (Ln - L) + (1 - beta) * B
        L, B = Ln, Bn
    return forecasts


# Per-engine series cached
engines_data = []
for eid, grp in test_df.groupby("engine_id", sort=True):
    engines_data.append((int(eid), grp["sensor_11"].values.astype(np.float64)))

results = []
for alpha in ALPHAS:
    for beta in BETAS:
        all_pred, all_actual = [], []
        for eid, series in engines_data:
            fc = holt_recursion(series, alpha, beta)
            # We use forecasts for index t+1 = WINDOW..end-1, matching forecast.py
            # forecasts[t] is for series[t+1]; we want t+1 in [WINDOW+1, len-1] i.e. t in [WINDOW, len-2]
            for t in range(WINDOW, len(series) - 1):
                all_pred.append(fc[t])
                all_actual.append(series[t + 1])
        all_pred   = np.asarray(all_pred)
        all_actual = np.asarray(all_actual)
        err  = all_pred - all_actual
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae  = float(np.mean(np.abs(err)))
        ss_r = float(np.sum(err ** 2))
        ss_t = float(np.sum((all_actual - all_actual.mean()) ** 2))
        r2   = 1.0 - ss_r / ss_t
        results.append({
            "alpha": alpha, "beta": beta,
            "rmse_scaled": rmse, "rmse_psia": rmse * scale,
            "mae_psia": mae * scale, "r2": r2,
            "n": len(all_pred),
        })

df = pd.DataFrame(results).sort_values("rmse_psia").reset_index(drop=True)
df.to_csv(ROOT / "alpha_sweep_results.csv", index=False)

print("=== Top 10 (alpha, beta) combos by RMSE_psia ===")
print(df.head(10)[["alpha", "beta", "rmse_psia", "r2", "mae_psia"]].to_string(index=False))

print(f"\nBest: alpha={df.iloc[0]['alpha']}, beta={df.iloc[0]['beta']}  "
      f"-> RMSE_psia={df.iloc[0]['rmse_psia']:.4f}, R²={df.iloc[0]['r2']:.4f}")
print(f"Worst: alpha={df.iloc[-1]['alpha']}, beta={df.iloc[-1]['beta']}  "
      f"-> RMSE_psia={df.iloc[-1]['rmse_psia']:.4f}, R²={df.iloc[-1]['r2']:.4f}")

# ── Plot 1: RMSE vs alpha, one line per beta ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for beta in BETAS:
    sub = df[df["beta"] == beta].sort_values("alpha")
    ax.plot(sub["alpha"], sub["rmse_psia"], marker="o", label=f"beta={beta}")
# Mark the best
best = df.iloc[0]
ax.axhline(best["rmse_psia"], color="red", linestyle=":", linewidth=1, alpha=0.6,
           label=f"best: alpha={best['alpha']}, beta={best['beta']} -> RMSE={best['rmse_psia']:.4f}")
ax.set_xlabel("alpha (level smoothing)")
ax.set_ylabel("RMSE (psia)")
ax.set_title("Holt ExpSmoothing — RMSE vs alpha (one curve per beta)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "alpha_sweep_curve.png", dpi=130);  plt.close()

# ── Plot 2: heatmap RMSE over (alpha, beta) ──────────────────────────────────
grid = np.zeros((len(ALPHAS), len(BETAS)))
for i, a in enumerate(ALPHAS):
    for j, b in enumerate(BETAS):
        grid[i, j] = df[(df["alpha"] == a) & (df["beta"] == b)]["rmse_psia"].iloc[0]

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(BETAS)));  ax.set_xticklabels([f"beta={b}" for b in BETAS])
ax.set_yticks(range(len(ALPHAS)));  ax.set_yticklabels([f"alpha={a}" for a in ALPHAS])
ax.set_xlabel("beta (trend smoothing)")
ax.set_ylabel("alpha (level smoothing)")
ax.set_title("RMSE (psia) — Holt sweep over (alpha, beta)")
mid = grid.mean()
for i in range(len(ALPHAS)):
    for j in range(len(BETAS)):
        v = grid[i, j]
        ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=8,
                color="white" if v > mid else "black")
fig.colorbar(im, ax=ax, label="RMSE (psia, lower = better)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "alpha_sweep_heatmap.png", dpi=130);  plt.close()

print(f"\nResults saved to {ROOT / 'alpha_sweep_results.csv'}")
print(f"Plots saved in {PLOTS_DIR}")
