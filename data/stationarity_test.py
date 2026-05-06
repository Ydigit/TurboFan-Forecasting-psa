"""
Stationarity tests for sensor_11.

Two complementary tests:
  - Augmented Dickey-Fuller (ADF):
      H0: series HAS a unit root (non-stationary)
      Reject H0 (p < 0.05) -> series is stationary
  - Kwiatkowski-Phillips-Schmidt-Shin (KPSS):
      H0: series IS stationary (around a constant or trend)
      Reject H0 (p < 0.05) -> series is non-stationary

The two tests have OPPOSITE null hypotheses, so a robust diagnosis combines both:
   ADF rejects + KPSS does NOT reject  -> stationary (consistent)
   ADF does NOT reject + KPSS rejects  -> non-stationary (consistent)
   Both reject / both accept           -> ambiguous

Per-engine, we test:
   1. raw sensor_11        (d=0)
   2. first difference     (d=1)
   3. second difference    (d=2)

Outputs:
   stationarity_results.csv        per-engine results
   plots/stationarity_series.png   3 sample engines x 3 differencing orders
   plots/stationarity_pvalues.png  histograms of ADF and KPSS p-values across engines
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

ALPHA = 0.05  # significance level

train_df = pd.read_csv(ROOT / "train.csv")
train_df = train_df[train_df["mask"] == 1].copy()
engines = sorted(train_df["engine_id"].unique())
print(f"Loaded {len(train_df)} real rows  |  {len(engines)} train engines\n")


def diff_series(s, k):
    """k-th difference of a 1D series."""
    out = s.copy()
    for _ in range(k):
        out = np.diff(out)
    return out


def safe_adf(series):
    """Returns (statistic, p-value) or (NaN, NaN) if test fails."""
    try:
        res = adfuller(series, autolag="AIC")
        return float(res[0]), float(res[1])
    except Exception:
        return float("nan"), float("nan")


def safe_kpss(series):
    """Returns (statistic, p-value)."""
    try:
        res = kpss(series, regression="c", nlags="auto")
        return float(res[0]), float(res[1])
    except Exception:
        return float("nan"), float("nan")


# ─── Run tests per engine x differencing order ─────────────────────────────
records = []
for d in (0, 1, 2):
    for eid in engines:
        s_raw = train_df[train_df["engine_id"] == eid]["sensor_11"].values
        s = diff_series(s_raw, d)
        if len(s) < 20:        # need enough points for the tests to be meaningful
            continue

        adf_stat, adf_p = safe_adf(s)
        kpss_stat, kpss_p = safe_kpss(s)

        records.append({
            "engine_id": int(eid),
            "d": d,
            "n_points": len(s),
            "adf_stat": adf_stat,
            "adf_p": adf_p,
            "adf_stationary": adf_p < ALPHA,
            "kpss_stat": kpss_stat,
            "kpss_p": kpss_p,
            "kpss_stationary": kpss_p > ALPHA,    # reverse logic
        })

df = pd.DataFrame(records)
df.to_csv(ROOT / "stationarity_results.csv", index=False)

# ─── Aggregate summary ─────────────────────────────────────────────────────
print("=" * 80)
print(f"STATIONARITY SUMMARY ACROSS {len(engines)} TRAIN ENGINES (alpha = {ALPHA})")
print("=" * 80)
print(f"\n{'d':>2s}  {'n':>3s}  {'ADF stationary':>16s}  {'KPSS stationary':>17s}  "
      f"{'ADF p mean':>11s}  {'KPSS p mean':>12s}")
print("-" * 80)

for d in (0, 1, 2):
    sub = df[df["d"] == d]
    if len(sub) == 0:
        continue
    adf_stat_pct = float(sub["adf_stationary"].mean()) * 100
    kpss_stat_pct = float(sub["kpss_stationary"].mean()) * 100
    adf_p_mean = float(sub["adf_p"].mean())
    kpss_p_mean = float(sub["kpss_p"].mean())

    label = {0: "raw    ", 1: "diff_1 ", 2: "diff_2 "}[d]
    print(f"{d:>2d}  {len(sub):>3d}  "
          f"{adf_stat_pct:>14.1f}%  {kpss_stat_pct:>15.1f}%  "
          f"{adf_p_mean:>11.4f}  {kpss_p_mean:>12.4f}   ({label.strip()})")

print("\nCOMBINED VERDICT (both tests must agree on stationarity):")
print("-" * 80)
for d in (0, 1, 2):
    sub = df[df["d"] == d]
    both_stationary = (sub["adf_stationary"] & sub["kpss_stationary"]).mean() * 100
    both_nonstat = ((~sub["adf_stationary"]) & (~sub["kpss_stationary"])).mean() * 100
    label = {0: "raw    ", 1: "diff_1 ", 2: "diff_2 "}[d]
    print(f"  d={d} ({label.strip()}):  "
          f"both tests say STATIONARY    = {both_stationary:>5.1f}%")
    print(f"          both tests say NON-stationary= {both_nonstat:>5.1f}%")

# ─── Visualization 1: sample engines, 3 levels of differencing ────────────
sample_engines = engines[:3]
fig, axes = plt.subplots(len(sample_engines), 3, figsize=(15, 3 * len(sample_engines)))
if len(sample_engines) == 1:
    axes = axes.reshape(1, -1)
for row, eid in enumerate(sample_engines):
    s_raw = train_df[train_df["engine_id"] == eid]["sensor_11"].values
    for col, d in enumerate((0, 1, 2)):
        s = diff_series(s_raw, d)
        ax = axes[row, col]
        ax.plot(s, color=["steelblue", "tomato", "seagreen"][col], linewidth=0.9)
        ax.axhline(np.mean(s), color="black", linestyle=":", linewidth=0.8, alpha=0.6,
                   label=f"mean={np.mean(s):.3f}")
        title_label = {0: "raw (d=0)", 1: "1st diff (d=1)", 2: "2nd diff (d=2)"}[d]
        # also include p-values
        sub = df[(df["engine_id"] == eid) & (df["d"] == d)]
        if len(sub):
            adf_p = sub["adf_p"].iloc[0]
            kpss_p = sub["kpss_p"].iloc[0]
            ax.set_title(f"Engine {eid} — {title_label}\nADF p={adf_p:.3f} | KPSS p={kpss_p:.3f}",
                         fontsize=10)
        else:
            ax.set_title(f"Engine {eid} — {title_label}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlabel("cycle"); ax.set_ylabel("value")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "stationarity_series.png", dpi=130); plt.close()

# ─── Visualization 2: distribution of ADF and KPSS p-values per d ─────────
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
for col, d in enumerate((0, 1, 2)):
    sub = df[df["d"] == d]
    axes[0, col].hist(sub["adf_p"], bins=20, color="steelblue", alpha=0.85)
    axes[0, col].axvline(ALPHA, color="red", linestyle="--", linewidth=1, label=f"α={ALPHA}")
    axes[0, col].set_title(f"ADF p-values — d={d}\n(p<{ALPHA} → stationary)\nstationary: {sub['adf_stationary'].mean()*100:.1f}%",
                           fontsize=10)
    axes[0, col].set_xlabel("ADF p-value"); axes[0, col].legend(fontsize=8); axes[0, col].grid(True, alpha=0.3)

    axes[1, col].hist(sub["kpss_p"], bins=20, color="tomato", alpha=0.85)
    axes[1, col].axvline(ALPHA, color="red", linestyle="--", linewidth=1, label=f"α={ALPHA}")
    axes[1, col].set_title(f"KPSS p-values — d={d}\n(p>{ALPHA} → stationary)\nstationary: {sub['kpss_stationary'].mean()*100:.1f}%",
                           fontsize=10)
    axes[1, col].set_xlabel("KPSS p-value"); axes[1, col].legend(fontsize=8); axes[1, col].grid(True, alpha=0.3)

fig.suptitle("Stationarity test p-value distributions across 80 train engines", fontsize=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "stationarity_pvalues.png", dpi=130); plt.close()

print(f"\nResults: {ROOT / 'stationarity_results.csv'}")
print(f"Plots:   {PLOTS_DIR}")
