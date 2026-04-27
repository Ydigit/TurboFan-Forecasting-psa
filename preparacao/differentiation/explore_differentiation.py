"""
Explore differentiation variants on sensor_11.

  1st order:  Δ(t)  = sensor(t) - sensor(t-1)        — speed of change
  2nd order:  ΔΔ(t) = Δ(t) - Δ(t-1)                  — acceleration

Visual goal:
  - Does the differentiated series oscillate around 0 (stationary)?
  - Does it expose the degradation trend (e.g. Δ becomes consistently positive)?
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT      = Path(__file__).parent
DATA_CSV  = ROOT.parent.parent / "data" / "train.csv"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_CSV)
real = df[df["mask"] == 1].copy()

# Differentiate per motor (groupby) — first row per motor will be NaN
real["diff_1"] = real.groupby("engine_id")["sensor_11"].diff(1)
real["diff_2"] = real.groupby("engine_id")["diff_1"].diff(1)

top_engines = real.groupby("engine_id").size().sort_values(ascending=False).head(3).index.tolist()

# ── Plot 1: original + diff1 + diff2 for 3 motors ─────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=False)
for row, eid in enumerate(top_engines):
    g = real[real["engine_id"] == eid].reset_index(drop=True)
    cycle = np.arange(len(g))

    axes[row, 0].plot(cycle, g["sensor_11"], color="steelblue", linewidth=1.2)
    axes[row, 0].set_title(f"Motor {eid} — sensor_11 (original)")
    axes[row, 0].set_ylabel("escalado")
    axes[row, 0].grid(True, alpha=0.3)

    axes[row, 1].plot(cycle, g["diff_1"], color="tomato", linewidth=1.0)
    axes[row, 1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[row, 1].set_title(f"Motor {eid} — Δ (1ª ordem)")
    axes[row, 1].grid(True, alpha=0.3)

    axes[row, 2].plot(cycle, g["diff_2"], color="seagreen", linewidth=1.0)
    axes[row, 2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[row, 2].set_title(f"Motor {eid} — ΔΔ (2ª ordem)")
    axes[row, 2].grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel("ciclo")

plt.tight_layout()
out = PLOTS_DIR / "01_differentiation_series.png"
plt.savefig(out, dpi=130)
plt.close()
print(f"Saved: {out}")

# ── Plot 2: Histograms — check stationarity (centered around 0?) ───────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(real["sensor_11"].dropna(), bins=60, color="steelblue", alpha=0.85)
axes[0].axvline(real["sensor_11"].mean(), color="black", linestyle="--", label=f"mean={real['sensor_11'].mean():.3f}")
axes[0].set_title("Original sensor_11");  axes[0].set_xlabel("valor");  axes[0].legend();  axes[0].grid(True, alpha=0.3)

axes[1].hist(real["diff_1"].dropna(), bins=60, color="tomato", alpha=0.85)
axes[1].axvline(0, color="black", linestyle="--", label="0")
axes[1].axvline(real["diff_1"].mean(), color="purple", linestyle=":", label=f"mean={real['diff_1'].mean():.4f}")
axes[1].set_title("Δ (1ª ordem)");  axes[1].set_xlabel("valor");  axes[1].legend();  axes[1].grid(True, alpha=0.3)

axes[2].hist(real["diff_2"].dropna(), bins=60, color="seagreen", alpha=0.85)
axes[2].axvline(0, color="black", linestyle="--", label="0")
axes[2].axvline(real["diff_2"].mean(), color="purple", linestyle=":", label=f"mean={real['diff_2'].mean():.4f}")
axes[2].set_title("ΔΔ (2ª ordem)");  axes[2].set_xlabel("valor");  axes[2].legend();  axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out = PLOTS_DIR / "02_differentiation_histograms.png"
plt.savefig(out, dpi=130)
plt.close()
print(f"Saved: {out}")

# ── Plot 3: rolling mean of the differentiated series (drift?) ───────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, col, title, color in [
    (axes[0], "diff_1", "Δ — rolling mean (window=20) por motor", "tomato"),
    (axes[1], "diff_2", "ΔΔ — rolling mean (window=20) por motor", "seagreen"),
]:
    for eid in top_engines:
        g = real[real["engine_id"] == eid].reset_index(drop=True)
        rolling = g[col].rolling(20, min_periods=5).mean()
        ax.plot(rolling, label=f"motor {eid}", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title);  ax.set_xlabel("ciclo");  ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)

plt.tight_layout()
out = PLOTS_DIR / "03_differentiation_rolling_mean.png"
plt.savefig(out, dpi=130)
plt.close()
print(f"Saved: {out}")

print("\nDiagnostico estatistico:")
print(f"  sensor_11 mean = {real['sensor_11'].mean():.4f}   std = {real['sensor_11'].std():.4f}")
print(f"  diff_1    mean = {real['diff_1'].mean():.4f}   std = {real['diff_1'].std():.4f}   <- idealmente perto de 0 e estavel")
print(f"  diff_2    mean = {real['diff_2'].mean():.4f}   std = {real['diff_2'].std():.4f}   <- se ja e estavel em diff_1, diff_2 nao acrescenta")
print("\nLeitura visual:")
print("  - Se diff_1 oscila em torno de 0 com std pequeno -> bom para ARIMA, talvez para MLP")
print("  - Se diff_1 tem deriva positiva crescente -> indica degradacao acelerada")
print("  - Se diff_2 for praticamente ruido branco -> diferenciar 2x e exagero")
