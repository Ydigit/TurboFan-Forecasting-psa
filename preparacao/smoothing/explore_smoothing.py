"""
Explore exponential smoothing variants on sensor_11.

For each motor (separately), apply exponential smoothing with different alphas.
Lower alpha = stronger smoothing. Higher alpha = closer to raw.

Visual goal: pick the alpha that removes noise WITHOUT killing the degradation trend.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT      = Path(__file__).parent
DATA_CSV  = ROOT.parent.parent / "data" / "train.csv"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

ALPHAS = [0.1, 0.3, 0.5]
SAMPLE_ENGINES = 3   # how many motors to plot

df = pd.read_csv(DATA_CSV)
real = df[df["mask"] == 1].copy()

# Apply EWM per motor (no cross-engine leakage)
for a in ALPHAS:
    col = f"smooth_a{a}"
    real[col] = real.groupby("engine_id")["sensor_11"].transform(
        lambda s: s.ewm(alpha=a, adjust=False).mean()
    )

# Pick first SAMPLE_ENGINES motors with most cycles (more interesting)
top_engines = real.groupby("engine_id").size().sort_values(ascending=False).head(SAMPLE_ENGINES).index.tolist()

fig, axes = plt.subplots(SAMPLE_ENGINES, 1, figsize=(14, 4 * SAMPLE_ENGINES))
if SAMPLE_ENGINES == 1: axes = [axes]

for ax, eid in zip(axes, top_engines):
    g = real[real["engine_id"] == eid].reset_index(drop=True)
    cycle = np.arange(len(g))

    ax.plot(cycle, g["sensor_11"], label="raw", color="lightgrey", linewidth=1, alpha=0.9)
    colors = ["steelblue", "tomato", "seagreen"]
    for a, c in zip(ALPHAS, colors):
        ax.plot(cycle, g[f"smooth_a{a}"], label=f"α={a}", color=c, linewidth=1.6)

    ax.set_title(f"Motor {eid}  ({len(g)} ciclos reais) — exponential smoothing")
    ax.set_xlabel("ciclo")
    ax.set_ylabel("sensor_11 (escalado)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = PLOTS_DIR / "01_smoothing_comparison.png"
plt.savefig(out, dpi=130)
plt.close()
print(f"Saved: {out}")

# Side-by-side zoomed comparison on 1 motor
eid = top_engines[0]
g = real[real["engine_id"] == eid].reset_index(drop=True)
cycle = np.arange(len(g))

fig, axes = plt.subplots(1, len(ALPHAS), figsize=(15, 4), sharey=True)
for ax, a in zip(axes, ALPHAS):
    ax.plot(cycle, g["sensor_11"], color="lightgrey", linewidth=1, label="raw")
    ax.plot(cycle, g[f"smooth_a{a}"], color="tomato", linewidth=1.5, label=f"α={a}")
    ax.set_title(f"α = {a}")
    ax.set_xlabel("ciclo")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("sensor_11 (escalado)")
fig.suptitle(f"Motor {eid} — smoothing isolado por α", y=1.02)
plt.tight_layout()
out = PLOTS_DIR / "02_smoothing_per_alpha.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("\nVisual reading guide:")
print("  alpha=0.1  -> muito suave, mas pode atrasar a tendencia")
print("  alpha=0.3  -> meio termo")
print("  alpha=0.5  -> quase igual ao raw, suaviza pouco")
