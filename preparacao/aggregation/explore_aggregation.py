"""
Explore aggregation features built from sensor_11.

For each timestep t, compute summary stats over a rolling window of the past N cycles
within the same motor. These become candidate EXTRA features to feed the MLP.

Variants tested:
  - rolling mean over window 7   ("estado da semana")
  - rolling mean over window 30  ("estado do mês")
  - rolling max  over window 30  ("pior caso do mês")
  - rolling std  over window 7   ("volatilidade da semana")

Visual goal: pick 2 aggregations whose lines look genuinely informative
(not just delayed copies of the original signal).
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

# Rolling stats per motor (no leakage across engines)
gb = real.groupby("engine_id")["sensor_11"]
real["rmean_7"]  = gb.transform(lambda s: s.rolling(7,  min_periods=1).mean())
real["rmean_30"] = gb.transform(lambda s: s.rolling(30, min_periods=1).mean())
real["rmax_30"]  = gb.transform(lambda s: s.rolling(30, min_periods=1).max())
real["rstd_7"]   = gb.transform(lambda s: s.rolling(7,  min_periods=2).std())

top_engines = real.groupby("engine_id").size().sort_values(ascending=False).head(3).index.tolist()

# ── Plot 1: original + all aggregations overlaid (per motor) ──────────────
fig, axes = plt.subplots(len(top_engines), 1, figsize=(14, 4 * len(top_engines)))
if len(top_engines) == 1: axes = [axes]

for ax, eid in zip(axes, top_engines):
    g = real[real["engine_id"] == eid].reset_index(drop=True)
    cycle = np.arange(len(g))

    ax.plot(cycle, g["sensor_11"], color="lightgrey", linewidth=1, alpha=0.9, label="raw")
    ax.plot(cycle, g["rmean_7"],   color="steelblue", linewidth=1.4, label="mean (7)")
    ax.plot(cycle, g["rmean_30"],  color="tomato",    linewidth=1.4, label="mean (30)")
    ax.plot(cycle, g["rmax_30"],   color="seagreen",  linewidth=1.4, linestyle="--", label="max (30)")

    ax.set_title(f"Motor {eid}  ({len(g)} ciclos) — rolling aggregations")
    ax.set_xlabel("ciclo");  ax.set_ylabel("sensor_11 (escalado)")
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

plt.tight_layout()
out = PLOTS_DIR / "01_aggregations_overlay.png"
plt.savefig(out, dpi=130)
plt.close()
print(f"Saved: {out}")

# ── Plot 2: rolling std on its own (volatility — different scale) ──────────
fig, axes = plt.subplots(len(top_engines), 1, figsize=(14, 3 * len(top_engines)))
if len(top_engines) == 1: axes = [axes]

for ax, eid in zip(axes, top_engines):
    g = real[real["engine_id"] == eid].reset_index(drop=True)
    cycle = np.arange(len(g))
    ax.plot(cycle, g["rstd_7"], color="purple", linewidth=1.2)
    ax.set_title(f"Motor {eid} — rolling std (window=7)")
    ax.set_xlabel("ciclo");  ax.set_ylabel("std (escalado)")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = PLOTS_DIR / "02_aggregation_rolling_std.png"
plt.savefig(out, dpi=130)
plt.close()
print(f"Saved: {out}")

# ── Plot 3: correlation between aggregations and target (= sensor_11 t+1) ──
agg_cols = ["sensor_11", "rmean_7", "rmean_30", "rmax_30", "rstd_7"]
corr = real[agg_cols + ["target"]].corr()["target"].drop("target").sort_values(key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["tomato" if v > 0 else "steelblue" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Correlação de cada agregação com o target (sensor_11 em t+1)")
ax.set_xlabel("Pearson r")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
out = PLOTS_DIR / "03_aggregation_correlation_with_target.png"
plt.savefig(out, dpi=130)
plt.close()
print(f"Saved: {out}")

print("\nCorrelação com target:")
for k, v in corr.items():
    print(f"  {k:12s}  r = {v:+.4f}")

print("\nLeitura visual:")
print("  • mean(7) e mean(30): se forem visualmente quase iguais ao raw, são redundantes")
print("  • max(30): bom se subir em escada (sinal clássico de degradação monotónica)")
print("  • std(7): se aumentar perto do fim de vida, é proxy de instabilidade pré-falha")
print("\nEstratégia: escolher 2 agregações com correlação alta com o target")
print("           E que sejam diferentes entre si (não duas médias parecidas).")
