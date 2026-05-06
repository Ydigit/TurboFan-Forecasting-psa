"""
Feature selection for multivariate models — identify redundant variables and
keep the top N informative, non-redundant ones.

Strategy (greedy, correlation-based):
  1. Rank features by |corr| with target (sensor_11).
  2. Pick the top-ranked.
  3. Iterate down the ranking: add a feature only if its |corr| with every
     already-selected feature is below REDUNDANCY_THRESH.
  4. Stop when TOP_N features are selected.

Outputs:
  data/selected_features.txt           list of selected feature names (one per line)
  data/plots/feature_selection_heatmap.png   full correlation heatmap
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT      = Path(__file__).parent
TRAIN_CSV = ROOT / "train.csv"
OUT_TXT   = ROOT / "selected_features.txt"
PLOT_PNG  = ROOT / "plots" / "feature_selection_heatmap.png"
PLOT_PNG.parent.mkdir(exist_ok=True)

TARGET            = "sensor_11"
TOP_N             = 8
REDUNDANCY_THRESH = 0.95   # any feature with |r| >= this to a selected one is dropped

CANDIDATES = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21",
    "setting_1", "setting_2", "setting_3",
]

df = pd.read_csv(TRAIN_CSV)
df = df[df["mask"] == 1].copy()
print(f"Loaded {len(df)} real training rows from {TRAIN_CSV.name}\n")

# Correlation with target, sorted by absolute value
corr_target = df[CANDIDATES + [TARGET]].corr()[TARGET].drop(TARGET)
abs_corr    = corr_target.abs().sort_values(ascending=False)

print("Features ranked by |corr| with sensor_11:")
print(f"  {'feature':<14s} {'corr':>8s}")
print("  " + "-" * 24)
for f in abs_corr.index:
    print(f"  {f:<14s} {corr_target[f]:>+8.4f}")

# Greedy selection with redundancy filter
selected = []
for f in abs_corr.index:
    if not selected:
        selected.append(f)
        continue
    max_inter = max(abs(df[f].corr(df[s])) for s in selected)
    if max_inter < REDUNDANCY_THRESH:
        selected.append(f)
    if len(selected) >= TOP_N:
        break

print(f"\n=== Selected top {len(selected)} non-redundant features (threshold |r| < {REDUNDANCY_THRESH}) ===")
print(f"  {'feature':<14s} {'corr w/ target':>16s}")
print("  " + "-" * 32)
for f in selected:
    print(f"  {f:<14s} {corr_target[f]:>+16.4f}")

removed = [f for f in CANDIDATES if f not in selected]
print(f"\nRemoved features (redundant or low correlation):")
print(f"  {'feature':<14s} {'reason':>50s}")
print("  " + "-" * 64)
for f in removed:
    rel = max(((s, abs(df[f].corr(df[s]))) for s in selected), key=lambda x: x[1])
    if rel[1] >= REDUNDANCY_THRESH:
        reason = f"redundant with {rel[0]} (|r|={rel[1]:.3f})"
    else:
        reason = f"top-{TOP_N} cap reached  (best: {rel[0]} |r|={rel[1]:.3f})"
    print(f"  {f:<14s} {reason:>50s}")

OUT_TXT.write_text("\n".join(selected))
print(f"\nSelected features written to {OUT_TXT}")

# Heatmap
corr_full = df[CANDIDATES + [TARGET]].corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr_full.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(corr_full.columns)))
ax.set_xticklabels(corr_full.columns, rotation=90)
ax.set_yticks(range(len(corr_full.columns)))
ax.set_yticklabels(corr_full.columns)

# Highlight selected features
for f in selected:
    i = list(corr_full.columns).index(f)
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="black", lw=2))

for i in range(len(corr_full)):
    for j in range(len(corr_full)):
        v = corr_full.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.6 else "black")

fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="correlation")
ax.set_title(f"Feature correlation matrix — selected (boxed): {', '.join(selected)}",
             fontsize=10)
plt.tight_layout()
plt.savefig(PLOT_PNG, dpi=130)
plt.close()
print(f"Heatmap saved to {PLOT_PNG}")
