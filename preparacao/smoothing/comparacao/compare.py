"""
Compare baseline vs smoothing at multiple alphas (0.1, 0.3, 0.5) on the same test set.
All models predict raw next-cycle sensor_11. Smoothing applied consistently to
both train and test (best ML practice).
"""

import importlib.util
import sys as _sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

ROOT          = Path(__file__).parent
PROJECT       = ROOT.parent.parent.parent
DATA_DIR      = PROJECT / "data"
SMOOTH_TRAIN  = ROOT.parent / "treino"
SMOOTH_TX     = ROOT.parent / "transformed"
BASELINE_DIR  = PROJECT / "Modeling" / "MLP" / "univariate" / "mask_baseline"
PLOTS_DIR     = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW = 10
ALPHAS = [0.1, 0.3, 0.5]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale = float(scaler_y.scale_[0])


def _load_model_class(folder: Path):
    ds_spec = importlib.util.spec_from_file_location("dataset_local", folder / "dataset.py")
    ds = importlib.util.module_from_spec(ds_spec)
    _sys.modules["dataset"] = ds
    ds_spec.loader.exec_module(ds)
    m_spec = importlib.util.spec_from_file_location("model_local", folder / "model.py")
    m = importlib.util.module_from_spec(m_spec)
    m_spec.loader.exec_module(m)
    return m.UnivariateMLP


# ── Baseline ──────────────────────────────────────────────────────────────────
baseline_model = _load_model_class(BASELINE_DIR)().to(device)
baseline_model.load_state_dict(torch.load(BASELINE_DIR / "best_model.pt", map_location=device))
baseline_model.eval()

df_b = pd.read_csv(DATA_DIR / "test.csv")
raw_b = df_b["sensor_11"].values.astype(np.float32)
m_b   = df_b["mask"].values.astype(np.float32)
eng_b = df_b["engine_id"].values
valid_b = np.where((m_b[:-1] == 1) & (m_b[1:] == 1))[0]

X_b, M_b, Y_b = [], [], []
for t in valid_b:
    s = t - WINDOW + 1
    if s < 0:
        pad = -s
        wv = np.concatenate([np.zeros(pad, dtype=np.float32), raw_b[0:t + 1]])
        wm = np.concatenate([np.zeros(pad, dtype=np.float32), m_b[0:t + 1]])
    else:
        wv = raw_b[s:t + 1];  wm = m_b[s:t + 1]
    X_b.append(wv);  M_b.append(wm);  Y_b.append(raw_b[t + 1])
X_b = torch.tensor(np.stack(X_b)).to(device)
M_b = torch.tensor(np.stack(M_b)).to(device)
y_b = np.array(Y_b, dtype=np.float32)
with torch.no_grad():
    p_base = baseline_model(X_b, M_b).cpu().numpy()
rmse_b = float(np.sqrt(np.mean((p_base - y_b) ** 2)))
ss_res_b = float(np.sum((y_b - p_base) ** 2))
ss_tot_b = float(np.sum((y_b - y_b.mean()) ** 2))
r2_b = 1.0 - ss_res_b / ss_tot_b


# ── Smoothing variants ────────────────────────────────────────────────────────
results = {"baseline": (rmse_b, p_base, y_b, valid_b, eng_b)}

for a in ALPHAS:
    tag = f"a{int(a * 10):02d}"
    model = _load_model_class(SMOOTH_TRAIN)().to(device)
    model.load_state_dict(torch.load(SMOOTH_TRAIN / f"best_model_{tag}.pt", map_location=device))
    model.eval()

    # User explicitly chose to NOT smooth the test set: evaluate on raw test data.
    df = pd.read_csv(DATA_DIR / "test.csv")
    raw    = df["sensor_11"].values.astype(np.float32)
    mk     = df["mask"].values.astype(np.float32)
    eng    = df["engine_id"].values
    valid  = np.where((mk[:-1] == 1) & (mk[1:] == 1))[0]

    X, M, Y = [], [], []
    for t in valid:
        s = t - WINDOW + 1
        if s < 0:
            pad = -s
            wv = np.concatenate([np.zeros(pad, dtype=np.float32), raw[0:t + 1]])
            wm = np.concatenate([np.zeros(pad, dtype=np.float32), mk[0:t + 1]])
        else:
            wv = raw[s:t + 1];  wm = mk[s:t + 1]
        X.append(wv);  M.append(wm);  Y.append(raw[t + 1])

    X = torch.tensor(np.stack(X)).to(device)
    M = torch.tensor(np.stack(M)).to(device)
    y = np.array(Y, dtype=np.float32)
    with torch.no_grad():
        p = model(X, M).cpu().numpy()
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    results[a] = (rmse, r2, p, y, valid, eng)

# ── Print table ──────────────────────────────────────────────────────────────
print(f"\n{'Modelo':25s} {'RMSE_scaled':>12s} {'RMSE_psia':>12s} {'R2':>8s} {'delta vs base':>15s}")
print("-" * 80)
print(f"{'Baseline (raw)':25s} {rmse_b:12.4f} {rmse_b*scale:12.4f} {r2_b:8.4f} {'—':>15s}")
for a in ALPHAS:
    r, r2, *_ = results[a]
    delta = (r - rmse_b) / rmse_b * 100
    print(f"{f'Smoothing alpha={a}':25s} {r:12.4f} {r*scale:12.4f} {r2:8.4f} {delta:+14.2f}%")

# ── Bar chart RMSE + R2 side by side ─────────────────────────────────────────
labels   = ["baseline"] + [f"alpha={a}" for a in ALPHAS]
rmse_vals = [rmse_b * scale] + [results[a][0] * scale for a in ALPHAS]
r2_vals   = [r2_b]            + [results[a][1]         for a in ALPHAS]
colors   = ["lightgrey", "steelblue", "tomato", "seagreen"]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].bar(labels, rmse_vals, color=colors, alpha=0.85)
for i, v in enumerate(rmse_vals):
    axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
axes[0].set_ylabel("RMSE (psia)")
axes[0].set_title("RMSE — menor é melhor")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(labels, r2_vals, color=colors, alpha=0.85)
for i, v in enumerate(r2_vals):
    axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
axes[1].set_ylabel("R²")
axes[1].set_title("R² — maior é melhor")
axes[1].grid(True, alpha=0.3, axis="y")

fig.suptitle("Smoothing vs baseline (3 alphas)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_rmse_by_alpha.png", dpi=130);  plt.close()

# ── Predictions overlay on 3 engines (best alpha highlighted) ────────────────
unique_engines = sorted(pd.unique(eng_b))[:3]
fig, axes = plt.subplots(len(unique_engines), 1, figsize=(13, 3 * len(unique_engines)))
if len(unique_engines) == 1: axes = [axes]
for ax, eid in zip(axes, unique_engines):
    idx_b = np.array([i for i, t in enumerate(valid_b) if eng_b[t] == eid])
    ax.plot(y_b[idx_b],          label="actual",          color="steelblue", linewidth=1.6)
    ax.plot(p_base[idx_b],       label="baseline",        color="lightgrey", linewidth=1.0, alpha=0.9)
    smoothing_colors = ["tomato", "seagreen", "purple"]
    for a, c in zip(ALPHAS, smoothing_colors):
        _, _, p_a, _, valid_a, eng_a = results[a]
        idx_a = np.array([i for i, t in enumerate(valid_a) if eng_a[t] == eid])
        ax.plot(p_a[idx_a], label=f"alpha={a}", color=c, linewidth=1.1, alpha=0.85)
    ax.set_title(f"Engine {eid}");  ax.set_xlabel("ciclo (test)")
    ax.set_ylabel("sensor_11 (escalado)");  ax.legend(fontsize=8);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_predictions_by_alpha.png", dpi=130);  plt.close()

print(f"\nPlots saved in {PLOTS_DIR}")
