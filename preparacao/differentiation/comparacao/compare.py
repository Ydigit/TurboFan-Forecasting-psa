"""
Compare baseline (raw input) vs diff_1 vs diff_2 on the same test set.
All three predict the raw next-cycle sensor_11.
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
DIFF_TRAIN    = ROOT.parent / "treino"
DIFF_TX       = ROOT.parent / "transformed"
BASELINE_DIR  = PROJECT / "Modeling" / "MLP" / "univariate" / "mask_baseline"
PLOTS_DIR     = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW = 10
ORDERS = [1, 2]
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


def build_windows(values: np.ndarray, mask: np.ndarray, target: np.ndarray, valid_t: np.ndarray):
    X, M, Y = [], [], []
    for t in valid_t:
        s = t - WINDOW + 1
        if s < 0:
            pad = -s
            wv = np.concatenate([np.zeros(pad, dtype=np.float32), values[0:t + 1]])
            wm = np.concatenate([np.zeros(pad, dtype=np.float32), mask[0:t + 1]])
        else:
            wv = values[s:t + 1];  wm = mask[s:t + 1]
        X.append(wv);  M.append(wm);  Y.append(target[t])
    return torch.tensor(np.stack(X)).to(device), torch.tensor(np.stack(M)).to(device), np.array(Y, dtype=np.float32)


# ── Baseline ──────────────────────────────────────────────────────────────────
baseline_model = _load_model_class(BASELINE_DIR)().to(device)
baseline_model.load_state_dict(torch.load(BASELINE_DIR / "best_model.pt", map_location=device))
baseline_model.eval()

raw_df = pd.read_csv(DATA_DIR / "test.csv")
raw    = raw_df["sensor_11"].values.astype(np.float32)
mask   = raw_df["mask"].values.astype(np.float32)
engines = raw_df["engine_id"].values
valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]
target_next = raw[1:].copy()
target_next = np.append(target_next, 0.0)

def metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return rmse, mae, r2


Xr, M, y = build_windows(raw, mask, target_next, valid_t)
with torch.no_grad():
    p_base = baseline_model(Xr, M).cpu().numpy()
rmse_b, mae_b, r2_b = metrics(y, p_base)


# ── Differentiation orders ────────────────────────────────────────────────────
results = {0: (rmse_b, mae_b, r2_b, p_base)}

for k in ORDERS:
    model = _load_model_class(DIFF_TRAIN)().to(device)
    model.load_state_dict(torch.load(DIFF_TRAIN / f"best_model_diff{k}.pt", map_location=device))
    model.eval()

    df = pd.read_csv(DIFF_TX / f"test_diff{k}.csv")
    diff_vals = df["sensor_11"].values.astype(np.float32)
    target_k  = df["target"].values.astype(np.float32)
    mask_k    = df["mask"].values.astype(np.float32)
    valid_k   = np.where((mask_k[:-1] == 1) & (mask_k[1:] == 1))[0]

    Xd, Mk, yk = build_windows(diff_vals, mask_k, target_k, valid_k)
    with torch.no_grad():
        pk = model(Xd, Mk).cpu().numpy()
    rmse_k, mae_k, r2_k = metrics(yk, pk)
    results[k] = (rmse_k, mae_k, r2_k, pk)


# ── Print table ──────────────────────────────────────────────────────────────
print(f"\n{'Modelo':25s} {'RMSE_scaled':>12s} {'RMSE_psia':>12s} {'MAE_psia':>12s} {'R2':>8s} {'delta vs base':>15s}")
print("-" * 92)
print(f"{'Baseline (raw)':25s} {rmse_b:12.4f} {rmse_b*scale:12.4f} {mae_b*scale:12.4f} {r2_b:8.4f} {'—':>15s}")
for k in ORDERS:
    r, mae, r2, _ = results[k]
    delta = (r - rmse_b) / rmse_b * 100
    print(f"{f'Differentiation diff_{k}':25s} {r:12.4f} {r*scale:12.4f} {mae*scale:12.4f} {r2:8.4f} {delta:+14.2f}%")


# ── Bar chart RMSE + R2 side by side ─────────────────────────────────────────
labels    = ["Baseline"] + [f"diff_{k}" for k in ORDERS]
rmse_vals = [rmse_b * scale] + [results[k][0] * scale for k in ORDERS]
r2_vals   = [r2_b]            + [results[k][2]         for k in ORDERS]
colors    = ["lightgrey", "seagreen", "purple"]

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

fig.suptitle("Differentiation vs baseline")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_rmse_comparison.png", dpi=130);  plt.close()


# ── Predictions overlay on first 3 engines ──────────────────────────────────
unique_engines = sorted(pd.unique(engines))[:3]
fig, axes = plt.subplots(len(unique_engines), 1, figsize=(13, 3 * len(unique_engines)))
if len(unique_engines) == 1: axes = [axes]
for ax, eid in zip(axes, unique_engines):
    idx = np.array([i for i, t in enumerate(valid_t) if engines[t] == eid])
    ax.plot(y[idx],      label="actual",   color="steelblue", linewidth=1.6)
    ax.plot(p_base[idx], label="baseline", color="lightgrey", linewidth=1.0, alpha=0.9)
    diff_colors = ["seagreen", "purple"]
    for k, c in zip(ORDERS, diff_colors):
        _, _, _, pk = results[k]
        ax.plot(pk[idx], label=f"diff_{k}", color=c, linewidth=1.1, alpha=0.85)
    ax.set_title(f"Engine {eid} — predictions");  ax.set_xlabel("ciclo (test)")
    ax.set_ylabel("sensor_11 (escalado)");  ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_predictions_overlay.png", dpi=130);  plt.close()
print(f"\nPlots saved in {PLOTS_DIR}")
