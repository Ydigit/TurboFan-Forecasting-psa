"""
Compare mask/ vs no_mask/ baselines on the test set.

Metrics:
  - Overall MSE, RMSE, MAE (scaled units)
  - Same metrics split by window type:
      * CLEAN windows   — no padding inside (pure-history case)
      * TRANSITION windows — at least 1 padding bit in the window
    The mask should help most on TRANSITION windows.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT.parent.parent.parent / "data"
MASK_DIR   = ROOT / "mask"
NOMASK_DIR = ROOT / "no_mask"
OUT_DIR    = ROOT / "comparison_plots"
OUT_DIR.mkdir(exist_ok=True)

WINDOW = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load both models ─────────────────────────────────────────────────────────
sys.path.insert(0, str(MASK_DIR))
from model import UnivariateMLP as MLP_Mask
model_mask = MLP_Mask().to(device)
model_mask.load_state_dict(torch.load(MASK_DIR / "best_model.pt", map_location=device))
model_mask.eval()
sys.path.pop(0)

sys.path.insert(0, str(NOMASK_DIR))
# reimport clean (model.py defines the same class name in the other file)
import importlib, model as _nm_model_module
importlib.reload(_nm_model_module)
MLP_NoMask = _nm_model_module.UnivariateMLP
model_nomask = MLP_NoMask().to(device)
model_nomask.load_state_dict(torch.load(NOMASK_DIR / "best_model.pt", map_location=device))
model_nomask.eval()
sys.path.pop(0)

# ── Build test windows ───────────────────────────────────────────────────────
df     = pd.read_csv(DATA_DIR / "test.csv")
series = df["sensor_11"].values.astype(np.float32)
mask   = df["mask"].values.astype(np.float32)

valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]

X_list, M_list, y_list, has_pad = [], [], [], []
for t in valid_t:
    start = t - WINDOW + 1
    if start < 0:
        p = -start
        win_vals = np.concatenate([np.zeros(p, dtype=np.float32), series[0:t + 1]])
        win_mask = np.concatenate([np.zeros(p, dtype=np.float32), mask[0:t + 1]])
    else:
        win_vals = series[start:t + 1]
        win_mask = mask[start:t + 1]
    X_list.append(win_vals)
    M_list.append(win_mask)
    y_list.append(series[t + 1])
    has_pad.append((win_mask == 0).any())

X    = torch.tensor(np.stack(X_list)).to(device)
M    = torch.tensor(np.stack(M_list)).to(device)
y    = torch.tensor(np.array(y_list, dtype=np.float32)).to(device)
has_pad = np.array(has_pad)

# ── Predict with both models ─────────────────────────────────────────────────
with torch.no_grad():
    p_mask   = model_mask(X, M).cpu().numpy()
    p_nomask = model_nomask(X).cpu().numpy()

y_np   = y.cpu().numpy()
err_m  = p_mask   - y_np
err_nm = p_nomask - y_np

# ── Unscale to sensor_11 original units (psia) using scaler_y ────────────────
scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale = float(scaler_y.data_range_[0])   # MinMax: max - min of target in psia

def metrics(err, label, unit_scale):
    mse  = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(err)))
    return {
        "label": label,
        "n": len(err),
        "MSE_scaled":  mse,
        "RMSE_scaled": rmse,
        "MAE_scaled":  mae,
        "RMSE_real":   rmse * unit_scale,   # in psia
        "MAE_real":    mae  * unit_scale,
    }

results = []
for tag, errs in [("MASK", err_m), ("NO_MASK", err_nm)]:
    results.append(metrics(errs,               f"{tag} — ALL",        scale))
    results.append(metrics(errs[~has_pad],     f"{tag} — CLEAN",      scale))
    results.append(metrics(errs[ has_pad],     f"{tag} — TRANSITION", scale))

# ── Print nicely ─────────────────────────────────────────────────────────────
print(f"\nTotal test samples: {len(y_np)}")
print(f"  clean (no padding in window)     : {(~has_pad).sum()}  ({(~has_pad).mean():.1%})")
print(f"  transition (padding in window)   : { has_pad.sum()}  ({ has_pad.mean():.1%})")

hdr = f"{'group':25s} {'n':>6s} {'MSE':>10s} {'RMSE':>10s} {'MAE':>10s} {'RMSE(psia)':>12s} {'MAE(psia)':>12s}"
print("\n" + hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['label']:25s} {r['n']:6d} "
          f"{r['MSE_scaled']:10.5f} {r['RMSE_scaled']:10.5f} {r['MAE_scaled']:10.5f} "
          f"{r['RMSE_real']:12.4f} {r['MAE_real']:12.4f}")

# ── Side-by-side bar plot (RMSE, scaled and real) ────────────────────────────
groups = ["ALL", "CLEAN", "TRANSITION"]
rmse_m  = [r["RMSE_scaled"] for r in results if r["label"].startswith("MASK")]
rmse_nm = [r["RMSE_scaled"] for r in results if r["label"].startswith("NO_MASK")]
rmse_m_real  = [r["RMSE_real"] for r in results if r["label"].startswith("MASK")]
rmse_nm_real = [r["RMSE_real"] for r in results if r["label"].startswith("NO_MASK")]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(len(groups));  w = 0.35

axes[0].bar(x - w/2, rmse_m,  w, label="MASK",    color="steelblue")
axes[0].bar(x + w/2, rmse_nm, w, label="NO_MASK", color="tomato")
axes[0].set_xticks(x);  axes[0].set_xticklabels(groups)
axes[0].set_ylabel("RMSE (scaled units)")
axes[0].set_title("RMSE by window type (scaled)")
axes[0].legend();  axes[0].grid(True, alpha=0.3, axis="y")
for i, (a, b) in enumerate(zip(rmse_m, rmse_nm)):
    axes[0].text(i - w/2, a, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
    axes[0].text(i + w/2, b, f"{b:.4f}", ha="center", va="bottom", fontsize=8)

axes[1].bar(x - w/2, rmse_m_real,  w, label="MASK",    color="steelblue")
axes[1].bar(x + w/2, rmse_nm_real, w, label="NO_MASK", color="tomato")
axes[1].set_xticks(x);  axes[1].set_xticklabels(groups)
axes[1].set_ylabel("RMSE (psia — original units)")
axes[1].set_title("RMSE by window type (original units)")
axes[1].legend();  axes[1].grid(True, alpha=0.3, axis="y")
for i, (a, b) in enumerate(zip(rmse_m_real, rmse_nm_real)):
    axes[1].text(i - w/2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].text(i + w/2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "01_rmse_by_window_type.png", dpi=130)
plt.close()

# ── Error distribution (histogram of residuals) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
bins = np.linspace(-2.5, 2.5, 60)

for ax, (errs, title) in zip(axes, [
    (err_m,  "MASK — residuals (pred - actual)"),
    (err_nm, "NO_MASK — residuals (pred - actual)"),
]):
    ax.hist(errs[~has_pad], bins=bins, alpha=0.6, label="clean",      color="steelblue")
    ax.hist(errs[ has_pad], bins=bins, alpha=0.6, label="transition", color="tomato")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("residual (scaled)");  ax.set_ylabel("count")
    ax.set_title(title);  ax.legend();  ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "02_residuals_histogram.png", dpi=130)
plt.close()

print(f"\nComparison plots saved in {OUT_DIR}")
