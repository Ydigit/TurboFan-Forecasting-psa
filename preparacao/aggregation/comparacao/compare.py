"""
Compare gran=3 vs gran=5 vs baseline.

CAVEAT: aggregation predicts a different target (mean of N flights) than baseline
(single next flight). RMSE values are NOT directly comparable, but they tell us
which granularity is internally most stable.
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
AGG_TRAIN     = ROOT.parent / "treino"
AGG_TX        = ROOT.parent / "transformed"
BASELINE_DIR  = PROJECT / "MLP" / "univariate" / "mask_baseline"
PLOTS_DIR     = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW = 10
GRANS  = [3, 5]
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


def eval_baseline():
    """Re-evaluate baseline on raw test set."""
    m = _load_model_class(BASELINE_DIR)().to(device)
    m.load_state_dict(torch.load(BASELINE_DIR / "best_model.pt", map_location=device))
    m.eval()

    df = pd.read_csv(DATA_DIR / "test.csv")
    s  = df["sensor_11"].values.astype(np.float32)
    mk = df["mask"].values.astype(np.float32)
    valid = np.where((mk[:-1] == 1) & (mk[1:] == 1))[0]
    X, M, Y = [], [], []
    for t in valid:
        st = t - WINDOW + 1
        if st < 0:
            pad = -st
            wv = np.concatenate([np.zeros(pad, dtype=np.float32), s[0:t + 1]])
            wm = np.concatenate([np.zeros(pad, dtype=np.float32), mk[0:t + 1]])
        else:
            wv = s[st:t + 1];  wm = mk[st:t + 1]
        X.append(wv);  M.append(wm);  Y.append(s[t + 1])
    X = torch.tensor(np.stack(X)).to(device)
    M = torch.tensor(np.stack(M)).to(device)
    y = np.array(Y, dtype=np.float32)
    with torch.no_grad():
        p = m(X, M).cpu().numpy()
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return rmse, r2, len(y)


def eval_agg(N):
    m = _load_model_class(AGG_TRAIN)().to(device)
    m.load_state_dict(torch.load(AGG_TRAIN / f"best_model_gran{N}.pt", map_location=device))
    m.eval()

    df = pd.read_csv(AGG_TX / f"test_gran{N}.csv")
    s  = df["sensor_11"].values.astype(np.float32)
    mk = df["mask"].values.astype(np.float32)
    valid = np.where((mk[:-1] == 1) & (mk[1:] == 1))[0]
    X, M, Y = [], [], []
    for t in valid:
        st = t - WINDOW + 1
        if st < 0:
            pad = -st
            wv = np.concatenate([np.zeros(pad, dtype=np.float32), s[0:t + 1]])
            wm = np.concatenate([np.zeros(pad, dtype=np.float32), mk[0:t + 1]])
        else:
            wv = s[st:t + 1];  wm = mk[st:t + 1]
        X.append(wv);  M.append(wm);  Y.append(s[t + 1])
    X = torch.tensor(np.stack(X)).to(device)
    M = torch.tensor(np.stack(M)).to(device)
    y = np.array(Y, dtype=np.float32)
    with torch.no_grad():
        p = m(X, M).cpu().numpy()
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return rmse, r2, len(y)


print("Evaluating baseline on test.csv...")
rmse_b, r2_b, n_b = eval_baseline()
print(f"  baseline:           RMSE_scaled={rmse_b:.4f}   RMSE_psia={rmse_b*scale:.4f}   R2={r2_b:.4f}   N={n_b}")

results = {1: (rmse_b, r2_b, n_b)}
for N in GRANS:
    print(f"Evaluating aggregation gran={N}...")
    r, r2, n = eval_agg(N)
    results[N] = (r, r2, n)
    print(f"  gran={N}:             RMSE_scaled={r:.4f}   RMSE_psia={r*scale:.4f}   R2={r2:.4f}   N={n}")

# Bar chart RMSE + R2 side by side
labels    = [f"gran=1\n(baseline)"] + [f"gran={N}\n(mean of {N})" for N in GRANS]
rmse_vals = [results[1][0] * scale] + [results[N][0] * scale for N in GRANS]
r2_vals   = [results[1][1]]          + [results[N][1]          for N in GRANS]
colors    = ["lightgrey", "steelblue", "tomato"]

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

fig.suptitle("Aggregation — RMSE & R² em diferentes granularidades\n(targets diferentes — não comparáveis em valor absoluto)",
             fontsize=10)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_rmse_by_granularity.png", dpi=130);  plt.close()
print(f"\nPlots saved in {PLOTS_DIR}")
print("\nNOTA: gran=3 e gran=5 prevêem 'média de N voos' — RMSE em psia mas em escala diferente do baseline.")
print("      O baseline (gran=1) prevê o próximo voo único.")
print("      Comparação directa do número não é justa, mas valores baixos indicam previsão consistente.")
