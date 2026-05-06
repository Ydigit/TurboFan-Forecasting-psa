"""
Master comparison across ALL forecasting models.

Models compared:
  Univariate:
    - MLP baseline                (Modeling/MLP/univariate/mask_baseline)
    - ExpSmoothing (Holt-add)     (Modeling/ExponentialSmoothing/univariate)
    - ARIMA(0,1,1)                (Modeling/ARIMA/univariate)
    - LSTM                        (Modeling/LSTM/univariate)
  Multivariate:
    - SARIMAX(0,1,1) + 8 exog     (Modeling/ARIMA/multivariate)
    - LSTM (W=10, 9 features)     (Modeling/LSTM/multivariate)

Metrics computed (in psia and dimensionless):
  RMSE   — root mean squared error
  MAE    — mean absolute error
  R²     — coefficient of determination
  MAPE   — mean absolute percentage error (% — careful: large when actual~0)
  sMAPE  — symmetric MAPE (bounded 0–200%, robust to small actuals)
  MASE   — mean absolute scaled error vs in-sample naive (1.0 = same as naive)
  Theil_U — sqrt(MSE_model / MSE_naive); <1 means beats naive
  Bias   — mean residual (signed; tells direction of systematic error)
  DirAcc — directional accuracy (% of times sign(Δactual) == sign(Δpredicted))

Outputs:
  master_results.csv               full table of metrics per model
  plots/01_rmse_psia.png           bar chart RMSE psia
  plots/02_r2.png                  bar chart R²
  plots/03_theil_u.png             bar chart Theil's U
  plots/04_predicted_vs_actual_eng{N}.png   overlay first 3 engines, all models
  plots/05_residual_distributions.png       histograms per model
"""

import importlib.util
import sys as _sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt

ROOT      = Path(__file__).parent
PROJECT   = ROOT.parent.parent
DATA_DIR  = PROJECT / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

scaler_y = joblib.load(DATA_DIR / "scaler_y.pkl")
scale    = float(scaler_y.data_range_[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── 1. Helper: dynamic import for MLP/LSTM model classes ──────────────────
def _load_module(label, file_path):
    spec = importlib.util.spec_from_file_location(label, file_path)
    mod  = importlib.util.module_from_spec(spec)
    _sys.modules[label] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_pair(folder):
    """Load dataset.py (registered as 'dataset') and model.py from `folder`. Returns (ds_mod, model_mod)."""
    ds_mod  = _load_module("dataset", folder / "dataset.py")  # MUST register as 'dataset'
    mod_mod = _load_module(f"model_{folder.name}", folder / "model.py")
    return ds_mod, mod_mod


def _gen_predictions_mlp(folder, csv_test):
    """Generate predictions.csv equivalent (engine_id, t, actual, predicted) for an MLP/LSTM model."""
    ds_mod, mod_mod = _load_pair(folder)
    DSCls = next(c for n, c in vars(ds_mod).items()
                 if isinstance(c, type) and n.endswith("EngineDataset"))
    MdCls = next(c for n, c in vars(mod_mod).items()
                 if isinstance(c, type) and n.endswith("MLP"))

    test_ds = DSCls(csv_test)
    X = test_ds.X.to(device); M = test_ds.mask.to(device); y = test_ds.y.cpu().numpy()
    model = MdCls().to(device)
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(X, M).cpu().numpy()

    # Reconstruct (engine_id, t) for each sample by re-walking the test CSV
    df = pd.read_csv(csv_test)
    mask = df["mask"].values
    engines = df["engine_id"].values
    valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]
    return pd.DataFrame({
        "engine_id": engines[valid_t].astype(int),
        "t":         valid_t.astype(int),
        "actual":    y.astype(float),
        "predicted": pred.astype(float),
    })


def _gen_predictions_lstm_uni(folder, csv_test):
    """Same as MLP but model class is named *LSTM*."""
    ds_mod, mod_mod = _load_pair(folder)
    DSCls = next(c for n, c in vars(ds_mod).items()
                 if isinstance(c, type) and n.endswith("EngineDataset"))
    MdCls = next(c for n, c in vars(mod_mod).items()
                 if isinstance(c, type) and n.endswith("LSTM"))

    test_ds = DSCls(csv_test)
    X = test_ds.X.to(device); M = test_ds.mask.to(device); y = test_ds.y.cpu().numpy()
    model = MdCls().to(device)
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(X, M).cpu().numpy()

    df = pd.read_csv(csv_test)
    mask = df["mask"].values; engines = df["engine_id"].values
    valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]
    return pd.DataFrame({
        "engine_id": engines[valid_t].astype(int),
        "t":         valid_t.astype(int),
        "actual":    y.astype(float),
        "predicted": pred.astype(float),
    })


def _gen_predictions_lstm_multi(folder, csv_test, W=10, hidden=16):
    ds_mod, mod_mod = _load_pair(folder)
    DSCls = ds_mod.MultivariateEngineDataset
    MdCls = mod_mod.MultivariateLSTM

    EXOG_FEATURES = (DATA_DIR / "selected_features.txt").read_text().strip().splitlines()
    FEATURES      = ["sensor_11"] + EXOG_FEATURES

    test_ds = DSCls(csv_test, window=W, features=FEATURES)
    X = test_ds.X.to(device); M = test_ds.mask.to(device); y = test_ds.y.cpu().numpy()
    model = MdCls(input_size=len(FEATURES), hidden=hidden).to(device)
    model.load_state_dict(torch.load(folder / f"best_model_W{W}_h{hidden}.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(X, M).cpu().numpy()

    df = pd.read_csv(csv_test)
    mask = df["mask"].values; engines = df["engine_id"].values
    valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]
    return pd.DataFrame({
        "engine_id": engines[valid_t].astype(int),
        "t":         valid_t.astype(int),
        "actual":    y.astype(float),
        "predicted": pred.astype(float),
    })


# ─── 2. Build per-engine context for "previous value" lookups ──────────────
test_csv = DATA_DIR / "test.csv"
test_df  = pd.read_csv(test_csv)
test_df_real = test_df[test_df["mask"] == 1].copy()

# y_prev[engine_id][t] gives the actual value at row t for that engine (in psia space, scaled)
prev_lookup = {}
for eid, grp in test_df.groupby("engine_id"):
    arr = grp["sensor_11"].values
    # We index by global t in the padded test frame
    prev_lookup[int(eid)] = arr  # local index per engine; actual->global mapping done below

# Compute MAE_naive on training data (in-sample naive baseline for MASE)
train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df_real = train_df[train_df["mask"] == 1].copy()
naive_diffs = []
for _, grp in train_df_real.groupby("engine_id"):
    s = grp["sensor_11"].values
    if len(s) > 1:
        naive_diffs.extend(np.abs(np.diff(s)))
mae_naive_train = float(np.mean(naive_diffs))
print(f"MAE_naive (in-sample, train): {mae_naive_train:.4f} (scaled)  =  {mae_naive_train*scale:.4f} psia")


# ─── 3. Metric definitions ──────────────────────────────────────────────────
def compute_metrics(actuals_scaled, preds_scaled, prev_actuals_scaled):
    """All inputs in scaled space. Returns dict with metrics in scaled and psia where applicable."""
    a = np.asarray(actuals_scaled, dtype=np.float64)
    p = np.asarray(preds_scaled,    dtype=np.float64)
    prev = np.asarray(prev_actuals_scaled, dtype=np.float64)

    err  = p - a
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_r = float(np.sum(err ** 2)); ss_t = float(np.sum((a - a.mean()) ** 2))
    r2   = 1.0 - ss_r / ss_t

    # Inverse-transform to psia for percentage metrics that need original units
    a_psia = a * scale + scaler_y.data_min_[0]
    p_psia = p * scale + scaler_y.data_min_[0]

    mape  = float(np.mean(np.abs((a_psia - p_psia) / a_psia))) * 100
    smape = float(np.mean(2 * np.abs(p_psia - a_psia) / (np.abs(a_psia) + np.abs(p_psia)))) * 100

    # MASE — mean absolute scaled error (vs in-sample naive on TRAIN)
    mase = mae / mae_naive_train

    # Theil's U2 — sqrt(MSE_model) / sqrt(MSE_naive_on_test_period)
    naive_err = a - prev
    rmse_naive = float(np.sqrt(np.mean(naive_err ** 2)))
    theil_u = rmse / rmse_naive if rmse_naive > 0 else float("nan")

    # Directional accuracy
    da_correct = (np.sign(a - prev) == np.sign(p - prev)).mean()
    dir_acc = float(da_correct) * 100

    return {
        "RMSE_scaled": rmse,
        "RMSE_psia":   rmse * scale,
        "MAE_psia":    mae * scale,
        "R2":          r2,
        "MAPE_%":      mape,
        "sMAPE_%":     smape,
        "MASE":        mase,
        "Theil_U":     theil_u,
        "Bias_psia":   bias * scale,
        "DirAcc_%":    dir_acc,
    }


# ─── 4. Collect predictions per model ──────────────────────────────────────
def add_prev_actual(predictions_df):
    """Within each engine, prev_actual = previous row's actual (shift(1) on sorted t).

    Works regardless of whether `t` is global (MLP/LSTM) or local (ARIMA), as long
    as predictions are dense (no gaps) within each engine. Loses one row per engine
    (the first cycle, where no prior actual is available in the predictions file).
    """
    df = predictions_df.sort_values(["engine_id", "t"]).copy()
    df["prev_actual"] = df.groupby("engine_id")["actual"].shift(1)
    return df.dropna(subset=["prev_actual"]).reset_index(drop=True)


print("\nGathering predictions...")

models = []  # list of (name, type, df_with_actual_predicted)

# 1. MLP baseline
print("  - MLP baseline...")
mlp_pred = _gen_predictions_mlp(
    PROJECT / "Modeling" / "MLP" / "univariate" / "mask_baseline", test_csv)
models.append(("MLP baseline",     "univariate", mlp_pred))

# 2. ExpSmoothing
print("  - ExpSmoothing (Holt-add)...")
exp_csv = PROJECT / "Modeling" / "ExponentialSmoothing" / "univariate" / "predictions.csv"
models.append(("ExpSmoothing",     "univariate", pd.read_csv(exp_csv)))

# 3. ARIMA(0,1,1) univariate
print("  - ARIMA(0,1,1) univariate...")
arima_csv = PROJECT / "Modeling" / "ARIMA" / "univariate" / "predictions.csv"
models.append(("ARIMA(0,1,1)",     "univariate", pd.read_csv(arima_csv)))

# 4. LSTM univariate
print("  - LSTM univariate...")
lstm_uni_pred = _gen_predictions_lstm_uni(
    PROJECT / "Modeling" / "LSTM" / "univariate", test_csv)
models.append(("LSTM (uni)",       "univariate", lstm_uni_pred))

# 5. SARIMAX(0,1,1) multivariate
print("  - SARIMAX(0,1,1) + 8 exog...")
sarimax_csv = PROJECT / "Modeling" / "ARIMA" / "multivariate" / "predictions.csv"
models.append(("SARIMAX(0,1,1)+8", "multivariate", pd.read_csv(sarimax_csv)))

# 6. LSTM multivariate (W=10)
print("  - LSTM multivariate W=10...")
lstm_multi_pred = _gen_predictions_lstm_multi(
    PROJECT / "Modeling" / "LSTM" / "multivariate", test_csv, W=10, hidden=16)
models.append(("LSTM (multi, W=10)", "multivariate", lstm_multi_pred))


# ─── 5. Compute metrics ────────────────────────────────────────────────────
print("\nComputing metrics...")
records = []
for name, kind, dfp in models:
    dfp = add_prev_actual(dfp)
    m = compute_metrics(dfp["actual"].values, dfp["predicted"].values, dfp["prev_actual"].values)
    m["model"] = name
    m["type"]  = kind
    m["n"]     = len(dfp)
    records.append(m)

cols = ["model", "type", "n", "RMSE_psia", "MAE_psia", "R2", "MAPE_%", "sMAPE_%",
        "MASE", "Theil_U", "Bias_psia", "DirAcc_%"]
master = pd.DataFrame(records)[cols].sort_values("RMSE_psia").reset_index(drop=True)
master.to_csv(ROOT / "master_results.csv", index=False)

print("\n=== MASTER COMPARISON (sorted by RMSE_psia) ===")
print(master.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ─── 6. Plots ──────────────────────────────────────────────────────────────
def color_by_type(types):
    return ["steelblue" if t == "univariate" else "tomato" for t in types]

# Plot 1: RMSE psia
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(master["model"], master["RMSE_psia"], color=color_by_type(master["type"]), alpha=0.85)
for i, v in enumerate(master["RMSE_psia"]):
    ax.text(v, i, f"  {v:.4f}", va="center", fontsize=9)
ax.set_xlabel("RMSE (psia)")
ax.set_title("RMSE — all models (lower = better)\n(blue = univariate, red = multivariate)")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_rmse_psia.png", dpi=130);  plt.close()

# Plot 2: R²
m_r2 = master.sort_values("R2", ascending=False)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(m_r2["model"], m_r2["R2"], color=color_by_type(m_r2["type"]), alpha=0.85)
for i, v in enumerate(m_r2["R2"]):
    ax.text(v, i, f"  {v:.4f}", va="center", fontsize=9)
ax.set_xlabel("R²")
ax.set_title("R² — all models (higher = better)")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_r2.png", dpi=130);  plt.close()

# Plot 3: Theil's U
m_u = master.sort_values("Theil_U")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(m_u["model"], m_u["Theil_U"], color=color_by_type(m_u["type"]), alpha=0.85)
ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="naive baseline (U=1)")
for i, v in enumerate(m_u["Theil_U"]):
    ax.text(v, i, f"  {v:.4f}", va="center", fontsize=9)
ax.set_xlabel("Theil's U  (lower = better; <1 beats naive)")
ax.set_title("Theil's U — all models")
ax.legend();  ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_theil_u.png", dpi=130);  plt.close()

# Plot 4: predicted vs actual on first 3 engines, all models overlaid
unique_engines = sorted(test_df["engine_id"].unique())[:3]
fig, axes = plt.subplots(len(unique_engines), 1, figsize=(14, 3.5 * len(unique_engines)))
if len(unique_engines) == 1: axes = [axes]
colors = ["steelblue", "darkorange", "seagreen", "purple", "tomato", "brown"]
for ax, eid in zip(axes, unique_engines):
    base = next(dfp for nm, _, dfp in models if nm == "MLP baseline")
    base_eng = base[base["engine_id"] == eid].sort_values("t")
    ax.plot(base_eng["t"].values, base_eng["actual"].values, label="actual",
            color="black", linewidth=2, alpha=0.85)
    for (nm, kind, dfp), c in zip(models, colors):
        sub = dfp[dfp["engine_id"] == eid].sort_values("t")
        ax.plot(sub["t"].values, sub["predicted"].values, label=nm,
                color=c, linewidth=1.0, alpha=0.7)
    ax.set_title(f"Engine {eid} — all models 1-step-ahead")
    ax.set_xlabel("global t");  ax.set_ylabel("sensor_11 (scaled)")
    ax.legend(fontsize=8, loc="best");  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_predicted_vs_actual_3engines.png", dpi=130);  plt.close()

# Plot 5: residual distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes_flat = axes.flatten()
for ax, (nm, kind, dfp), c in zip(axes_flat, models, colors):
    res = (dfp["predicted"].values - dfp["actual"].values) * scale
    ax.hist(res, bins=40, color=c, alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{nm}\nbias={res.mean():+.4f} psia, RMSE={np.sqrt((res**2).mean()):.4f}",
                 fontsize=10)
    ax.set_xlabel("residual (psia)");  ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_residual_distributions.png", dpi=130);  plt.close()

print(f"\nPlots saved in {PLOTS_DIR}")
print(f"Master CSV saved at {ROOT / 'master_results.csv'}")
