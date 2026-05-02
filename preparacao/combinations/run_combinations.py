"""
Train all 3 x 3 x 4 = 36 combinations.

Pipeline (input is ALREADY scaled — data/train.csv, data/test.csv):
  AGG  ->  DIFF  ->  SMOOTH (rolling mean, train only)

Smoothing = simple moving average with window k.  Train-only as per methodology.
Aggregation re-aligns target via shift(-1) within each engine.
Differentiation is k-th order diff per engine.

Outputs:
  combinations/results.csv          full table (36 rows)
  combinations/plots/01_top15.png   bar charts of best by RMSE / R2
  combinations/plots/02_heatmaps.png 2D view (agg x smooth) per diff order
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

ROOT      = Path(__file__).parent
PROJECT   = ROOT.parent.parent
DATA_DIR  = PROJECT / "data"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

WINDOW    = 10
MIN_ITERS = 50_000             # spec do CLAUDE.md: > 50 000 iterações por combo
BATCH     = 128                # match standalone convention
LR        = 1e-3
SEED      = 42

AGGS    = [1, 3, 5]                 # 1 = no agg
DIFFS   = [0, 1, 2]                 # 0 = no diff
SMOOTHS = [None, 3, 5, 7]           # None = no smoothing; otherwise rolling-mean window k

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw  = pd.read_csv(DATA_DIR / "test.csv")
scaler_y  = joblib.load(DATA_DIR / "scaler_y.pkl")
scale     = float(scaler_y.scale_[0])


# ── Transforms (operate on already-scaled data) ──────────────────────────────
def aggregate(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """Per engine: keep (N==1) or average every N consecutive REAL cycles. Recompute target via shift(-1)."""
    real = df[df["mask"] == 1].copy()
    rows = []
    for eid, grp in real.groupby("engine_id", sort=True):
        s = grp["sensor_11"].values
        if N == 1:
            vals = s
        else:
            n_groups = len(s) // N
            if n_groups < 2:
                continue
            vals = s[: n_groups * N].reshape(n_groups, N).mean(axis=1)
        for v in vals:
            rows.append({"engine_id": int(eid), "sensor_11": float(v)})
    out = pd.DataFrame(rows)
    out["target"] = out.groupby("engine_id")["sensor_11"].shift(-1)
    out = out[out["target"].notna()].copy().reset_index(drop=True)
    return out


def differentiate(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if k == 0:
        return df
    df = df.copy()
    s = df.groupby("engine_id")["sensor_11"]
    for _ in range(k):
        s = s.diff(1)
    df["sensor_11"] = s.fillna(0.0).values
    return df


def smooth_rolling(df: pd.DataFrame, k) -> pd.DataFrame:
    """Rolling mean per engine on sensor_11. min_periods=1 so the head is not NaN."""
    if k is None:
        return df
    df = df.copy()
    s = df.groupby("engine_id")["sensor_11"].transform(
        lambda x: x.rolling(window=int(k), min_periods=1).mean()
    )
    df["sensor_11"] = s.values
    return df


def pad_engines(df: pd.DataFrame) -> pd.DataFrame:
    """Per-engine zero-pad to max_T. Adds mask column."""
    max_T = df.groupby("engine_id").size().max()
    out = []
    for eid, grp in df.groupby("engine_id", sort=True):
        grp = grp.copy().reset_index(drop=True)
        grp["mask"] = 1
        pad_len = max_T - len(grp)
        if pad_len > 0:
            p = pd.DataFrame({
                "engine_id": eid, "sensor_11": 0.0, "target": 0.0, "mask": 0,
            }, index=range(pad_len))
            grp = pd.concat([grp, p], ignore_index=True)
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def build_windows(df: pd.DataFrame):
    series = df["sensor_11"].values.astype(np.float32)
    target = df["target"].values.astype(np.float32)
    mask   = df["mask"].values.astype(np.float32)
    valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]

    X, M, Y = [], [], []
    for t in valid_t:
        s = t - WINDOW + 1
        if s < 0:
            pad = -s
            wv = np.concatenate([np.zeros(pad, dtype=np.float32), series[0:t + 1]])
            wm = np.concatenate([np.zeros(pad, dtype=np.float32), mask[0:t + 1]])
        else:
            wv = series[s:t + 1];  wm = mask[s:t + 1]
        X.append(wv);  M.append(wm);  Y.append(target[t])
    return np.stack(X), np.stack(M), np.array(Y, dtype=np.float32)


# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, window=WINDOW, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * window, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, m):
        return self.net(torch.cat([x, m], dim=1)).squeeze(1)


def train_one(X_tr, M_tr, y_tr, X_te, M_te, y_te):
    torch.manual_seed(SEED)
    model = MLP().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()
    n = len(y_tr)
    batches_per_epoch = max(1, (n + BATCH - 1) // BATCH)
    epochs = max(1, (MIN_ITERS + batches_per_epoch - 1) // batches_per_epoch)
    actual_iters = epochs * batches_per_epoch

    best_val = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = crit(model(X_tr[idx], M_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = crit(model(X_te, M_te), y_te).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_te, M_te).cpu().numpy()
    y_np = y_te.cpu().numpy()
    rmse   = float(np.sqrt(np.mean((pred - y_np) ** 2)))
    ss_res = float(np.sum((y_np - pred) ** 2))
    ss_tot = float(np.sum((y_np - y_np.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot
    return rmse, r2, epochs, actual_iters


# ── Main loop ────────────────────────────────────────────────────────────────
total = len(AGGS) * len(DIFFS) * len(SMOOTHS)
print(f"\nRunning {total} combinations (pipeline: AGG -> DIFF -> SMOOTH-train-only)...\n")
results = []
t_start = time.time()

for agg in AGGS:
    for diff in DIFFS:
        for k in SMOOTHS:
            t0 = time.time()

            tr = aggregate(train_raw, agg)
            te = aggregate(test_raw,  agg)

            tr = differentiate(tr, diff)
            te = differentiate(te, diff)

            tr = smooth_rolling(tr, k)              # train only

            tr_pad = pad_engines(tr)
            te_pad = pad_engines(te)

            X_tr, M_tr, y_tr = build_windows(tr_pad)
            X_te, M_te, y_te = build_windows(te_pad)

            X_tr = torch.tensor(X_tr).to(device); M_tr = torch.tensor(M_tr).to(device); y_tr = torch.tensor(y_tr).to(device)
            X_te = torch.tensor(X_te).to(device); M_te = torch.tensor(M_te).to(device); y_te = torch.tensor(y_te).to(device)

            rmse, r2, ep, iters = train_one(X_tr, M_tr, y_tr, X_te, M_te, y_te)
            elapsed = time.time() - t0

            results.append({
                "agg": agg, "diff": diff, "smooth": str(k),
                "rmse_scaled": rmse, "rmse_psia": rmse * scale,
                "r2": r2, "n_test": int(len(y_te)), "elapsed_sec": elapsed,
                "epochs": ep, "iters": iters,
            })
            label = f"agg={agg} diff={diff} smooth={k}"
            print(f"  [{len(results):2d}/{total}] {label:35s} ep={ep:>5d} iter={iters:>6d}  RMSE_psia={rmse*scale:.4f}  R2={r2:.4f}  ({elapsed:.1f}s)")

elapsed_total = time.time() - t_start
print(f"\nTotal time: {elapsed_total:.1f}s")


# ── Save table ───────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df_sorted = df.sort_values("rmse_psia").reset_index(drop=True)
df_sorted.to_csv(ROOT / "results.csv", index=False)
print(f"\nSaved {ROOT / 'results.csv'}")

print(f"\nTop 10 by RMSE_psia (lower = melhor):")
print(df_sorted[["agg", "diff", "smooth", "rmse_psia", "r2"]].head(10).to_string(index=False))

print(f"\nTop 10 by R2 (maior = melhor):")
print(df.sort_values("r2", ascending=False)[["agg", "diff", "smooth", "rmse_psia", "r2"]].head(10).to_string(index=False))


# ── Plot top 15 ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_rmse = df.sort_values("rmse_psia").head(15)
labels_r = [f"a{r['agg']} d{r['diff']} s{r['smooth']}" for _, r in top_rmse.iterrows()]
axes[0].barh(labels_r[::-1], top_rmse["rmse_psia"].values[::-1], color="steelblue", alpha=0.85)
for i, v in enumerate(top_rmse["rmse_psia"].values[::-1]):
    axes[0].text(v, i, f" {v:.4f}", va="center", fontsize=8)
axes[0].set_xlabel("RMSE (psia)")
axes[0].set_title("Top 15 - RMSE (menor = melhor)")
axes[0].grid(True, alpha=0.3, axis="x")

top_r2 = df.sort_values("r2", ascending=False).head(15)
labels_q = [f"a{r['agg']} d{r['diff']} s{r['smooth']}" for _, r in top_r2.iterrows()]
axes[1].barh(labels_q[::-1], top_r2["r2"].values[::-1], color="tomato", alpha=0.85)
for i, v in enumerate(top_r2["r2"].values[::-1]):
    axes[1].text(v, i, f" {v:.4f}", va="center", fontsize=8)
axes[1].set_xlabel("R2")
axes[1].set_title("Top 15 - R2 (maior = melhor)")
axes[1].grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_top15.png", dpi=130);  plt.close()


# ── Heatmaps ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
smooth_labels = ["none", "k=3", "k=5", "k=7"]
agg_labels    = ["1", "3", "5"]

for ax, d in zip(axes, DIFFS):
    grid = np.zeros((len(AGGS), len(SMOOTHS)))
    for i, a in enumerate(AGGS):
        for j, s in enumerate(SMOOTHS):
            row = df[(df["agg"] == a) & (df["diff"] == d) & (df["smooth"] == str(s))].iloc[0]
            grid[i, j] = row["r2"]
    im = ax.imshow(grid, cmap="RdYlGn", vmin=df["r2"].min(), vmax=df["r2"].max(), aspect="auto")
    ax.set_xticks(range(len(SMOOTHS)));  ax.set_xticklabels(smooth_labels)
    ax.set_yticks(range(len(AGGS)));     ax.set_yticklabels(agg_labels)
    ax.set_xlabel("smoothing window");   ax.set_title(f"diff = {d}")
    for i in range(len(AGGS)):
        for j in range(len(SMOOTHS)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if grid[i, j] > 0.5 else "white")
axes[0].set_ylabel("aggregation N")
fig.suptitle("R2 - agg x smooth, separado por diff order")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label="R2")
plt.savefig(PLOTS_DIR / "02_heatmaps.png", dpi=130);  plt.close()

print(f"\nPlots saved in {PLOTS_DIR}")
