import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

TRAIN_RATIO = 0.80
SEED        = 42

DROP_SENSORS = {"sensor_1", "sensor_5", "sensor_6",
                "sensor_10", "sensor_16", "sensor_18", "sensor_19"}

DATA_DIR = Path(__file__).parent
RAW_CSV  = DATA_DIR.parent / "FD001_train_timeseries.csv"

# ── Load & assign engine IDs ─────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV)
df["engine_id"] = (df["timeline"].diff() > 1).cumsum().astype(int)

drop_cols = DROP_SENSORS | {"timeline"}
keep_cols = [c for c in df.columns if c not in drop_cols]
df = df[keep_cols]

feature_cols = [c for c in keep_cols if c not in {"engine_id", "target"}]

# ── 80/20 split at engine level ───────────────────────────────────────────────
rng = np.random.default_rng(SEED)
engines = df["engine_id"].unique()
rng.shuffle(engines)

n_train       = int(len(engines) * TRAIN_RATIO)
train_engines = engines[:n_train]
test_engines  = engines[n_train:]

train_df = df[df["engine_id"].isin(train_engines)].copy()
test_df  = df[df["engine_id"].isin(test_engines)].copy()

# ── Scale features + target (fit on train real cycles only) ──────────────────
train_real = train_df  # all rows before padding are real

scaler_X = StandardScaler()
train_df[feature_cols] = scaler_X.fit_transform(train_df[feature_cols])
test_df[feature_cols]  = scaler_X.transform(test_df[feature_cols])
joblib.dump(scaler_X, DATA_DIR / "scaler_X.pkl")

scaler_y = StandardScaler()
train_df[["target"]] = scaler_y.fit_transform(train_df[["target"]])
test_df[["target"]]  = scaler_y.transform(test_df[["target"]])
joblib.dump(scaler_y, DATA_DIR / "scaler_y.pkl")

# ── Post-padding: pad each engine to max_cycles with zeros ────────────────────
# mask=1 real cycle, mask=0 padding
max_cycles = df.groupby("engine_id").size().max()  # 361
print(f"Max cycles per engine: {max_cycles}")

def pad_engines(subset_df, max_len):
    rows = []
    for eid, grp in subset_df.groupby("engine_id", sort=True):
        grp = grp.copy().reset_index(drop=True)
        grp["mask"] = 1

        T = len(grp)
        pad_len = max_len - T
        if pad_len > 0:
            pad = pd.DataFrame(
                np.zeros((pad_len, len(grp.columns))),
                columns=grp.columns
            )
            pad["engine_id"] = eid
            pad["mask"]      = 0
            grp = pd.concat([grp, pad], ignore_index=True)

        rows.append(grp)

    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "t", out.index)   # global timeline index
    return out

print("Padding training engines...")
train_out = pad_engines(train_df, max_cycles)
print("Padding test engines...")
test_out  = pad_engines(test_df,  max_cycles)

train_out.to_csv(DATA_DIR / "train.csv", index=False)
test_out.to_csv(DATA_DIR  / "test.csv",  index=False)

print(f"\ntrain.csv: {len(train_out)} linhas ({len(train_engines)} motores x {max_cycles} timesteps)")
print(f"test.csv : {len(test_out)} linhas ({len(test_engines)} motores x {max_cycles} timesteps)")
print(f"Colunas  : {train_out.columns.tolist()}")
print(f"\nPadding ratio train: {(train_out['mask']==0).mean():.1%}")
print(f"Padding ratio test : {(test_out['mask']==0).mean():.1%}")
