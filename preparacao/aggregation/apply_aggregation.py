"""
Downsample by averaging consecutive flights of the same motor.

Granularity N: each output row = mean of N consecutive real cycles.
After aggregation, target is recomputed as the next aggregated sensor_11.
Padding is rebuilt to match the longest aggregated motor.

Output: train_gran{N}.csv, test_gran{N}.csv for N in [3, 5].
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT     = Path(__file__).parent
DATA_DIR = ROOT.parent.parent / "data"
OUT_DIR  = ROOT / "transformed"
OUT_DIR.mkdir(exist_ok=True)

GRANS = [3, 5]


def downsample(df: pd.DataFrame, N: int) -> pd.DataFrame:
    real = df[df["mask"] == 1].copy()

    rows = []
    for eid, grp in real.groupby("engine_id", sort=True):
        s = grp["sensor_11"].values
        n_groups = len(s) // N           # drop the last partial chunk
        if n_groups < 2:
            continue                     # too short to forecast
        agg = s[: n_groups * N].reshape(n_groups, N).mean(axis=1)

        for i, val in enumerate(agg):
            rows.append({"engine_id": int(eid), "super_cycle": i, "sensor_11": float(val)})
    out = pd.DataFrame(rows)

    # Re-build target = next sensor_11 within same motor
    out["target"] = out.groupby("engine_id")["sensor_11"].shift(-1)

    # Drop last super-cycle of each motor (no target)
    valid = out["target"].notna()
    out = out[valid].copy()

    # Re-pad to a rectangular grid (max super-cycles)
    max_T = out.groupby("engine_id").size().max()
    padded = []
    for eid, grp in out.groupby("engine_id", sort=True):
        grp = grp.reset_index(drop=True)
        grp["mask"] = 1
        pad_len = max_T - len(grp)
        if pad_len > 0:
            pad = pd.DataFrame({
                "engine_id":   eid,
                "super_cycle": range(len(grp), len(grp) + pad_len),
                "sensor_11":   0.0,
                "target":      0.0,
                "mask":        0,
            })
            grp = pd.concat([grp, pad], ignore_index=True)
        padded.append(grp)
    out = pd.concat(padded, ignore_index=True)
    out.insert(0, "t", range(len(out)))
    return out


for csv_name in ["train.csv", "test.csv"]:
    df = pd.read_csv(DATA_DIR / csv_name)
    for N in GRANS:
        agg = downsample(df, N)
        stem = "train" if csv_name == "train.csv" else "test"
        out_path = OUT_DIR / f"{stem}_gran{N}.csv"
        agg.to_csv(out_path, index=False)
        n_motors = agg["engine_id"].nunique()
        n_real   = (agg["mask"] == 1).sum()
        print(f"  {csv_name:10s}  N={N}  ->  {out_path.name}  ({n_motors} motores, {n_real} super-cycles reais)")

print("Done.")
