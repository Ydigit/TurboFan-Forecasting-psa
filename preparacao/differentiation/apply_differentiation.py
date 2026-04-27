"""
Replace sensor_11 with its k-th order difference per motor (k in {1, 2}).

  diff_1(t) = sensor_11(t)   - sensor_11(t-1)
  diff_2(t) = diff_1(t)      - diff_1(t-1)

Cycles without enough history get 0.0. target column kept untouched (raw next-cycle).
"""

from pathlib import Path
import pandas as pd

ROOT     = Path(__file__).parent
DATA_DIR = ROOT.parent.parent / "data"
OUT_DIR  = ROOT / "transformed"
OUT_DIR.mkdir(exist_ok=True)

ORDERS = [1, 2]


def differentiate(csv_in: Path, csv_out: Path, order: int):
    df = pd.read_csv(csv_in)
    real = df["mask"] == 1

    diffed = df.loc[real].groupby("engine_id")["sensor_11"]
    for _ in range(order):
        diffed = diffed.diff(1)
    diffed = diffed.fillna(0.0)

    df.loc[real, "sensor_11"] = diffed.values
    df.to_csv(csv_out, index=False)
    print(f"  {csv_in.name} -> {csv_out.name}  (diff_{order})")


print("Applying differentiation...")
for k in ORDERS:
    differentiate(DATA_DIR / "train.csv", OUT_DIR / f"train_diff{k}.csv", k)
    differentiate(DATA_DIR / "test.csv",  OUT_DIR / f"test_diff{k}.csv",  k)
print("Done.")
