"""
Apply exponential smoothing to sensor_11 per motor — TRAIN ONLY.

The user explicitly chose NOT to smooth the test set. The model trains on
smoothed inputs and is evaluated on raw test inputs (distribution mismatch
by design).

Each alpha generates one train CSV in transformed/:
  train_smoothed_a01.csv  (alpha=0.1)
  train_smoothed_a03.csv  (alpha=0.3)
  train_smoothed_a05.csv  (alpha=0.5)

In each CSV, the sensor_11 column is overwritten with the smoothed values.
The target column (= raw next-cycle sensor_11, scaled) is untouched.
"""

from pathlib import Path
import pandas as pd

ALPHAS   = [0.1, 0.3, 0.5]
ROOT     = Path(__file__).parent
DATA_DIR = ROOT.parent.parent / "data"
OUT_DIR  = ROOT / "transformed"
OUT_DIR.mkdir(exist_ok=True)


def smooth_train(csv_in: Path, csv_out: Path, alpha: float):
    df = pd.read_csv(csv_in)
    real = df["mask"] == 1
    smoothed = df.loc[real].groupby("engine_id")["sensor_11"].transform(
        lambda s: s.ewm(alpha=alpha, adjust=False).mean()
    )
    df.loc[real, "sensor_11"] = smoothed.values
    df.to_csv(csv_out, index=False)


for a in ALPHAS:
    tag = f"a{int(a * 10):02d}"
    smooth_train(DATA_DIR / "train.csv", OUT_DIR / f"train_smoothed_{tag}.csv", a)
    print(f"  alpha={a}  ->  train_smoothed_{tag}.csv  (test stays raw)")

print("Done.")
