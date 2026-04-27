import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WINDOW = 10


class UnivariateEngineDataset(Dataset):
    """Sliding window on sensor_11 — NO mask input. Returns (X, y)."""

    def __init__(self, csv_path, window: int = WINDOW):
        df = pd.read_csv(csv_path)
        series = df["sensor_11"].values.astype(np.float32)
        mask   = df["mask"].values.astype(np.float32)

        valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]

        X_list, y_list = [], []
        for t in valid_t:
            start = t - window + 1
            if start < 0:
                pad = -start
                win_vals = np.concatenate([np.zeros(pad, dtype=np.float32), series[0:t + 1]])
            else:
                win_vals = series[start:t + 1]

            X_list.append(win_vals)
            y_list.append(series[t + 1])

        self.X = torch.tensor(np.stack(X_list))
        self.y = torch.tensor(np.array(y_list, dtype=np.float32))
        self.window = window

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
