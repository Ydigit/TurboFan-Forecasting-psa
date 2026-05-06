import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultivariateEngineDataset(Dataset):
    """Sliding-window dataset for multivariate LSTM forecasting of sensor_11.

    Returns (X, mask_window, y):
      X     shape (W, F)  — window of F features over W timesteps
      mask  shape (W,)    — 1 where real cycle, 0 where padding
      y     scalar        — sensor_11 at the next cycle (target)

    Features:  endogenous (sensor_11) followed by all exog feature columns.
    """

    def __init__(self, csv_path, window: int, features):
        df = pd.read_csv(csv_path)
        self.window = int(window)
        self.features = list(features)

        target = df["sensor_11"].values.astype(np.float32)
        feat_arr = df[self.features].values.astype(np.float32)
        mask = df["mask"].values.astype(np.float32)

        valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]
        F = feat_arr.shape[1]

        X_list, M_list, y_list = [], [], []
        for t in valid_t:
            start = t - self.window + 1
            if start < 0:
                pad = -start
                wv = np.concatenate([
                    np.zeros((pad, F), dtype=np.float32),
                    feat_arr[0:t + 1],
                ])
                wm = np.concatenate([
                    np.zeros(pad, dtype=np.float32),
                    mask[0:t + 1],
                ])
            else:
                wv = feat_arr[start:t + 1]
                wm = mask[start:t + 1]
            X_list.append(wv)
            M_list.append(wm)
            y_list.append(target[t + 1])

        self.X    = torch.tensor(np.stack(X_list))         # (N, W, F)
        self.mask = torch.tensor(np.stack(M_list))         # (N, W)
        self.y    = torch.tensor(np.array(y_list, dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.mask[i], self.y[i]
