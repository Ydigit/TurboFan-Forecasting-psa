import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WINDOW = 10


class DiffEngineDataset(Dataset):
    """
    Input window built from diff_1 of sensor_11.
    Target = raw target column (next-cycle absolute value).

    The model has to learn to integrate the diff window back into an absolute prediction.
    """

    def __init__(self, csv_path, window: int = WINDOW):
        df = pd.read_csv(csv_path)
        diff_series = df["sensor_11"].values.astype(np.float32)   # already diff_1
        target_arr  = df["target"].values.astype(np.float32)
        mask        = df["mask"].values.astype(np.float32)

        valid_t = np.where((mask[:-1] == 1) & (mask[1:] == 1))[0]

        X_list, M_list, y_list = [], [], []
        for t in valid_t:
            start = t - window + 1
            if start < 0:
                pad = -start
                win_vals = np.concatenate([np.zeros(pad, dtype=np.float32), diff_series[0:t + 1]])
                win_mask = np.concatenate([np.zeros(pad, dtype=np.float32), mask[0:t + 1]])
            else:
                win_vals = diff_series[start:t + 1]
                win_mask = mask[start:t + 1]
            X_list.append(win_vals)
            M_list.append(win_mask)
            y_list.append(target_arr[t])

        self.X    = torch.tensor(np.stack(X_list))
        self.mask = torch.tensor(np.stack(M_list))
        self.y    = torch.tensor(np.array(y_list, dtype=np.float32))

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.mask[i], self.y[i]
