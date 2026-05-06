import torch
import torch.nn as nn
from dataset import WINDOW


class UnivariateLSTM(nn.Module):
    """LSTM with 1 hidden layer + linear head.

    Input X has shape (B, W) — reshaped to (B, W, 1) for LSTM.
    The mask is multiplied element-wise to ensure padded positions feed zero
    into the recurrent computation (consistent with the MLP setup).
    """

    def __init__(self, hidden: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        # x: (B, W) ; mask: (B, W)
        x = x.unsqueeze(-1)            # (B, W, 1)
        x = x * mask.unsqueeze(-1)     # zero out padded positions
        out, _ = self.lstm(x)          # (B, W, hidden)
        last = out[:, -1, :]           # (B, hidden) — final timestep encoding
        return self.fc(last).squeeze(1)
