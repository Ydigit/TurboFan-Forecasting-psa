import torch
import torch.nn as nn


class MultivariateLSTM(nn.Module):
    """LSTM with F input features per timestep, 1 hidden recurrent layer + linear head.

    Forward expects X shape (B, W, F) and mask shape (B, W). Padded positions
    are zeroed before the recurrent computation.
    """

    def __init__(self, input_size: int, hidden: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        # x: (B, W, F)  ; mask: (B, W)
        x = x * mask.unsqueeze(-1)        # zero out padded timesteps
        out, _ = self.lstm(x)             # (B, W, hidden)
        last = out[:, -1, :]              # (B, hidden) — last timestep encoding
        return self.fc(last).squeeze(1)
