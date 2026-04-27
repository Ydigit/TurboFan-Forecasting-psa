import torch
import torch.nn as nn
from dataset import WINDOW


class UnivariateMLP(nn.Module):
    def __init__(self, window: int = WINDOW, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, mask):
        return self.net(torch.cat([x, mask], dim=1)).squeeze(1)
