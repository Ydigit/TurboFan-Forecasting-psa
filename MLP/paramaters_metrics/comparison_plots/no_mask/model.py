import torch
import torch.nn as nn
from MLP.univariate.comparison_plots.no_mask.dataset import WINDOW


class UnivariateMLP(nn.Module):
    """MLP that receives only W sensor values — no mask input."""

    def __init__(self, window: int = WINDOW, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
