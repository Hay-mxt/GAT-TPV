# models/ae_oc.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AEOC(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, bottleneck: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        # x: [B, D]
        z = self.enc(x)
        recon = self.dec(z)
        s = F.mse_loss(recon, x, reduction="none").mean(dim=1)
        return recon, z, s
