
import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNOC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        bottleneck: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        act = nn.ReLU

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            act(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
  
        z = self.encode(x)
        recon = self.decode(z)
        # Reconstruction error as anomaly score
        s = F.mse_loss(recon, x, reduction="none").mean(dim=-1)
        return recon, z, s
