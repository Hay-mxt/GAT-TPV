
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNOC(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.out = nn.Linear(hidden_dim, input_dim)

        self._init_weights()

    def _init_weights(self):
        
        nn.init.xavier_uniform_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
      
        _, h = self.encoder(x)           # h: [num_layers, B, hidden_dim]
        z = h[-1]                        # [B, hidden_dim]
        return z

    def decode(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
      
        out, _ = self.decoder(x, h0)     # [B, T, hidden_dim]
        recon = self.out(out)            # [B, T, D]
        return recon

    def forward(self, x: torch.Tensor):
   
        # encode
        _, h = self.encoder(x)           # h: [num_layers, B, hidden_dim]
        h0 = h
        z = h[-1]                        # [B, hidden_dim]

        # decode
        recon = self.decode(x, h0)

        # one-class score: reconstruction error
        s = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))
        return recon, z, s
