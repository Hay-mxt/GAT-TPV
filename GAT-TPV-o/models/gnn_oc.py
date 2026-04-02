# models/gnn_oc.py
# -*- coding: utf-8 -*-
"""
Graph 视角的一类自编码器（Graph-AE）。

说明：
- 在 data/processed/{dataset}/gnn/graphs.pkl 中，已经对每个 graph["node_feat"]
  做了 mean pooling 得到图级向量 v_g ∈ R^{D_g}。
- 因此这里的 GNNOC 实际上是一个作用在图级向量上的 MLP 自编码器。

接口约定：
    recon, z, s = model(x)
  - x:     [B, D_in]
  - recon: [B, D_in]  重构向量
  - z:     [B, D_latent] 潜表示
  - s:     [B]  一类分数（重构误差，越大越异常）

该接口与 AEOC / RNNOC 保持一致，便于 train_oc.py 统一训练。
"""

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

        # 这里实际是图级向量上的 MLP-AE
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
        # 对线性层做 Kaiming 初始化，训练更稳定
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """返回图级潜表示 z: [B, bottleneck]."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜表示重构图级特征 x: [B, D_in]."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        x: [B, D_in]
        返回:
            recon: [B, D_in]
            z:     [B, bottleneck]
            s:     [B] 一类分数 (MSE(recon, x))
        """
        z = self.encode(x)
        recon = self.decode(z)
        # Reconstruction error as anomaly score
        s = F.mse_loss(recon, x, reduction="none").mean(dim=-1)
        return recon, z, s
