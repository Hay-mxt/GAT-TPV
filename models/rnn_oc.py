# models/rnn_oc.py
# -*- coding: utf-8 -*-
"""
RNN 视角的一类自编码器（RNN-AE）。

输入:
  x: [B, T, D]  流量序列特征 (例如按时间窗口/包序列展开)

结构:
  - encoder: GRU, 读入序列，取最后一层 hidden 作为 z
  - decoder: GRU, 以原始 x 为输入 (teacher forcing)，
             以 encoder 的 hidden 作为初始状态，重构整个序列
  - out:     线性层，将 hidden 投影回输入维度

接口:
  recon, z, s = model(x)
    - recon: [B, T, D] 重构序列
    - z:     [B, hidden_dim] 序列级潜表示
    - s:     [B] 一类分数 (MSE(recon, x))
"""

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
        # 对线性层初始化；GRU 内部有官方初始化，通常无需额外修改
        nn.init.xavier_uniform_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        返回 z: [B, hidden_dim] 取最后一层 GRU 的最终隐状态
        """
        _, h = self.encoder(x)           # h: [num_layers, B, hidden_dim]
        z = h[-1]                        # [B, hidden_dim]
        return z

    def decode(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        x:  [B, T, D]  作为 decoder 的输入 (teacher forcing)
        h0: [num_layers, B, hidden_dim] encoder 的最终隐状态
        返回 recon: [B, T, D]
        """
        out, _ = self.decoder(x, h0)     # [B, T, hidden_dim]
        recon = self.out(out)            # [B, T, D]
        return recon

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]
        返回:
          recon: [B, T, D]
          z:     [B, hidden_dim]
          s:     [B] = MSE(recon, x) 按 (T, D) 平均
        """
        # encode
        _, h = self.encoder(x)           # h: [num_layers, B, hidden_dim]
        h0 = h
        z = h[-1]                        # [B, hidden_dim]

        # decode
        recon = self.decode(x, h0)

        # one-class score: reconstruction error
        s = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))
        return recon, z, s
