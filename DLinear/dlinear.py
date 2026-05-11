"""
DLinear model (channel-independent, trend + seasonality decomposition).

Hyperparameters (passed to DLinear.__init__):
    seq_len      int  — input window length (lookback).
    pred_len     int  — forecast horizon length.
    n_channels   int  — number of independent channels (e.g. IV cells).
    kernel_size  int  — moving-average kernel for trend/seasonality
                        decomposition. Must be odd and <= seq_len.
                        Default: 13.
"""

import torch
import torch.nn as nn


class _MovingAvg(nn.Module):
    """Boundary-padded 1-D moving average preserving sequence length."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, C]
        x = torch.cat([
            x[:, :1].expand(-1, self.pad, -1),
            x,
            x[:, -1:].expand(-1, self.pad, -1),
        ], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinear(nn.Module):
    """
    Channel-independent DLinear.

    Each IV feature gets its own pair of linear maps (seasonal and trend).
    Vectorised batched matmul over channels — equivalent to n_channels
    independent nn.Linear layers but ~100× faster than a ModuleList loop.

    Input:  [B, seq_len, C]
    Output: [B, pred_len, C]
    """
    def __init__(self, seq_len: int, pred_len: int, n_channels: int, kernel_size: int = 13):
        super().__init__()
        self.decomp = _MovingAvg(kernel_size)
        w0 = (1.0 / seq_len) * torch.ones(n_channels, pred_len, seq_len)
        self.W_s = nn.Parameter(w0.clone())
        self.W_t = nn.Parameter(w0.clone())
        self.b_s = nn.Parameter(torch.zeros(n_channels, pred_len))
        self.b_t = nn.Parameter(torch.zeros(n_channels, pred_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, S, C]
        trend = self.decomp(x)
        seas  = x - trend
        s = seas.permute(0, 2, 1)
        t = trend.permute(0, 2, 1)
        out = (torch.einsum('bcs,cps->bcp', s, self.W_s) + self.b_s +
               torch.einsum('bcs,cps->bcp', t, self.W_t) + self.b_t)
        return out.permute(0, 2, 1)
