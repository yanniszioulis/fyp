"""
DLinear model (channel-independent, trend + seasonality decomposition).

Hyperparameters (passed to DLinear.__init__):
    seq_len      int  — input window length (lookback).
    pred_len     int  — forecast horizon length.
    n_channels   int  — number of independent channels (e.g. IV cells).
    kernel_size  int  — moving-average kernel for trend/seasonality
                        decomposition. Must be odd and <= seq_len.
                        Default: 13.
    revin        bool — per-cell RevIN norm/denorm around the model:
                        each channel's lookback mean/std are stripped
                        before the linear maps and reapplied after, in
                        the style of Kim et al. 2022 and matching the
                        per-cell RevIN used by PatchTST / HOT /
                        iTransformer in this project. Default: False
                        (the original Zeng et al. 2022 DLinear does no
                        normalisation).
    revin_affine bool — learnable per-channel scale/bias inside RevIN.
                        Default: False.
    revin_eps    float — numerical stabilizer in std computation.
                        Default: 1e-5.
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


class RevIN(nn.Module):
    """
    Per-cell RevIN (Kim et al. 2022): for input [B, T, C], reduce over
    T only and keep per-channel mean/std. Each cell's own lookback
    level/scale is stripped before the model and reapplied after.

    Affine parameters, if enabled, are per-channel (one pair per
    feature, broadcasting across batch and time).
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(mode)

    def _get_statistics(self, x):
        dims = tuple(range(1, x.ndim - 1))
        self.mean  = x.mean(dim=dims, keepdim=True).detach()
        self.stdev = torch.sqrt(
            x.var(dim=dims, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        return x * self.stdev + self.mean


class DLinear(nn.Module):
    """
    Channel-independent DLinear.

    Each IV feature gets its own pair of linear maps (seasonal and trend).
    Vectorised batched matmul over channels — equivalent to n_channels
    independent nn.Linear layers but ~100× faster than a ModuleList loop.

    Input:  [B, seq_len, C]
    Output: [B, pred_len, C]
    """
    def __init__(self, seq_len: int, pred_len: int, n_channels: int,
                 kernel_size: int = 21,
                 revin: bool = False, revin_affine: bool = False,
                 revin_eps: float = 1e-5):
        super().__init__()
        self.decomp = _MovingAvg(kernel_size)
        w0 = (1.0 / seq_len) * torch.ones(n_channels, pred_len, seq_len)
        self.W_s = nn.Parameter(w0.clone())
        self.W_t = nn.Parameter(w0.clone())
        self.b_s = nn.Parameter(torch.zeros(n_channels, pred_len))
        self.b_t = nn.Parameter(torch.zeros(n_channels, pred_len))
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(
                num_features=n_channels, eps=revin_eps, affine=revin_affine,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, S, C]
        if self.revin:
            x = self.revin_layer(x, "norm")
        trend = self.decomp(x)
        seas  = x - trend
        s = seas.permute(0, 2, 1)
        t = trend.permute(0, 2, 1)
        out = (torch.einsum('bcs,cps->bcp', s, self.W_s) + self.b_s +
               torch.einsum('bcs,cps->bcp', t, self.W_t) + self.b_t)
        out = out.permute(0, 2, 1)
        if self.revin:
            out = self.revin_layer(out, "denorm")
        return out
