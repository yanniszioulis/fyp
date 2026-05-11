"""
DLinear model (channel-independent, trend + seasonality decomposition).

Hyperparameters (passed to DLinear.__init__):
    seq_len      int  — input window length (lookback).
    pred_len     int  — forecast horizon length.
    n_channels   int  — number of independent channels (e.g. IV cells).
    kernel_size  int  — moving-average kernel for trend/seasonality
                        decomposition. Must be odd and <= seq_len.
                        Default: 13.
    revin        bool — joint per-window RevIN norm/denorm around the
                        model: strips mean/std jointly over (L, C),
                        preserving cross-channel structure within the
                        window. Default: True. This is a deviation from
                        the original DLinear (Zeng et al. 2022), added
                        for parity with PatchTST / HOT / TuckerDLinear in
                        this project. Pass revin=False to reproduce the
                        original behaviour.
    revin_affine bool — learnable scalar scale/bias inside RevIN.
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


class JointRevIN(nn.Module):
    """
    Joint RevIN: per-window mean/std reduction over all non-batch
    axes. For input [B, L, C], reduces over (L, C) jointly and
    produces scalar mean/std per window. Strips overall window
    level/scale while preserving cross-channel structure.

    Affine parameters, if enabled, are scalar (one pair per
    window broadcasting across all positions).
    """

    def __init__(self, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1))
            self.affine_bias   = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(mode)

    def _get_statistics(self, x):
        dims = tuple(range(1, x.ndim))
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
            self.revin_layer = JointRevIN(eps=revin_eps, affine=revin_affine)

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
