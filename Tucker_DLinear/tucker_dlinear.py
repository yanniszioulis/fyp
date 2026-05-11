"""
TuckerDLinear — DLinear over W x H surfaces with Tucker-decomposed weights.

Treats the input as a lookback window of W x H surfaces and decomposes it
into trend (per-cell moving average along time) and seasonal (residual)
components, mirroring standard DLinear. Each component is mapped to the
prediction horizon by a linear map whose implicit weight tensor — full
shape [L, W, H, P, W, H] — is represented as a Tucker decomposition with
a six-mode core G and six factor matrices A_L, A_Wi, A_Hi, A_P, A_Wo,
A_Ho. Input/output spatial factors are independently parameterised but
share their rank (rank_W applies to A_Wi and A_Wo; rank_H to A_Hi and
A_Ho).

At full spatial ranks (rank_W = W, rank_H = H), this Tucker class strictly
contains channel-independent DLinear: setting A_Wi = A_Wo = I_W and
A_Hi = A_Ho = I_H reduces the model to a per-cell rank-(rank_L, rank_P)
temporal map. The init exploits this — when rank_W = W and rank_H = H the
spatial factors are initialised as identity and the core's spatial
diagonal is set to 1/seq_len, exactly recovering DLinear's per-cell
uniform-average init. When spatial ranks are reduced, the model falls
back to the global-broadcast init (T = 1/seq_len uniformly).

Hyperparameters (passed to TuckerDLinear.__init__):
    seq_len      int — input window length (lookback).
    pred_len     int — forecast horizon length.
    W            int — width of the surface grid (e.g. moneyness axis).
    H            int — height of the surface grid (e.g. tau axis).
    rank_L       int — Tucker rank along the lookback axis. In [1, seq_len].
    rank_P       int — Tucker rank along the forecast horizon. In [1, pred_len].
    rank_W       int — shared Tucker rank for A_Wi and A_Wo. In [1, W].
    rank_H       int — shared Tucker rank for A_Hi and A_Ho. In [1, H].
    kernel_size  int — moving-avg kernel for trend/seasonality decomposition.
                       Must be odd. Default: 13.
    norm         bool — joint per-window normalisation across (L, W, H):
                        strip the surface-wide mean and std before
                        decomposition and the linear maps, add back at
                        the output. Preserves cross-cell structure within
                        a window while removing the overall vol
                        level/scale. Default: True.

Input:  [B, L, W, H]
Output: [B, P, W, H]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SurfaceMovingAvg(nn.Module):
    """Pointwise moving average along the lookback axis (per spatial cell)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, W, H = x.shape
        if self.pad > 0:
            left  = x[:, :1].expand(-1, self.pad, -1, -1)
            right = x[:, -1:].expand(-1, self.pad, -1, -1)
            x = torch.cat([left, x, right], dim=1)
        x = x.reshape(B, L + 2 * self.pad, W * H).permute(0, 2, 1)
        x = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)
        return x.permute(0, 2, 1).reshape(B, L, W, H)


class _TuckerLinear(nn.Module):
    """
    Tucker-decomposed linear map [B, L, W, H] -> [B, P, W, H].

    Implicit weight tensor (never materialised):
        T[l, wi, hi, p, wo, ho] = sum_{a,b,c,d,e,f}
              G[a, b, c, d, e, f]
              * A_L[l, a]   * A_Wi[wi, b] * A_Hi[hi, c]
              * A_P[p, d]   * A_Wo[wo, e] * A_Ho[ho, f]

    Ranks
        rank_L  in [1, seq_len]   lookback temporal rank
        rank_P  in [1, pred_len]  forecast temporal rank
        rank_W  in [1, W]         shared rank for A_Wi and A_Wo
        rank_H  in [1, H]         shared rank for A_Hi and A_Ho

    At rank_W = W and rank_H = H, channel-independent DLinear is contained
    in this class: the spatial factors initialise to identity and the
    core's spatial diagonal G[0, w, h, 0, w, h] = 1/seq_len reproduces
    DLinear's per-cell uniform-average map. When rank_W < W or rank_H < H,
    information is forced to mix across spatial cells through the shared
    rank dimension, and the init falls back to global broadcast.

    Forward computes the contractions in three steps:
        z[n, a, b, c]   = sum_{l, wi, hi}
                          x[n, l, wi, hi]
                          * A_L[l, a] * A_Wi[wi, b] * A_Hi[hi, c]
        mid[n, d, e, f] = sum_{a, b, c} z[n, a, b, c] * G[a, b, c, d, e, f]
        out[n, p, w, h] = sum_{d, e, f}
                          mid[n, d, e, f]
                          * A_P[p, d] * A_Wo[w, e] * A_Ho[h, f]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        W: int,
        H: int,
        rank_L: int,
        rank_P: int,
        rank_W: int,
        rank_H: int,
    ):
        super().__init__()
        if not (1 <= rank_L <= seq_len):
            raise ValueError(f"rank_L must be in [1, seq_len={seq_len}], got {rank_L}")
        if not (1 <= rank_P <= pred_len):
            raise ValueError(f"rank_P must be in [1, pred_len={pred_len}], got {rank_P}")
        if not (1 <= rank_W <= W):
            raise ValueError(f"rank_W must be in [1, W={W}], got {rank_W}")
        if not (1 <= rank_H <= H):
            raise ValueError(f"rank_H must be in [1, H={H}], got {rank_H}")

        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.W        = W
        self.H        = H
        self.rank_L   = rank_L
        self.rank_P   = rank_P
        self.rank_W   = rank_W
        self.rank_H   = rank_H

        self.A_L  = nn.Parameter(torch.empty(seq_len,  rank_L))
        self.A_Wi = nn.Parameter(torch.empty(W,        rank_W))
        self.A_Hi = nn.Parameter(torch.empty(H,        rank_H))
        self.A_P  = nn.Parameter(torch.empty(pred_len, rank_P))
        self.A_Wo = nn.Parameter(torch.empty(W,        rank_W))
        self.A_Ho = nn.Parameter(torch.empty(H,        rank_H))
        self.G    = nn.Parameter(
            torch.empty(rank_L, rank_W, rank_H, rank_P, rank_W, rank_H)
        )

        with torch.no_grad():
            # Temporal factors: column 0 = 1 (uniform). Higher-rank columns
            # carry small noise to break symmetries during optimisation.
            self.A_L.normal_(0.0, 1e-2); self.A_L[:, 0] = 1.0
            self.A_P.normal_(0.0, 1e-2); self.A_P[:, 0] = 1.0
            # Spatial factors: identity at full rank (channel-independent capable);
            # otherwise column 0 = 1 (global broadcast) plus small noise.
            full_W = (rank_W == W)
            full_H = (rank_H == H)
            if full_W:
                self.A_Wi.copy_(torch.eye(W))
                self.A_Wo.copy_(torch.eye(W))
            else:
                self.A_Wi.normal_(0.0, 1e-2); self.A_Wi[:, 0] = 1.0
                self.A_Wo.normal_(0.0, 1e-2); self.A_Wo[:, 0] = 1.0
            if full_H:
                self.A_Hi.copy_(torch.eye(H))
                self.A_Ho.copy_(torch.eye(H))
            else:
                self.A_Hi.normal_(0.0, 1e-2); self.A_Hi[:, 0] = 1.0
                self.A_Ho.normal_(0.0, 1e-2); self.A_Ho[:, 0] = 1.0
            # Core: zero except for deliberate seed entries. Cross-cell G entries
            # are O(R_W*R_H) in number and even small noise on each (~1e-2) sums
            # to O(0.3) noise on every output cell, swamping the 1/seq_len signal.
            # Symmetry-breaking instead flows from the temporal factors' higher-
            # rank columns, which only enter the output through the (currently
            # zero) higher-temporal-rank slices of G — gradients pull those slices
            # off zero during training.
            self.G.zero_()
            if full_W and full_H:
                # Channel-independent uniform-average map: each output cell
                # averages its own input cell across the lookback. With A_Wi,
                # A_Wo, A_Hi, A_Ho = identity and A_L, A_P first columns = 1,
                # G[0, w, h, 0, w, h] = 1/seq_len makes T = (1/seq_len)*delta
                # along (wi=wo, hi=ho) — exactly DLinear's init.
                inv = 1.0 / seq_len
                for w in range(W):
                    for h in range(H):
                        self.G[0, w, h, 0, w, h] = inv
            else:
                # Global broadcast init: T = 1/seq_len uniformly across all
                # input/output indices. Same baseline as the previous CP version.
                self.G[0, 0, 0, 0, 0, 0] = 1.0 / seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.einsum(
            'nlwh,la,wb,hc->nabc',
            x, self.A_L, self.A_Wi, self.A_Hi,
        )
        mid = torch.einsum('nabc,abcdef->ndef', z, self.G)
        return torch.einsum(
            'ndef,pd,we,hf->npwh',
            mid, self.A_P, self.A_Wo, self.A_Ho,
        )


class TuckerDLinear(nn.Module):
    """
    DLinear over W x H surfaces with Tucker-decomposed weights.

    Input:  [B, seq_len,  W, H]
    Output: [B, pred_len, W, H]
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        W: int,
        H: int,
        rank_L: int,
        rank_P: int,
        rank_W: int,
        rank_H: int,
        kernel_size: int = 31,
        norm: bool = False,
    ):
        super().__init__()
        self.seq_len     = seq_len
        self.pred_len    = pred_len
        self.W           = W
        self.H           = H
        self.rank_L      = rank_L
        self.rank_P      = rank_P
        self.rank_W      = rank_W
        self.rank_H      = rank_H
        self.kernel_size = kernel_size
        self.norm        = norm

        self.decomp       = _SurfaceMovingAvg(kernel_size)
        self.trend_map    = _TuckerLinear(seq_len, pred_len, W, H,
                                          rank_L, rank_P, rank_W, rank_H)
        self.seasonal_map = _TuckerLinear(seq_len, pred_len, W, H,
                                          rank_L, rank_P, rank_W, rank_H)
        self.bias         = nn.Parameter(torch.zeros(pred_len, W, H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm:
            mu  = x.mean(dim=(1, 2, 3), keepdim=True)
            std = torch.sqrt(x.var(dim=(1, 2, 3), keepdim=True, unbiased=False) + 1e-5)
            x_in = (x - mu) / std
        else:
            x_in = x

        trend    = self.decomp(x_in)
        seasonal = x_in - trend
        out      = self.trend_map(trend) + self.seasonal_map(seasonal) + self.bias

        if self.norm:
            out = (out * std) + mu
        return out
