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
uniform-average init.

When spatial ranks are reduced (rank_W < W or rank_H < H), the spatial
factors are initialised to orthonormal D × rank columns and the core
keeps the same per-(rank-W, rank-H) diagonal pattern G[0, b, c, 0, b, c]
= 1/seq_len. The implicit spatial map A @ A.T becomes a rank-r orthogonal
projector — the Frobenius-optimal rank-r approximation of I_D — so the
forward at init is "project the input surface onto the rank-r spatial
subspace, average over lookback". Output magnitude stays bounded by
||x||_F regardless of rank.

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


def _orth(D: int, rank: int) -> torch.Tensor:
    """Orthonormal D × rank init.

    Returns I_D when rank == D (full-rank case; preserves the DLinear-equivalent
    init path bitwise). When rank < D, returns the Q-factor of QR(random) — a
    random orthonormal basis for a rank-r subspace of R^D. The implicit spatial
    map A @ A.T is then a rank-r orthogonal projector, the best rank-r Frobenius
    approximation of I_D.
    """
    if rank == D:
        return torch.eye(D)
    a = torch.randn(D, rank)
    q, _ = torch.linalg.qr(a)
    return q


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
        tie_spatial: bool = True,
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

        self.seq_len     = seq_len
        self.pred_len    = pred_len
        self.W           = W
        self.H           = H
        self.rank_L      = rank_L
        self.rank_P      = rank_P
        self.rank_W      = rank_W
        self.rank_H      = rank_H
        self.tie_spatial = tie_spatial

        self.A_L  = nn.Parameter(torch.empty(seq_len,  rank_L))
        self.A_P  = nn.Parameter(torch.empty(pred_len, rank_P))
        if tie_spatial:
            # Single shared spatial factor used as both input and output. Kills
            # the asymmetric (A_Wi, A_Wo) and (A_Hi, A_Ho) gauge freedom — the
            # dilation/skew directions in the spatial-factor gauge group — and
            # keeps only the orthogonal-rotation residue.
            self.A_W = nn.Parameter(torch.empty(W, rank_W))
            self.A_H = nn.Parameter(torch.empty(H, rank_H))
        else:
            self.A_Wi = nn.Parameter(torch.empty(W, rank_W))
            self.A_Hi = nn.Parameter(torch.empty(H, rank_H))
            self.A_Wo = nn.Parameter(torch.empty(W, rank_W))
            self.A_Ho = nn.Parameter(torch.empty(H, rank_H))
        self.G    = nn.Parameter(
            torch.empty(rank_L, rank_W, rank_H, rank_P, rank_W, rank_H)
        )

        with torch.no_grad():
            # Temporal factors: column 0 = 1 (uniform). Higher-rank columns
            # carry small noise so gradient can flow into the higher temporal-
            # rank slices of G (which are zero at init) during training.
            self.A_L.normal_(0.0, 1e-2); self.A_L[:, 0] = 1.0
            self.A_P.normal_(0.0, 1e-2); self.A_P[:, 0] = 1.0
            # Spatial factors: orthonormal D × rank columns. At full spatial
            # rank _orth returns identity, exactly preserving the previous
            # DLinear-equivalent path. At partial spatial rank A @ A.T is a
            # rank-r orthogonal projector — the Frobenius-optimal rank-r
            # approximation of I_D — keeping the init scale bounded
            # (||out||_F ≤ ||x||_F) regardless of rank.
            if tie_spatial:
                self.A_W.copy_(_orth(W, rank_W))
                self.A_H.copy_(_orth(H, rank_H))
            else:
                # Init A_Wi = A_Wo (and A_Hi = A_Ho) to the same orthonormal
                # matrix so the implicit spatial map is a projector at init
                # in both tied and untied modes. Training is free to drift
                # them apart in the untied case.
                Q_W = _orth(W, rank_W)
                Q_H = _orth(H, rank_H)
                self.A_Wi.copy_(Q_W); self.A_Wo.copy_(Q_W)
                self.A_Hi.copy_(Q_H); self.A_Ho.copy_(Q_H)
            # Core: G[0, b, c, 0, b, c] = 1/seq_len for (b, c) in
            # [rank_W] × [rank_H], else 0. At full spatial rank this is the
            # per-cell DLinear-equivalent init (output = lookback mean per
            # cell, recovered bitwise). At partial spatial rank, combined
            # with orthonormal A_W/A_H, the implicit map is
            # T = (1/L) · P_W ⊗ P_H along (input cell, output cell): the
            # output projects the input surface onto the rank-r spatial
            # subspace and averages over lookback. Bounded scale; no
            # collapse to a single global scalar like the previous fallback.
            self.G.zero_()
            inv = 1.0 / seq_len
            for b in range(rank_W):
                for c in range(rank_H):
                    self.G[0, b, c, 0, b, c] = inv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tie_spatial:
            A_Wi = A_Wo = self.A_W
            A_Hi = A_Ho = self.A_H
        else:
            A_Wi, A_Wo = self.A_Wi, self.A_Wo
            A_Hi, A_Ho = self.A_Hi, self.A_Ho
        z = torch.einsum(
            'nlwh,la,wb,hc->nabc',
            x, self.A_L, A_Wi, A_Hi,
        )
        mid = torch.einsum('nabc,abcdef->ndef', z, self.G)
        return torch.einsum(
            'ndef,pd,we,hf->npwh',
            mid, self.A_P, A_Wo, A_Ho,
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
        tie_spatial: bool = True,
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
        self.tie_spatial = tie_spatial

        self.decomp       = _SurfaceMovingAvg(kernel_size)
        self.trend_map    = _TuckerLinear(seq_len, pred_len, W, H,
                                          rank_L, rank_P, rank_W, rank_H,
                                          tie_spatial=tie_spatial)
        self.seasonal_map = _TuckerLinear(seq_len, pred_len, W, H,
                                          rank_L, rank_P, rank_W, rank_H,
                                          tie_spatial=tie_spatial)
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
