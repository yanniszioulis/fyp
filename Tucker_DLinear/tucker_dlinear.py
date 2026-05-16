"""
TuckerDLinear — two-branch (trend + seasonal) DLinear over W x H
surfaces with Tucker-decomposed weights.

Decomposition
-------------
A single moving average splits the input into two bands:

    trend    = MA(x, kernel_trend)
    seasonal = x - trend

This mirrors standard DLinear's trend/seasonal split, applied per (W, H)
cell along the lookback. The decomposition is parameter-free.

Each band is mapped to the forecast horizon by its own Tucker linear
map with independent ranks. The output is the sum plus a static
spatial bias:

    out = trend_map(trend) + seasonal_map(seasonal) + bias

Tucker linear map
-----------------
Each _TuckerLinear represents an implicit weight tensor
T[l, wi, hi, p, wo, ho] of shape [L, W, H, P, W, H] via a six-mode core
G and six factor matrices A_L, A_Wi, A_Hi, A_P, A_Wo, A_Ho.

The ranks decouple temporal and spatial capacity:

    rank_L  in [1, seq_len]   how many temporal modes from the lookback
    rank_P  in [1, pred_len]  how many horizon shapes the output can take
    rank_W  in [1, W]         spatial rank along the moneyness axis
    rank_H  in [1, H]         spatial rank along the tenor axis

When rank_W = W and rank_H = H, the spatial factors initialise to
identity, so the map preserves the input surface exactly through the
spatial pathway at init and all spatial mixing must be learned in G.
When rank_W < W or rank_H < H, the spatial factors form a random
orthonormal basis for a rank-r subspace, so the spatial pathway
projects through that subspace by construction — a low-rank inductive
bias on the spatial structure.

Initialisation
--------------
The init has a single goal: at step 0, the model predicts the lookback
mean broadcast across the horizon, with all higher-rank channels seeded
to receive nonzero gradient. Concretely:

  A_L[:, 0]              = 1/sqrt(L) · 1  (uniform; "rank-0 mean path")
  A_L[:, 1:]             orthonormal noise (Gram-Schmidt'd)
  A_P[:, 0]              = 1/sqrt(P) · 1  (uniform)
  A_P[:, 1:]             orthonormal noise
  A_Wi, A_Wo             orthonormal (identity if rank_W = W)
  A_Hi, A_Ho             orthonormal (identity if rank_H = H)
  G[0, b, c, 0, b, c]    = sqrt(P/L) · 1  (rank-0 diagonal warm start)
  G rest                 ~ N(0, 1e-3) noise

When rank_W = W and rank_H = H (spatial factors are identity), the
rank-0 forward at init evaluates exactly to the per-cell lookback
mean broadcast across the horizon:
    out[p, w, h] = (1/L) · sum_l x[l, w, h]
which is exactly DLinear's trend init. When rank_W < W or rank_H < H,
the spatial pathway projects the input through a random orthonormal
rank-r subspace, so the rank-0 forward becomes
    out[p, w, h] = (P_W ⊗ P_H)(mean_l(x[l, :, :]))[w, h]
where P_W, P_H are the rank-r orthogonal projectors. For bands whose
mean is approximately zero (e.g. the seasonal residual), this
projected init is also approximately zero — fine in practice — but
the equality with the per-cell mean is no longer exact.

The dense noise on the rest of G keeps gradient flowing into
A_L[:, 1:] / A_P[:, 1:] / off-diagonal G entries from step 1 —
otherwise those channels would be gradient-stranded since they'd
start at exactly zero output.

Persistence init was tried earlier and removed: starting the model at
"predict yesterday's surface" placed it in a basin where it had to
climb out to discover the mean-reverting structure that DLinear-style
training finds naturally. Mean init is the cleaner first-principles
choice.

Output bias
-----------
A static spatial bias [W, H] is added at the end. This lets the model
learn the typical surface shape without leaking it into horizon-
dependent capacity. Any horizon-dependent structure must come from the
conditional path (A_P · G), not from a free [P, W, H] offset.

Input:  [B, seq_len,  W, H]
Output: [B, pred_len, W, H]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SurfaceMovingAvg(nn.Module):
    """Pointwise moving average along the lookback axis (per spatial cell).

    Replicate-pads both ends so the output has the same length as the
    input. The kernel must be a positive odd integer so the padding is
    symmetric.
    """

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
        # Treat (W, H) cells as channels for the 1D pool.
        x = x.reshape(B, L + 2 * self.pad, W * H).permute(0, 2, 1)
        x = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)
        return x.permute(0, 2, 1).reshape(B, L, W, H)


def _orth_basis(D: int, rank: int) -> torch.Tensor:
    """Return a D × rank orthonormal matrix.

    When rank == D, returns the identity I_D (gives the spatial pathway
    a clean identity init at full rank). When rank < D, returns the Q
    factor of QR(random) — a uniformly random orthonormal basis for a
    rank-r subspace of R^D. The implicit projector A @ A.T is then a
    rank-r orthogonal projector, the best rank-r Frobenius approximation
    of I_D.
    """
    if rank == D:
        return torch.eye(D)
    a = torch.randn(D, rank)
    q, _ = torch.linalg.qr(a)
    return q


def _uniform_then_orth(D: int, rank: int) -> torch.Tensor:
    """Return D × rank with col 0 fixed to the unit-norm uniform vector
    and cols 1+ forming an orthonormal basis for the orthogonal
    complement (via Gram-Schmidt on Gaussian noise).

    The fixed col-0 carries the rank-0 mean path; cols 1+ start as a
    random orthonormal direction set that's free to specialise during
    training, with gradient seeded by the dense G noise.
    """
    A = torch.randn(D, rank)
    A[:, 0] = torch.ones(D) / (D ** 0.5)
    for k in range(1, rank):
        v = A[:, k].clone()
        for j in range(k):
            v = v - (A[:, j] @ v) * A[:, j]
        A[:, k] = v / (v.norm() + 1e-12)
    return A


class _TuckerLinear(nn.Module):
    """Tucker-decomposed linear map [B, L, W, H] -> [B, P, W, H].

    Implicit weight tensor (never materialised):
        T[l, wi, hi, p, wo, ho] = sum_{a, b, c, d, e, f}
              G[a, b, c, d, e, f]
              * A_L[l, a]  * A_Wi[wi, b] * A_Hi[hi, c]
              * A_P[p, d]  * A_Wo[wo, e] * A_Ho[ho, f]

    Forward computes three einsum contractions:
        z   = einsum('nlwh, la, wb, hc -> nabc', x, A_L, A_Wi, A_Hi)
        mid = einsum('nabc, abcdef -> ndef',     z, G)
        out = einsum('ndef, pd, we, hf -> npwh', mid, A_P, A_Wo, A_Ho)

    At init the forward evaluates to the lookback mean broadcast across
    the horizon (the standard DLinear trend init):
        out[p, w, h] ≈ (1/L) · sum_l x[l, w, h]
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
        g_init_noise: float = 1e-3,
    ):
        super().__init__()
        if not (1 <= rank_L <= seq_len):
            raise ValueError(f"rank_L must be in [1, {seq_len}], got {rank_L}")
        if not (1 <= rank_P <= pred_len):
            raise ValueError(f"rank_P must be in [1, {pred_len}], got {rank_P}")
        if not (1 <= rank_W <= W):
            raise ValueError(f"rank_W must be in [1, {W}], got {rank_W}")
        if not (1 <= rank_H <= H):
            raise ValueError(f"rank_H must be in [1, {H}], got {rank_H}")

        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.W = W
        self.H = H
        self.rank_L = rank_L
        self.rank_P = rank_P
        self.rank_W = rank_W
        self.rank_H = rank_H

        self.A_L  = nn.Parameter(torch.empty(seq_len,  rank_L))
        self.A_P  = nn.Parameter(torch.empty(pred_len, rank_P))
        self.A_Wi = nn.Parameter(torch.empty(W, rank_W))
        self.A_Hi = nn.Parameter(torch.empty(H, rank_H))
        self.A_Wo = nn.Parameter(torch.empty(W, rank_W))
        self.A_Ho = nn.Parameter(torch.empty(H, rank_H))
        self.G    = nn.Parameter(
            torch.empty(rank_L, rank_W, rank_H, rank_P, rank_W, rank_H)
        )

        with torch.no_grad():
            # Temporal factors: col 0 = unit-uniform (mean path),
            # cols 1+ = orthonormal noise. The uniform col 0 means the
            # rank-0 forward evaluates to a sum over l of (1/sqrt(L)) ·
            # x[l, ...], which becomes the per-cell mean after the G
            # diagonal multiplies by sqrt(P/L) (see below).
            self.A_L.copy_(_uniform_then_orth(seq_len,  rank_L))
            self.A_P.copy_(_uniform_then_orth(pred_len, rank_P))

            # Spatial factors: orthonormal columns. At full rank these
            # are identity; at partial rank they're a random orthonormal
            # basis. A_Wi = A_Wo (and A_Hi = A_Ho) at init so the
            # spatial pathway is a symmetric projector at step 0;
            # training is free to drift them apart.
            Q_W = _orth_basis(W, rank_W)
            Q_H = _orth_basis(H, rank_H)
            self.A_Wi.copy_(Q_W); self.A_Wo.copy_(Q_W)
            self.A_Hi.copy_(Q_H); self.A_Ho.copy_(Q_H)

            # G: small dense noise + a single diagonal warm start on
            # the rank-0 path. The (0, b, c, 0, b, c) diagonal entries
            # are set to sqrt(P/L) so the rank-0 forward at init reads
            #     out[p, w, h] = (1/sqrt(L) sum_l x[l, w, h]) · sqrt(P/L) · (1/sqrt(P))
            #                  = (1/L) sum_l x[l, w, h]
            # which is exactly the per-cell mean of the lookback,
            # broadcast across all horizons — the standard DLinear
            # trend init.
            #
            # The dense N(0, g_init_noise) noise on the remaining G
            # entries gives all higher-rank channels of A_L, A_P, A_Wi,
            # A_Hi, A_Wo, A_Ho nonzero gradient from step 1. Without it,
            # cols 1+ of those factors would be gradient-stranded
            # (because G's higher-rank slices would be exactly zero,
            # producing zero contribution and zero gradient).
            self.G.normal_(0.0, g_init_noise)
            g_diag = (pred_len / seq_len) ** 0.5
            for b in range(rank_W):
                for c in range(rank_H):
                    self.G[0, b, c, 0, b, c] += g_diag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.einsum(
            'nlwh, la, wb, hc -> nabc',
            x, self.A_L, self.A_Wi, self.A_Hi,
        )
        mid = torch.einsum('nabc, abcdef -> ndef', z, self.G)
        return torch.einsum(
            'ndef, pd, we, hf -> npwh',
            mid, self.A_P, self.A_Wo, self.A_Ho,
        )


class TuckerDLinear(nn.Module):
    """Two-branch (trend + seasonal) DLinear over W x H surfaces with
    Tucker-decomposed weights.

    The trend branch sees the heavily-smoothed signal and should
    preserve the full surface — defaults to full spatial rank so that
    no signal is destroyed by the spatial pathway.

    The seasonal branch sees the high-frequency residual and benefits
    from low-rank spatial structure (the residual lives mostly in the
    dominant shape modes — level/slope/skew/butterfly). Its defaults
    are moderate spatial rank.

    Both branches have small temporal rank by default; this is the
    standard DLinear-style "single temporal mode plus a few free
    directions" setup.

    Args:
        seq_len, pred_len  : lookback and forecast horizon lengths.
        W, H               : surface width (moneyness) and height (tenor).
        rank_L_trend       : trend branch temporal rank along lookback.
        rank_P_trend       : trend branch temporal rank along horizon.
        rank_W_trend       : trend branch spatial rank along W. Default W
                             (full rank → identity factor at init → no
                             signal loss through the spatial pathway).
        rank_H_trend       : trend branch spatial rank along H. Default H.
        rank_L_seasonal    : seasonal branch temporal rank along lookback.
        rank_P_seasonal    : seasonal branch temporal rank along horizon.
        rank_W_seasonal    : seasonal branch spatial rank along W.
                             Default 6 (low-rank spatial bias).
        rank_H_seasonal    : seasonal branch spatial rank along H.
                             Default 4.
        kernel_trend       : MA kernel size for the trend/seasonal
                             split. Default 41.
        g_init_noise       : std of the dense noise on the non-diagonal
                             entries of G; controls how strong the
                             gradient signal is for the higher-rank
                             factor channels. Default 1e-3.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        W: int,
        H: int,
        rank_L_trend: int = 2,
        rank_P_trend: int = 2,
        rank_W_trend: int = None,    # default: W (full spatial rank)
        rank_H_trend: int = None,    # default: H (full spatial rank)
        rank_L_seasonal: int = 4,
        rank_P_seasonal: int = 2,
        rank_W_seasonal: int = 6,
        rank_H_seasonal: int = 4,
        kernel_trend: int = 41,
        g_init_noise: float = 1e-3,
    ):
        super().__init__()
        # Default trend spatial ranks to full.
        if rank_W_trend is None:
            rank_W_trend = W
        if rank_H_trend is None:
            rank_H_trend = H

        # Clip seasonal ranks to be within valid range (so callers don't
        # have to know W and H upfront when passing defaults).
        rank_W_seasonal = min(rank_W_seasonal, W)
        rank_H_seasonal = min(rank_H_seasonal, H)

        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.W = W
        self.H = H
        self.kernel_trend = kernel_trend

        self.decomp = _SurfaceMovingAvg(kernel_trend)

        self.trend_map = _TuckerLinear(
            seq_len, pred_len, W, H,
            rank_L=rank_L_trend,    rank_P=rank_P_trend,
            rank_W=rank_W_trend,    rank_H=rank_H_trend,
            g_init_noise=g_init_noise,
        )
        self.seasonal_map = _TuckerLinear(
            seq_len, pred_len, W, H,
            rank_L=rank_L_seasonal, rank_P=rank_P_seasonal,
            rank_W=rank_W_seasonal, rank_H=rank_H_seasonal,
            g_init_noise=g_init_noise,
        )

        # Static spatial bias. Lets the model learn a typical surface
        # shape without leaking that into the horizon-dependent path.
        self.bias = nn.Parameter(torch.zeros(W, H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend    = self.decomp(x)
        seasonal = x - trend
        return (
            self.trend_map(trend)
            + self.seasonal_map(seasonal)
            + self.bias
        )