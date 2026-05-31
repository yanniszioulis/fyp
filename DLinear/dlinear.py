"""
dlinear.py
===================================================================================
DLinear — the LINEAR floor for the SANTA family.

The structural change vs PerCellTransformer is one move: the per-cell temporal
*attention* backbone is replaced by a per-cell *linear* map (DLinear, Zeng et al.
2023). No attention, no embeddings, no non-linearity in the trunk — just a
trend/seasonal series decomposition and one linear layer per series. Everything
on the I/O contract below stays byte-for-byte identical to the rest of the family,
so the trainer-side adapter is the SANTA one.

Where it sits in the ablation ladder
------------------------------------
    | model              | backbone                 | cross-cell | non-linear |
    | SANTA              | factored M/T/L attention | yes        | yes        |
    | SANTA-Temporal     | per-cell L attention     | none       | yes        |
    | PerCellTransformer | per-cell L attention     | none       | yes        |
    | DLinear            | per-cell L LINEAR        | none       | NO         |   <-- this

DLinear is the simplest model that still respects the surface contract: it is the
linear analogue of PerCellTransformer (per-cell, no cross-cell mixing) with the
temporal attention stack collapsed to a single affine map of the lookback. It
answers "how much of the SANTA performance is just a good per-cell linear filter?"

Diff to PerCellTransformer (everything else stays the same)
-----------------------------------------------------------
  1) DELETE value_proj, emb_lag, emb_drop, the TemporalOnlyLayer stack and the
     final LayerNorm + MLP head. There is no d-dimensional latent at all.
  2) Backbone = classic channel-independent DLinear, applied to the centred
     history u (B, L, M, T) with the M·T cells as independent channels:
       - series decomposition: trend = moving-average(u over L), seasonal = u−trend
       - per-cell linear maps  Linear(L → Hh) for trend and seasonal, summed.
     Each (m, τ) cell carries its OWN pair of (Hh × L) weight matrices — no weight
     sharing across cells, no mixing between cells (channel-independent).
  3) The linear map's output IS netDelta directly (see below). No head, no
     level / scale re-injection.

What stays exactly the same
---------------------------
  - Instance-norm centring on today's slice (per cell). Because the window is
    centred on today (u[:, -1] == 0), a forecast of the future *centred* level is
    by construction the change from today — so the DLinear output is netDelta with
    no extra bookkeeping, and a zero map reproduces the random-walk null.
  - Window scale is NOT divided out: the centred history keeps its amplitude, so
    the linear maps see the regime cue directly (matches SANTA's step-0 note).
  - Target: predict netDelta (residual on today); reconstruction
    ẑ_{t+h} = z_today + netDelta_h, identical to SANTA.
  - Loss: `surface_loss` from santa.py (uniform MSE on cumulative standardised
    changes); same horizons, same γ.
  - L, n_horizons, dropout, optimiser, early-stop, seeds — all driven from the
    shared Config (DLinear ignores the attention-only fields d / n_heads /
    n_layers / d_ff_mult / d_head_hidden / k_grid / tau_grid_years).

Parameter note
--------------
Channel-independent DLinear is naturally large: each of the M·T cells owns two
(Hh × L) matrices, so the count is ~ 2 · M·T · Hh · L (≈ 293k at M·T=110, L=63,
Hh=21). This is intrinsic to the model class (and matches the original DLinear
baseline in this project); it is NOT budget-matched to the ~50k SANTA family —
it is a different, deliberately minimal hypothesis, not a same-budget competitor.
"""
from __future__ import annotations
import os
import sys

# Make the repo root importable (surface_core) for standalone runs. train.py
# puts the root on sys.path already, so this only matters for
# `python DLinear/dlinear.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

# Shared Config + instance norm so callers write DLinear(Config(...)) like the
# rest of the family. DLinear ignores the attention-only Config fields (d,
# n_heads, n_layers, d_ff_mult, d_head_hidden, k_grid, tau_grid_years).
from surface_core import Config, instance_norm


# ----------------------------------------------------------------------------------
# Series decomposition: boundary-padded moving average (trend), residual (seasonal)
# ----------------------------------------------------------------------------------
class _MovingAvg(nn.Module):
    """Length-preserving 1-D moving average over the time axis of (Bx, L, C).

    Ends are replicate-padded so the trend has the same length L as the input
    (identical to the original DLinear / project baseline).
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (Bx, L, C)
        x = torch.cat([
            x[:, :1].expand(-1, self.pad, -1),
            x,
            x[:, -1:].expand(-1, self.pad, -1),
        ], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


# ----------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------
class DLinear(nn.Module):
    """Channel-independent DLinear under the SANTA I/O contract.

    Input  : (B, L, M, T) standardised log-IV window — the same contract as the
             rest of the SANTA family, so the trainer-side adapter is _SANTAAdapter.
    Output : (B, n_horizons, M, T) netDelta — predicted residual change from today
             to each horizon, in standardised log-IV units.

    Internal sequence:
      1) Instance-norm centring on today's slice (per cell), identical to SANTA;
      2) Flatten the centred surface to (B, L, M·T) — the M·T cells become the
         independent channels;
      3) Series decomposition: trend = moving-average over L, seasonal = u − trend;
      4) Per-cell linear maps Linear(L → Hh) for trend and seasonal, summed, giving
         the per-cell forecast of the future centred level = netDelta directly;
      5) Reshape to the canonical (B, Hh, M, T).
    """

    def __init__(self, cfg: Config, kernel_size: int = 25):
        super().__init__()
        self.cfg = cfg
        n_cells = cfg.M * cfg.T
        self._n_cells = n_cells
        Hh = cfg.n_horizons
        L = cfg.L

        # Kernel must be odd and no larger than the window (replicate-pad needs
        # pad < L for the expand to be well defined).
        k = min(kernel_size, L if L % 2 == 1 else L - 1)
        if k % 2 == 0:
            k -= 1
        self.kernel_size = max(k, 1)
        self.decomp = _MovingAvg(self.kernel_size)

        # Channel-independent weights: each cell c owns its own (Hh × L) trend and
        # seasonal matrices, plus a per-horizon bias. Initialised to the DLinear
        # prior 1/L (output = window mean of each component), matching the original
        # project baseline; on the centred history this is a mild reversion toward
        # the window mean, with the random-walk null (netDelta = 0) one step away.
        w0 = (1.0 / L) * torch.ones(n_cells, Hh, L)
        self.W_s = nn.Parameter(w0.clone())          # seasonal map (C, Hh, L)
        self.W_t = nn.Parameter(w0.clone())          # trend map    (C, Hh, L)
        self.b_s = nn.Parameter(torch.zeros(n_cells, Hh))
        self.b_t = nn.Parameter(torch.zeros(n_cells, Hh))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, L, M, T = z.shape
        assert (L, M, T) == (cfg.L, cfg.M, cfg.T), "window shape mismatch"

        # Step 1 — centre each cell's window on today. DLinear needs neither the
        # today-level nor the window scale separately: the centring folds the level
        # in and the amplitude is kept inside u (predicting the future centred level
        # IS predicting the change from today).
        u, _, _ = instance_norm(z)                    # (B, L, M, T)

        # Step 2 — flatten the surface to M·T independent channels. Order is
        # (M, T) flattened row-major; the inverse reshape at the end restores it,
        # so the cell↔channel mapping is internal and order-agnostic.
        u_flat = u.reshape(B, L, M * T)               # (B, L, C)

        # Step 3 — series decomposition (trend + seasonal residual).
        trend = self.decomp(u_flat)                   # (B, L, C)
        seas  = u_flat - trend                        # (B, L, C)

        # Step 4 — per-cell linear maps L → Hh, summed. Channel-independent: the
        # einsum applies cell c's own matrix to cell c's own series only.
        s = seas.permute(0, 2, 1)                     # (B, C, L)
        t = trend.permute(0, 2, 1)                    # (B, C, L)
        out = (torch.einsum("bcl,chl->bch", s, self.W_s) + self.b_s
               + torch.einsum("bcl,chl->bch", t, self.W_t) + self.b_t)  # (B, C, Hh)

        # Step 5 — restore the canonical (B, Hh, M, T). out is (B, C, Hh).
        netDelta = out.permute(0, 2, 1).reshape(B, cfg.n_horizons, M, T).contiguous()
        return netDelta

    # No attention maps to collect — provided for API parity with the family so
    # eval scripts can call it unconditionally.
    def enable_attn_collection(self, on: bool = True):
        pass


# ----------------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    from surface_core import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    # Match the trainer's at-runtime config: M=11, T=10 (current dataset).
    cfg = Config(M=11, T=10, L=63, horizons=tuple(range(1, 22)),
                 d=16, n_heads=4, n_layers=2, d_ff_mult=1,
                 d_head_hidden=24, dropout=0.1)
    model = DLinear(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"DLinear parameters: {n:,}   (kernel_size={model.kernel_size})")

    # Fake standardised series (300 days × 11×10 surface).
    S = 300
    Z = torch.randn(S, cfg.M, cfg.T).cumsum(0) * 0.02
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)

    windows, z_today, z_future, _ = build_windows(Z, cfg.horizons, cfg.L)
    print("windows :", tuple(windows.shape))
    print("z_today :", tuple(z_today.shape))
    print("z_future:", tuple(z_future.shape))

    bs = 64
    xb, tb, fb = windows[:bs], z_today[:bs], z_future[:bs]
    netDelta = model(xb)
    print("netDelta:", tuple(netDelta.shape))

    loss = surface_loss(netDelta, tb, fb)
    base = rw_loss(tb, fb)
    print(f"model loss = {loss.item():.4f}   RW loss = {base.item():.4f}")

    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters()
                  if p.requires_grad)
    print("backward OK, all params have grads:", grad_ok)
