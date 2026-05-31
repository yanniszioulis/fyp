"""
linear.py
===================================================================================
Linear — the simplest floor for the SANTA family: DLinear without decomposition.

Identical to DLinear except the trend/seasonal series decomposition is dropped, so
each (m, τ) cell maps its centred lookback to the horizons through a SINGLE linear
layer instead of two (one for trend, one for seasonal). It is the Zeng et al. 2023
"Linear"/"NLinear" baseline: because the SANTA contract centres each cell's window
on today (instance_norm subtracts the last slice), the input is already
last-value-normalised — exactly NLinear's normalisation — so this is NLinear under
the family's netDelta target.

Where it sits in the ladder
---------------------------
    | model      | per-cell temporal map                    | params (h=21) |
    | DLinear    | trend linear + seasonal linear (summed)  | ~296k         |
    | Linear     | one linear map of the centred lookback   | ~148k         |   <-- this

DLinear vs Linear isolates exactly the value of the moving-average decomposition;
both are channel-independent (no cross-cell mixing) and otherwise identical.

What stays the same as the rest of the family: instance-norm centring on today;
netDelta target with ẑ_{t+h} = z_today + netDelta_h; surface_loss. Because the
window is centred on today (u[:, -1] == 0), the linear forecast of the future
centred level IS the change from today, and a zero map reproduces the random walk.
"""
from __future__ import annotations
import os
import sys

# Make the repo root importable (surface_core) for standalone runs. train.py puts
# the root on sys.path already, so this only matters for `python Linear/linear.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

# Shared Config + instance norm so callers write Linear(Config(...)) like the rest
# of the family. The attention-only Config fields (d, n_heads, n_layers, …) are
# ignored.
from surface_core import Config, instance_norm


class LinearForecaster(nn.Module):
    """Channel-independent single linear map under the SANTA I/O contract.

    Input  : (B, L, M, T) standardised log-IV window.
    Output : (B, n_horizons, M, T) netDelta.

    Each of the M·T cells owns one (Hh × L) weight matrix and a per-horizon bias,
    applied to its centred lookback. No decomposition, no attention, no coordinate
    embeddings — the minimal learned per-cell forecaster.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        n_cells = cfg.M * cfg.T
        Hh, L = cfg.n_horizons, cfg.L
        # One linear map per cell. Initialised to the 1/L prior (output = window
        # mean of the centred lookback), matching DLinear's init; on the centred
        # history this is a mild reversion toward the window mean, with the
        # random-walk null (netDelta = 0) one step away.
        self.W = nn.Parameter((1.0 / L) * torch.ones(n_cells, Hh, L))   # (C, Hh, L)
        self.b = nn.Parameter(torch.zeros(n_cells, Hh))                 # (C, Hh)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, L, M, T = z.shape
        assert (L, M, T) == (cfg.L, cfg.M, cfg.T), "window shape mismatch"

        # Step 1 — centre each cell's window on today (== NLinear's last-value
        # subtraction). The future centred level is the change from today.
        u, _, _ = instance_norm(z)                    # (B, L, M, T)

        # Step 2 — flatten to M·T independent channels, one linear map each.
        x = u.reshape(B, L, M * T).permute(0, 2, 1)   # (B, C, L)
        out = torch.einsum("bcl,chl->bch", x, self.W) + self.b  # (B, C, Hh)

        # Step 3 — restore canonical (B, Hh, M, T).
        netDelta = out.permute(0, 2, 1).reshape(B, cfg.n_horizons, M, T).contiguous()
        return netDelta

    # No attention to collect — provided for API parity with the family.
    def enable_attn_collection(self, on: bool = True):
        pass


# ----------------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    from surface_core import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    cfg = Config(M=11, T=10, L=63, horizons=tuple(range(1, 22)),
                 d=16, n_heads=4, n_layers=2, d_ff_mult=1,
                 d_head_hidden=24, dropout=0.1)
    model = LinearForecaster(cfg)
    print(f"Linear parameters: {sum(p.numel() for p in model.parameters()):,}")

    S = 300
    Z = torch.randn(S, cfg.M, cfg.T).cumsum(0) * 0.02
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)

    windows, z_today, z_future, _ = build_windows(Z, cfg.horizons, cfg.L)
    bs = 64
    xb, tb, fb = windows[:bs], z_today[:bs], z_future[:bs]
    netDelta = model(xb)
    print("netDelta:", tuple(netDelta.shape))

    loss = surface_loss(netDelta, tb, fb)
    base = rw_loss(tb, fb)
    print(f"model loss = {loss.item():.4f}   RW loss = {base.item():.4f}")

    loss.backward()
    print("backward OK, all params have grads:",
          all(p.grad is not None for p in model.parameters() if p.requires_grad))
