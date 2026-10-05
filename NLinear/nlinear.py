"""NLinear — one shared linear map on the centred lookback (simplest forecasting floor)."""
from __future__ import annotations
import os
import sys


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from surface_core import Config, instance_norm


class NLinearForecaster(nn.Module):
    """Single linear map shared across all M·T cells (NLinear, individual=False).

    (B, L, M, T) standardised log-IV in, (B, n_horizons, M, T) netDelta out: one
    (Hh × L) weight + per-horizon bias applied to each cell's centred lookback.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        Hh, L = cfg.n_horizons, cfg.L
        # one linear map shared across all cells, init 1/L (output = window mean of the centred lookback)
        self.proj = nn.Linear(L, Hh)                                    # (L -> Hh)
        nn.init.constant_(self.proj.weight, 1.0 / L)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, L, M, T = z.shape
        assert (L, M, T) == (cfg.L, cfg.M, cfg.T), "window shape mismatch"

        # centre each cell's window on today; the future centred level is the change from today
        u, _, _ = instance_norm(z)                    # (B, L, M, T)

        # flatten to M·T channels and apply the shared linear map
        x = u.reshape(B, L, M * T).permute(0, 2, 1)   # (B, C, L)
        out = self.proj(x)                            # (B, C, Hh)

        # restore canonical (B, Hh, M, T)
        netDelta = out.permute(0, 2, 1).reshape(B, cfg.n_horizons, M, T).contiguous()
        return netDelta

    def enable_attn_collection(self, on: bool = True):
        pass


# smoke test
if __name__ == "__main__":
    from surface_core import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    cfg = Config(M=11, T=10, L=63, horizons=tuple(range(1, 22)),
                 d=16, n_heads=4, n_layers=2, d_ff_mult=1,
                 d_head_hidden=24, dropout=0.1)
    model = NLinearForecaster(cfg)
    print(f"NLinear parameters: {sum(p.numel() for p in model.parameters()):,}")

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
