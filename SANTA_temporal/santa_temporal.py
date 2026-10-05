"""SANTA-Temporal — temporal-only ablation: both spatial attention blocks removed."""

from __future__ import annotations
import os
import sys

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SANTA = os.path.join(_ROOT, "SANTA")
for _p in (_ROOT, _SANTA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from surface_core import Config, SubBlock
from santa import SANTA



# temporal-only layer
class TemporalOnlyLayer(nn.Module):
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.block_time = SubBlock(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M, T, d = x.shape
        xC = x.permute(0, 2, 3, 1, 4).reshape(B * M * T, L, d)  # (B*M*T, L, d)
        xC = self.block_time(xC)
        x  = xC.reshape(B, M, T, L, d).permute(0, 3, 1, 2, 4)   # -> (B,L,M,T,d)
        return x


# top-level model
class SANTATemporal(SANTA):

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.layers = nn.ModuleList(
            [TemporalOnlyLayer(cfg) for _ in range(cfg.n_layers)]
        )


# smoke test
if __name__ == "__main__":
    from surface_core import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    cfg = Config()
    model = SANTATemporal(cfg)
    print(f"SANTATemporal parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    print("backward OK:",
          all(p.grad is not None for p in model.parameters() if p.requires_grad))

    model.enable_attn_collection(True)
    _ = model(xb[:2])
    cmap = model.layers[0].block_time.attn._attn   # (B*M*T, heads, L, L)
    print("temporal attn map:", tuple(cmap.shape))
