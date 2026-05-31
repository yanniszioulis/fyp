"""
santa_temporal.py
===================================================================================
SANTA-Temporal — temporal-only ablation of SANTA.

Keeps Block C (per-cell temporal attention over the L lags) and REMOVES both
spatial blocks — no moneyness attention, no maturity attention. Each cell evolves
purely as its own time series, attended over its own history; no information flows
BETWEEN cells inside the backbone. Everything else is inherited verbatim from SANTA
by subclassing.

This is the mirror-image ablation of SANTA-Flat:

    SANTA          : factored spatial (A + B) + temporal (C)
    SANTA-Flat     : JOINT spatial (S)        + temporal (C)   [vary the spatial mix]
    SANTA-Temporal : no spatial               + temporal (C)   [remove the spatial mix]

so the three together isolate whether the model needs any cross-cell information at
all, and if so whether the two spatial axes should be mixed jointly or in factored
Kronecker style. Cross-cell information can still enter only through the (constant)
coordinate embeddings at Step 1 and the shared head — never inside the trunk.
"""

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


# ----------------------------------------------------------------------------------
# Temporal-only layer: one SubBlock over the L lags at every cell. No spatial mix.
# ----------------------------------------------------------------------------------
class TemporalOnlyLayer(nn.Module):
    """One layer of the SANTA-Temporal backbone: a single temporal SubBlock.

      C) Temporal attention — sequence axis is L. The reshape/permute is
         byte-for-byte the same as SANTA's FactoredLayer.block_time; the temporal
         mechanism is the half of the model held constant across all three variants.

    No moneyness/maturity/joint-spatial block: the M·T cells are processed as
    independent per-cell time series at every layer.
    """
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


# ----------------------------------------------------------------------------------
# Top-level model: SANTA with both spatial blocks deleted (temporal-only stack)
# ----------------------------------------------------------------------------------
class SANTATemporal(SANTA):
    """SANTA with both spatial blocks removed; layer stack is temporal-only.

    Inherits SANTA's __init__ (embeddings, head, buffers, the FactoredLayer stack)
    then replaces self.layers in place. cfg.n_layers and every per-block hyperparam
    are unchanged; forward / instance norm / attention collection are inherited.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.layers = nn.ModuleList(
            [TemporalOnlyLayer(cfg) for _ in range(cfg.n_layers)]
        )


# ----------------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------------
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
