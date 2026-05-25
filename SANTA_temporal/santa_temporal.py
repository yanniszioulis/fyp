"""
santa_temporal.py
===================================================================================
SANTA-Temporal — temporal-only ablation of SANTA.

Keeps Block C (per-cell temporal attention over the L lags) and REMOVES
both spatial blocks — no moneyness attention, no maturity attention. Each
of the 150 cells evolves purely as its own time series, attended over its
own history; no information ever flows BETWEEN cells inside the backbone.
Every other component — Config, instance norm, value/coordinate/lag
embeddings, the final LayerNorm, the regression head, the forward
signature, the attention-collection switch — is inherited verbatim from
``santa.SANTA`` by subclassing.

This is the mirror-image ablation of SANTAFlat:

    SANTA          : factored spatial (A: moneyness, B: maturity) + C (temporal)
    SANTAFlat      : JOINT spatial (S: M·T cells) + C (temporal)      [vary spatial]
    SANTATemporal  : no spatial          + C (temporal)               [vary spatial]

So the three together isolate, with everything else held constant:

    Does the model need any cross-cell information at all (SANTATemporal),
    and if so, should the two spatial axes be mixed jointly (SANTAFlat)
    or in factored Kronecker style (SANTA)?

Per-layer block layout
----------------------
                  SANTA (factored)              SANTAFlat (joint)        SANTATemporal (no-spatial)
    spatial     A: L·T seqs of length M=15      S: L  seqs of length 150  —
                B: L·M seqs of length T=10
    temporal    C: M·T seqs of length L=63      C: identical              C: identical

Parameter count: each layer now has ONE SubBlock instead of three (SANTA)
or two (SANTAFlat). SANTATemporal therefore has the fewest parameters of
the family, but the same per-cell temporal mixer as the other two — so a
loss in MSE relative to SANTA/SANTAFlat is attributable specifically to
the loss of cross-cell information flow.

All imports come from santa.py so any tweak to embeddings, instance
norm, MHSA, FFN, or SubBlock automatically propagates here.
"""

from __future__ import annotations
import os
import sys

# Make SANTA/ importable when invoked standalone. train.py already adds
# SANTA and SANTA_temporal to sys.path, so this insert is only needed
# for `python SANTA_temporal/santa_temporal.py` smoke runs.
_HERE  = os.path.dirname(os.path.abspath(__file__))
_SANTA = os.path.join(os.path.dirname(_HERE), "SANTA")
if _SANTA not in sys.path:
    sys.path.insert(0, _SANTA)

import torch
import torch.nn as nn

# Re-export the shared Config so callers can write SANTATemporal(Config(...))
# the same way they write SANTA(Config(...)) / SANTAFlat(Config(...)).
from santa import (                                           # noqa: E402
    Config,
    CoordinateEmbedding,
    MultiHeadSelfAttention,
    FeedForward,
    SubBlock,
    SANTA,
)


# ----------------------------------------------------------------------------------
# Temporal-only layer: one SubBlock over the L lags at every cell. No spatial mix.
# ----------------------------------------------------------------------------------
class TemporalOnlyLayer(nn.Module):
    """One layer of the SANTATemporal backbone.

    Acts on (B, L, M, T, d) and applies ONE pre-LN transformer SubBlock:

      C) Temporal attention — sequence axis is L. The reshape/permute is
         byte-for-byte the same as ``santa.FactoredLayer``'s block_time
         and ``santa_flat.FlatSpatialLayer``'s block_time, because the
         temporal mechanism is the half of the model we are deliberately
         holding constant across all three variants.

    No moneyness block, no maturity block, no joint-spatial block — the
    150 cells are processed as 150 independent per-cell time series at
    every layer of the backbone. Cross-cell information can only flow
    through the coordinate embeddings added at Step 1 (which are
    constant per cell) and through the head's shared MLP (which sees
    each cell's readout independently). Inside the trunk, cells never
    talk to each other.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Single temporal block — copy of SANTA.FactoredLayer.block_time.
        self.block_time = SubBlock(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M, T, d = x.shape
        # ---- Temporal block C (identical to SANTA's FactoredLayer.block_time)
        xC = x.permute(0, 2, 3, 1, 4).reshape(B * M * T, L, d)  # (B*M*T, L, d)
        xC = self.block_time(xC)
        x  = xC.reshape(B, M, T, L, d).permute(0, 3, 1, 2, 4)   # -> (B,L,M,T,d)
        return x


# ----------------------------------------------------------------------------------
# Top-level model: inherits everything from SANTA except the layer stack
# ----------------------------------------------------------------------------------
class SANTATemporal(SANTA):
    """SANTA with both spatial blocks deleted; layer stack is temporal-only.

    Inherits ``__init__`` (which builds embeddings, the final LayerNorm,
    the head, the coordinate / lag buffers, and the original
    ``self.layers`` ModuleList of ``FactoredLayer``s) from SANTA, then
    REPLACES ``self.layers`` in place with the temporal-only variant.
    Forward, instance norm, and attention-collection switching are
    inherited unchanged.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # The only difference from SANTA: swap the factored layer stack
        # for the temporal-only one. cfg.n_layers and every per-block
        # hyperparam (d, n_heads, d_ff_mult, dropout) are unchanged.
        self.layers = nn.ModuleList(
            [TemporalOnlyLayer(cfg) for _ in range(cfg.n_layers)]
        )


# ----------------------------------------------------------------------------------
# Smoke test: verify shapes + a single training step run end-to-end
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    from santa import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    cfg = Config()
    model = SANTATemporal(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SANTATemporal parameters: {n_params:,}")

    # fake standardised series: 300 days of a 15x10 surface
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
    print("netDelta:", tuple(netDelta.shape))      # (bs, Hh, M, T)

    loss = surface_loss(netDelta, tb, fb)
    base = rw_loss(tb, fb)
    print(f"model loss = {loss.item():.4f}   RW loss = {base.item():.4f}")

    loss.backward()
    grad_ok = all(p.grad is not None for p in model.parameters()
                  if p.requires_grad)
    print("backward OK, all params have grads:", grad_ok)

    # Diagnostic: pull the (only) temporal attention map.
    # Expect shape (B*M*T, n_heads, L, L) — identical to SANTA's C.
    model.enable_attn_collection(True)
    _ = model(xb[:2])
    cmap = model.layers[0].block_time.attn._attn
    print("temporal attn map:", tuple(cmap.shape))
