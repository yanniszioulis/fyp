"""
santa_flat.py
===================================================================================
SANTA-Flat — joint-spatial ablation of SANTA.

Replaces SANTA's factored moneyness (A) + maturity (B) sub-blocks with a
single JOINT spatial sub-block that attends over all M·T = 150 cells of
the surface at once. The temporal block (C) is unchanged. Every other
component — Config, instance norm, value/coordinate/lag embeddings, the
final LayerNorm, the regression head, the forward signature, the
attention-collection switch — is inherited verbatim from ``santa.SANTA``
by subclassing, so the ablation isolates exactly one design choice:

    Should the two spatial axes (moneyness × maturity) be mixed
    SEPARATELY (Kronecker / factored — SANTA) or JOINTLY (full self-
    attention over the 150 flattened cells — SANTAFlat)?

Per-layer block layout
----------------------
                   SANTA (factored)              SANTAFlat (joint)
    spatial      A: L·T seqs of length M=15      S: L  seqs of length M·T = 150
                 B: L·M seqs of length T=10
    temporal     C: M·T seqs of length L=63      C: M·T seqs of length L = 63
                                                   (IDENTICAL to SANTA's C)

Parameter count: each SubBlock has the same width, so SANTAFlat has one
fewer SubBlock per layer (2 vs 3) and is therefore slightly SMALLER in
parameters — but its joint spatial block runs over a length-150 sequence
so compute per layer is higher. We accept the asymmetry: the ablation is
about WHAT the spatial mixer sees, not about matching FLOPs.

All imports come from santa.py so any tweak to embeddings, instance
norm, MHSA, FFN, or SubBlock automatically propagates to SANTAFlat.
"""

from __future__ import annotations
import os
import sys

# Make SANTA/ importable when invoked standalone. train.py already adds
# both SANTA and SANTA_flat to sys.path, so this insert is only needed
# for `python SANTA_flat/santa_flat.py` smoke runs.
_HERE  = os.path.dirname(os.path.abspath(__file__))
_SANTA = os.path.join(os.path.dirname(_HERE), "SANTA")
if _SANTA not in sys.path:
    sys.path.insert(0, _SANTA)

import torch
import torch.nn as nn

# Re-export the shared Config so callers can write SANTAFlat(Config(...))
# the same way they write SANTA(Config(...)).
from santa import (                                           # noqa: E402
    Config,
    CoordinateEmbedding,
    MultiHeadSelfAttention,
    FeedForward,
    SubBlock,
    SANTA,
)


# ----------------------------------------------------------------------------------
# Joint-spatial layer: one SubBlock over flattened (M·T), then SANTA's C verbatim
# ----------------------------------------------------------------------------------
class FlatSpatialLayer(nn.Module):
    """One layer of the SANTAFlat backbone.

    Acts on (B, L, M, T, d) and applies two pre-LN transformer SubBlocks:

      S) JOINT spatial attention — sequence axis is the flattened M·T = 150
         cells, with everything else (B and L) in the batch. Replaces the
         A→B factored pair from SANTA's FactoredLayer with one block.

      C) Temporal attention — sequence axis is L. The reshape/permute is
         byte-for-byte the same as ``santa.FactoredLayer``'s block_time,
         because the temporal mechanism is the half of the model we are
         deliberately holding constant.

    Attention weights are shared across the flattened-batch positions
    (standard self-attention), so this is one joint spatial operator
    applied at every day, plus one shared temporal operator applied at
    every cell — same architectural style as SANTA, different spatial
    factoring.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # Joint spatial block over (M·T) cells. SubBlock(cfg) is the
        # same pre-LN MHSA + FFN unit SANTA uses for A/B/C — only the
        # sequence-axis dimensions differ.
        self.block_spatial = SubBlock(cfg)
        # Temporal block — copy of SANTA.FactoredLayer.block_time.
        self.block_time = SubBlock(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M, T, d = x.shape

        # ---- Joint spatial block: attend over ALL M·T cells per (B,L) slice
        # Flatten (M, T) -> single axis of length M*T. Batch is B*L: every
        # (sample, day) pair is one independent self-attention problem.
        xS = x.reshape(B * L, M * T, d)                       # (B*L, M*T, d)
        xS = self.block_spatial(xS)
        x  = xS.reshape(B, L, M, T, d)                        # back to canonical

        # ---- Temporal block C (identical to SANTA's FactoredLayer.block_time)
        xC = x.permute(0, 2, 3, 1, 4).reshape(B * M * T, L, d)  # (B*M*T, L, d)
        xC = self.block_time(xC)
        x  = xC.reshape(B, M, T, L, d).permute(0, 3, 1, 2, 4)   # -> (B,L,M,T,d)
        return x


# ----------------------------------------------------------------------------------
# Top-level model: inherits everything from SANTA except the layer stack
# ----------------------------------------------------------------------------------
class SANTAFlat(SANTA):
    """SANTA with the factored-spatial layer stack swapped for joint-spatial.

    Inherits ``__init__`` (which builds embeddings, the final LayerNorm,
    the head, the coordinate / lag buffers, and the original
    ``self.layers`` ModuleList of ``FactoredLayer``s) from SANTA, then
    REPLACES ``self.layers`` in place with the joint-spatial variant.
    Forward, instance norm, and attention-collection switching are
    inherited unchanged.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # The only difference from SANTA: swap the factored layer stack
        # for the joint-spatial one. cfg.n_layers and every per-block
        # hyperparam (d, n_heads, d_ff_mult, dropout) are unchanged.
        self.layers = nn.ModuleList(
            [FlatSpatialLayer(cfg) for _ in range(cfg.n_layers)]
        )


# ----------------------------------------------------------------------------------
# Smoke test: verify shapes + a single training step run end-to-end
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    from santa import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    cfg = Config()
    model = SANTAFlat(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SANTAFlat parameters: {n_params:,}")

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

    # Diagnostic: pull the JOINT-spatial attention map (block_spatial,
    # layer 0). Expect shape (B*L, n_heads, M*T, M*T) = (B*L, h, 150, 150).
    model.enable_attn_collection(True)
    _ = model(xb[:2])
    smap = model.layers[0].block_spatial.attn._attn
    cmap = model.layers[0].block_time.attn._attn
    print("joint-spatial attn map:", tuple(smap.shape))
    print("temporal attn map     :", tuple(cmap.shape))
