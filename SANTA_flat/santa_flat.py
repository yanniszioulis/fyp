"""
santa_flat.py
===================================================================================
SANTA-Flat — joint-spatial ablation of SANTA.

Replaces SANTA's factored moneyness (A) + maturity (B) sub-blocks with a single
JOINT spatial sub-block that attends over all M·T cells of the surface at once. The
temporal block (C) is unchanged, and everything else — embeddings, instance norm,
head, forward — is inherited verbatim from SANTA by subclassing. The ablation
isolates one design choice:

    should the two spatial axes (moneyness × maturity) be mixed SEPARATELY
    (Kronecker / factored — SANTA) or JOINTLY (full self-attention over the
    flattened cells — SANTA-Flat)?

Per-layer block layout
----------------------
                   SANTA (factored)                 SANTA-Flat (joint)
    spatial      A: L·T seqs of length M            S: L seqs of length M·T
                 B: L·M seqs of length T
    temporal     C: M·T seqs of length L            C: identical to SANTA's C
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
# Joint-spatial layer: one SubBlock over flattened (M·T), then SANTA's C verbatim
# ----------------------------------------------------------------------------------
class FlatSpatialLayer(nn.Module):
    """One layer of the SANTA-Flat backbone: joint-spatial block S, then temporal C.

      S) JOINT spatial attention — sequence axis is the flattened M·T cells, with
         (B, L) in the batch. Replaces SANTA's factored A→B pair with one block.
      C) Temporal attention — byte-for-byte the same reshape as SANTA's block_time,
         because the temporal mechanism is held constant across the family.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.block_spatial = SubBlock(cfg)   # S: over M·T
        self.block_time = SubBlock(cfg)      # C: over L (copy of SANTA's block_time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M, T, d = x.shape

        # ---- Joint spatial block S: attend over ALL M·T cells per (B,L) slice ---
        xS = x.reshape(B * L, M * T, d)                       # (B*L, M*T, d)
        xS = self.block_spatial(xS)
        x  = xS.reshape(B, L, M, T, d)

        # ---- Temporal block C (identical to SANTA's FactoredLayer.block_time) ---
        xC = x.permute(0, 2, 3, 1, 4).reshape(B * M * T, L, d)  # (B*M*T, L, d)
        xC = self.block_time(xC)
        x  = xC.reshape(B, M, T, L, d).permute(0, 3, 1, 2, 4)   # -> (B,L,M,T,d)
        return x


# ----------------------------------------------------------------------------------
# Top-level model: SANTA with the factored layer stack swapped for joint-spatial
# ----------------------------------------------------------------------------------
class SANTAFlat(SANTA):
    """SANTA with the factored-spatial layer stack replaced by joint-spatial.

    Inherits SANTA's __init__ (embeddings, head, buffers, the FactoredLayer stack)
    then replaces self.layers in place. cfg.n_layers and every per-block hyperparam
    (d, n_heads, d_ff_mult, dropout) are unchanged; forward / instance norm /
    attention collection are inherited.
    """
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.layers = nn.ModuleList(
            [FlatSpatialLayer(cfg) for _ in range(cfg.n_layers)]
        )


# ----------------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    from surface_core import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    cfg = Config()
    model = SANTAFlat(cfg)
    print(f"SANTAFlat parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    smap = model.layers[0].block_spatial.attn._attn   # (B*L, heads, M*T, M*T)
    cmap = model.layers[0].block_time.attn._attn      # (B*M*T, heads, L, L)
    print("joint-spatial attn map:", tuple(smap.shape))
    print("temporal attn map     :", tuple(cmap.shape))
