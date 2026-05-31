"""
embeddings.py
===================================================================================
Surface-aware coordinate embeddings.

The one component that makes a model "surface-aware": it maps each grid cell's
*continuous* financial coordinate (log-fwd-moneyness k, or log-maturity log τ) to
a d-vector, so proximity in coordinate space becomes proximity in embedding space
and the ATM pivot (k=0) is encoded explicitly. SANTA and its spatial ablations add
these to the per-cell value embedding; the vanilla / per-cell transformers and
DLinear deliberately omit them (that omission is the point of those ablations).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateEmbedding(nn.Module):
    """MLP mapping a scalar financial coordinate -> R^d. One embedding per grid value.

    Using the continuous coordinate (not a grid index) is what makes the model
    surface-aware: nearby strikes / maturities get nearby embeddings, and the model
    can interpolate along the surface rather than treating cells as unordered.
    """
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d), nn.GELU(), nn.Linear(d, d)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (n,) -> (n, d)
        return self.net(coords.unsqueeze(-1))
