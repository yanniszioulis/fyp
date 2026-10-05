"""Coordinate embedding: MLP mapping a scalar grid coordinate to R^d."""

from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateEmbedding(nn.Module):
    """MLP mapping a scalar financial coordinate to R^d (one embedding per grid value)."""


    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d), nn.GELU(), nn.Linear(d, d)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (n,) -> (n, d)
        return self.net(coords.unsqueeze(-1))
