"""
per_cell_transformer.py
===================================================================================
PerCellTransformer — SANTA-Temporal with the coordinate embeddings removed.

The missing cell in the factorial design:

    model               coord embeddings   cross-cell mixing
    SANTA-Temporal      yes (k, log τ)     none
    VanillaTransformer  no                 bottleneck (Linear(M·T → d))
    PerCellTransformer  NO                 NONE                       <-- this file

Per-cell tokenisation (each (m, τ) cell is its own length-L sequence, attended over
with shared weights, cells never mix in the trunk) is kept; only the surface-aware
coordinate embeddings are dropped — so the model keeps the value embedding and the
lag embedding (without which temporal attention would be order-invariant).

What this isolates
    A. PerCell vs SANTA-Temporal     → the cost/value of the coordinate embeddings
    B. PerCell vs VanillaTransformer → per-cell sequences vs the day-token bottleneck
Together they decompose the SANTA-vs-VanillaTransformer gap into its coordinate and
tokenisation components.
"""

from __future__ import annotations
import os
import sys

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SANTA = os.path.join(_ROOT, "SANTA")
_TEMPO = os.path.join(_ROOT, "SANTA_temporal")
for _p in (_ROOT, _SANTA, _TEMPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn

from surface_core import Config, instance_norm
from santa import SANTA
from santa_temporal import TemporalOnlyLayer


class PerCellTransformer(SANTA):
    """SANTA-Temporal minus the coordinate embeddings.

    Inherits enable_attn_collection from SANTA. __init__ is built from scratch (it
    does NOT call SANTA.__init__) so the deleted coordinate modules / buffers are
    never instantiated — no wasted parameters, no orphan buffers.
    """

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        # Value embedding only (per-cell scalar → d); coordinate embeddings omitted.
        self.value_proj = nn.Linear(1, cfg.d)

        # Lag embedding stays — without it temporal attention is permutation-
        # invariant in time. Identical to SANTA's lag table.
        self.emb_lag = nn.Embedding(cfg.L, cfg.d)
        self.register_buffer("lag_idx", torch.arange(cfg.L))
        self.emb_drop = nn.Dropout(cfg.dropout)

        # Backbone: temporal-only layers (identical to SANTA-Temporal).
        self.layers = nn.ModuleList(
            [TemporalOnlyLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_ln = nn.LayerNorm(cfg.d)

        # Head: per-cell shared MLP with level + scale re-injection (same as SANTA).
        self.head = nn.Sequential(
            nn.Linear(cfg.d + 2, cfg.d_head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_head_hidden, cfg.n_horizons),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B, L, M, T = z.shape
        assert (L, M, T) == (cfg.L, cfg.M, cfg.T), "window shape mismatch"

        # Step 0: instance-norm centring.
        u, L0, s_tilde = instance_norm(z)

        # Step 1: value embedding + lag only — no coordinate embeddings.
        val = self.value_proj(u.unsqueeze(-1))           # (B, L, M, T, d)
        pL  = self.emb_lag(self.lag_idx)                  # (L, d)
        x   = val + pL[None, :, None, None, :]            # add lag along L
        x   = self.emb_drop(x)

        # Step 2: temporal-only attention stack (per-cell, shared weights).
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)                              # (B, L, M, T, d)

        # Step 3: today-token readout at every cell.
        r = x[:, -1, :, :, :]                             # (B, M, T, d)

        # Step 4: re-inject level + scale and project through the head.
        g = torch.cat([r, L0.unsqueeze(-1), s_tilde.unsqueeze(-1)], dim=-1)
        netDelta = self.head(g)                           # (B, M, T, Hh)
        netDelta = netDelta.permute(0, 3, 1, 2).contiguous()
        return netDelta


# ----------------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    from surface_core import build_windows, surface_loss, rw_loss
    from santa_temporal import SANTATemporal
    torch.manual_seed(0)
    cfg = Config(
        M=11, T=10, L=63,
        horizons=tuple(range(1, 22)),
        d=56, n_heads=4, n_layers=2, d_ff_mult=1,
        d_head_hidden=24, dropout=0.1,
    )
    model = PerCellTransformer(cfg)
    n = sum(p.numel() for p in model.parameters())
    n_st = sum(p.numel() for p in SANTATemporal(cfg).parameters())
    print(f"PerCellTransformer parameters: {n:,}")
    print(f"SANTA-Temporal parameters:     {n_st:,}")
    print(f"Difference (== coord emb cost): {n_st - n:,}")

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
