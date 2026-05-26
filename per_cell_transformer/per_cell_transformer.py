"""
per_cell_transformer.py
===================================================================================
PerCellTransformer — SANTA-Temporal with coordinate embeddings removed.

The missing cell in the factorial design

    | model              | coord embeddings | cross-cell mixing |
    | SANTA-Temporal     |   yes (k, log τ) |   none            |
    | VanillaTransformer |   no             |   bottleneck      |
    | PerCellTransformer |   NO             |   NONE            |    <-- this file

Diff from SANTA-Temporal (everything else stays byte-for-byte)

    DELETE  CoordinateEmbedding for moneyness         (self.emb_money)
    DELETE  CoordinateEmbedding for log τ              (self.emb_mat)
    DELETE  Buffers k_coords / log_tau_coords
    DELETE  Their addition into the token in forward(): val + pK + pT + pL  →  val + pL

Kept verbatim:

    * per-cell tokenisation: each (m, τ) cell is its own sequence; attention
      runs per-cell over L with shared weights; cells never mix at forward time
    * value_proj (Linear(1 → d)) per-cell value embedding
    * emb_lag (Embedding(L, d)) — temporal positional info, without which the
      temporal attention would be order-invariant and the model couldn't tell
      "yesterday" from "60 days ago"
    * instance-norm centring on today, per-cell window scale
    * TemporalOnlyLayer × n_layers (same as SANTA-Temporal)
    * final LayerNorm
    * head (per-cell shared MLP with re-injected level + scale)
    * forward signature: (B, L, M, T) -> netDelta (B, n_horizons, M, T)

What this isolates

    Comparison A   PerCellTransformer  vs  SANTA-Temporal
                   Same per-block specs (d, n_heads, n_layers, ff_mult,
                   d_head_hidden, dropout), same tokenisation, same backbone,
                   same head. ONLY DIFFERENCE: coordinate embeddings on/off.
                   The 6-7k param delta IS the coord-embedding cost.

    Comparison B   PerCellTransformer  vs  VanillaTransformer
                   Both lack coords. The only architectural difference is
                   tokenisation: per-cell sequences (PerCell) vs day-token
                   bottleneck (Vanilla). Isolates whether the per-cell
                   structure adds value over reducing 110 cells through a
                   d-dim bottleneck.

Together (A) and (B) decompose the SANTA-vs-VanillaTransformer gap into
its two additive components: how much was coords, how much was tokenisation.
"""
from __future__ import annotations
import os
import sys

# Make SANTA/ and SANTA_temporal/ importable when invoked standalone.
_HERE  = os.path.dirname(os.path.abspath(__file__))
_SANTA = os.path.join(os.path.dirname(_HERE), "SANTA")
_TEMPO = os.path.join(os.path.dirname(_HERE), "SANTA_temporal")
for p in (_SANTA, _TEMPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn

# Re-export the shared Config so callers write PerCellTransformer(Config(...))
# the same way they write SANTATemporal(Config(...)). k_grid / tau_grid_years
# are accepted for Config completeness but deliberately ignored — no coord
# embeddings consume them.
from santa import (                                          # noqa: E402
    Config,
    SubBlock,
    MultiHeadSelfAttention,
    SANTA,
)
from santa_temporal import TemporalOnlyLayer                 # noqa: E402


class PerCellTransformer(SANTA):
    """SANTA-Temporal minus the coordinate embeddings.

    Inherits ``_instance_norm`` and ``enable_attn_collection`` from SANTA
    (they don't depend on the deleted modules). The remaining state +
    forward are built from scratch so that the deleted modules are never
    instantiated — no wasted parameters, no orphan buffers, no need for
    runtime null-checks.
    """

    def __init__(self, cfg: Config):
        # Skip SANTA's __init__ — it would build emb_money / emb_mat /
        # k_coords / log_tau_coords (the things we're removing). Initialise
        # nn.Module directly and build only what we need.
        nn.Module.__init__(self)
        self.cfg = cfg

        # Step 1 (partial): value-projection embedding (per-cell scalar -> d).
        # Coordinate embeddings deliberately omitted.
        self.value_proj = nn.Linear(1, cfg.d)

        # Temporal positional encoding stays — without it the temporal
        # attention is permutation-invariant in time, which would gut
        # the model. This is the "lag" embedding from SANTA verbatim.
        self.emb_lag = nn.Embedding(cfg.L, cfg.d)
        self.register_buffer("lag_idx", torch.arange(cfg.L))
        self.emb_drop = nn.Dropout(cfg.dropout)

        # Backbone: temporal-only layers (identical to SANTA-Temporal's
        # self.layers, including the per-cell-batched reshape inside).
        self.layers = nn.ModuleList(
            [TemporalOnlyLayer(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_ln = nn.LayerNorm(cfg.d)

        # Head: per-cell shared MLP with level + scale re-injection.
        # Identical signature to SANTA's head — (d+2) input includes
        # today's per-cell level and per-cell window scale, output is
        # the n_horizons netDelta per cell.
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

        # Step 0: instance-norm centring (inherited from SANTA).
        u, L0, s_tilde, z_today = self._instance_norm(z)

        # Step 1: value embedding only — no coordinate embeddings.
        val = self.value_proj(u.unsqueeze(-1))           # (B, L, M, T, d)
        pL  = self.emb_lag(self.lag_idx)                  # (L, d)
        x   = val + pL[None, :, None, None, :]            # add lag along L
        x   = self.emb_drop(x)

        # Step 2: temporal-only attention stack (per-cell with shared weights).
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
    from santa import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    cfg = Config(
        M=11, T=10, L=63,
        horizons=tuple(range(1, 22)),
        d=56, n_heads=4, n_layers=2, d_ff_mult=1,
        d_head_hidden=24, dropout=0.1,
    )
    model = PerCellTransformer(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"PerCellTransformer parameters: {n:,}")
    # Compare to SANTA-Temporal at the same config.
    from santa_temporal import SANTATemporal
    n_st = sum(p.numel() for p in SANTATemporal(cfg).parameters())
    print(f"SANTA-Temporal parameters:     {n_st:,}")
    print(f"Difference (== coord emb cost): {n_st - n:,}")

    # Verify shape contract.
    S = 300
    Z = torch.randn(S, cfg.M, cfg.T).cumsum(0) * 0.02
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    windows, z_today, z_future, _ = build_windows(Z, cfg.horizons, cfg.L)
    bs = 64
    xb, tb, fb = windows[:bs], z_today[:bs], z_future[:bs]
    netDelta = model(xb)
    print("netDelta:", tuple(netDelta.shape))            # (bs, Hh, M, T)
    loss = surface_loss(netDelta, tb, fb)
    base = rw_loss(tb, fb)
    print(f"model loss = {loss.item():.4f}   RW loss = {base.item():.4f}")
    loss.backward()
    print("backward OK:",
          all(p.grad is not None for p in model.parameters() if p.requires_grad))
