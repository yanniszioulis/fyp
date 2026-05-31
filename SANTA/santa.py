"""
santa.py
===================================================================================
SANTA — Surface-Aware Neural Tensor Attention.

The full model. The surface is carried throughout as a (B, L, M, T) object — never
flattened to a vector or compressed to factors — and processed by N factored
attention layers, each attending over one axis at a time:

    A) over moneyness   (central-smile skew / curvature)
    B) over maturity    (term structure)
    C) over time / lags (state-dependent persistence / long memory)

Pipeline (the shared framework pieces — instance norm, embeddings, loss — live in
surface_core.py / embeddings.py):
    0. instance-norm: centre each cell's window on today; keep today's level L0 and
       the per-cell window scale s̃ as head features (amplitude is a regime cue).
    1. per-cell value embedding (scalar→d) + coordinate embeddings of
       (log-fwd-moneyness, log τ) + a learned lag embedding.
    2. N FactoredLayers (A→B→C), each a pre-LN SubBlock per axis.
    3. read out the TODAY token at every cell (it has attended over history+surface).
    4. shared per-cell head with today's level + window scale re-injected, so
       level-dependent mean reversion is learnable.
    5. predict the per-cell change as a residual on today (netDelta).
"""

from __future__ import annotations
import os
import sys

# Make the repo root importable (surface_core, embeddings) for standalone runs.
# train.py puts the root on sys.path already, so this only matters for
# `python SANTA/santa.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from surface_core import Config, SubBlock, MultiHeadSelfAttention, instance_norm
from embeddings import CoordinateEmbedding


# ----------------------------------------------------------------------------------
# One factored layer: A (moneyness) -> B (maturity) -> C (time)
# ----------------------------------------------------------------------------------
class FactoredLayer(nn.Module):
    """Three SubBlocks, each attending over ONE axis of the (B,L,M,T,d) tensor.

    The reshaping is the whole trick: to attend over an axis we move that axis to
    the sequence position (-2) and flatten everything else into the batch, run a
    standard MHSA, then restore the layout. One operator is learned per axis (e.g. a
    single temporal-dynamics operator applied at every cell), but its output is
    state-dependent because the inputs — and hence the softmax weights — differ per
    location/day.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.block_money = SubBlock(cfg)   # A: over M
        self.block_mat = SubBlock(cfg)     # B: over T
        self.block_time = SubBlock(cfg)    # C: over L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, M, T, d = x.shape

        # ---- Block A: attend over MONEYNESS (M), batch = (B,L,T) ----------------
        xA = x.permute(0, 1, 3, 2, 4).reshape(B * L * T, M, d)   # (B*L*T, M, d)
        xA = self.block_money(xA)
        x = xA.reshape(B, L, T, M, d).permute(0, 1, 3, 2, 4)     # -> (B,L,M,T,d)

        # ---- Block B: attend over MATURITY (T), batch = (B,L,M) -----------------
        xB = x.reshape(B * L * M, T, d)                          # (B*L*M, T, d)
        xB = self.block_mat(xB)
        x = xB.reshape(B, L, M, T, d)                            # -> (B,L,M,T,d)

        # ---- Block C: attend over TIME (L), batch = (B,M,T) ---------------------
        xC = x.permute(0, 2, 3, 1, 4).reshape(B * M * T, L, d)   # (B*M*T, L, d)
        xC = self.block_time(xC)
        x = xC.reshape(B, M, T, L, d).permute(0, 3, 1, 2, 4)     # -> (B,L,M,T,d)

        return x


# ----------------------------------------------------------------------------------
# The full model
# ----------------------------------------------------------------------------------
class SANTA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # --- Step 1 embeddings ---------------------------------------------------
        # moneyness coordinate k directly; maturity as log(τ in years) — the data's
        # natural metric; lag as a learned discrete embedding over L.
        self.value_proj = nn.Linear(1, cfg.d)                 # lift scalar -> d
        self.emb_money = CoordinateEmbedding(cfg.d)           # pK(k)
        self.emb_mat = CoordinateEmbedding(cfg.d)             # pT(log τ)
        self.emb_lag = nn.Embedding(cfg.L, cfg.d)             # pL(lag); learned table
        self.emb_drop = nn.Dropout(cfg.dropout)

        # fixed grid coordinates as buffers (move with .to(device))
        self.register_buffer("k_coords",
                             torch.tensor(cfg.k_grid, dtype=torch.float32))         # (M,)
        self.register_buffer("log_tau_coords",
                             torch.log(torch.tensor(cfg.tau_grid_years,
                                                    dtype=torch.float32)))          # (T,)
        self.register_buffer("lag_idx", torch.arange(cfg.L))                        # (L,)

        # --- Step 2 stacked factored layers -------------------------------------
        self.layers = nn.ModuleList([FactoredLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(cfg.d)

        # --- Step 4 head: input is [r ; today_level ; window_scale] = d + 2 ------
        self.head = nn.Sequential(
            nn.Linear(cfg.d + 2, cfg.d_head_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_head_hidden, cfg.n_horizons)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, M, T) standardised log-IV window.
        Returns netDelta: (B, n_horizons, M, T) = predicted cumulative standardised
        change from today to each horizon (the residual added to today's surface).
        """
        cfg = self.cfg
        B, L, M, T = z.shape
        assert (L, M, T) == (cfg.L, cfg.M, cfg.T), "window shape mismatch with config"

        # Step 0 ------------------------------------------------------------------
        u, L0, s_tilde = instance_norm(z)

        # Step 1: embeddings ------------------------------------------------------
        val = self.value_proj(u.unsqueeze(-1))                 # (B,L,M,T,d)
        pK = self.emb_money(self.k_coords)                     # (M,d)
        pT = self.emb_mat(self.log_tau_coords)                 # (T,d)
        pL = self.emb_lag(self.lag_idx)                        # (L,d)
        x = (val
             + pK[None, None, :, None, :]                      # broadcast over (B,L,*,T)
             + pT[None, None, None, :, :]                      # broadcast over (B,L,M,*)
             + pL[None, :, None, None, :])                     # broadcast over (B,*,M,T)
        x = self.emb_drop(x)                                   # (B,L,M,T,d)

        # Step 2: factored attention layers --------------------------------------
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)                                   # (B,L,M,T,d)

        # Step 3: readout the TODAY token at every cell --------------------------
        r = x[:, -1, :, :, :]                                  # (B,M,T,d)

        # Step 4: re-inject level + scale, shared head over cells ----------------
        g = torch.cat([r, L0.unsqueeze(-1), s_tilde.unsqueeze(-1)], dim=-1)  # (B,M,T,d+2)
        netDelta = self.head(g)                                # (B,M,T,Hh)
        netDelta = netDelta.permute(0, 3, 1, 2).contiguous()   # (B,Hh,M,T)
        return netDelta

    def enable_attn_collection(self, on: bool = True):
        """Turn on attention-map stashing for diagnostics."""
        for m in self.modules():
            if isinstance(m, MultiHeadSelfAttention):
                m.collect_attn = on


# ----------------------------------------------------------------------------------
# Smoke test: verify shapes + a single training step run end-to-end
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    from surface_core import build_windows, surface_loss, rw_loss
    torch.manual_seed(0)
    cfg = Config()
    model = SANTA(cfg)
    print(f"SANTA parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    print("netDelta:", tuple(netDelta.shape))

    loss = surface_loss(netDelta, tb, fb)
    base = rw_loss(tb, fb)
    print(f"model loss = {loss.item():.4f}   RW loss = {base.item():.4f}")

    loss.backward()
    print("backward OK, all params have grads:",
          all(p.grad is not None for p in model.parameters() if p.requires_grad))

    model.enable_attn_collection(True)
    _ = model(xb[:2])
    cmap = model.layers[0].block_time.attn._attn   # (B*M*T, heads, L, L)
    print("temporal attn map:", tuple(cmap.shape))
