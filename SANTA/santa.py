"""
santa.py
===================================================================================
SANTA — Surface-Aware Neural Tensor Attention.

  data  : per-cell TRAIN-standardised log-IV, grid M=15 (log-fwd-moneyness in [-0.1,0.1],
          ATM=0) x T=10 (tau in [30/365, 1]yr). Window of L past days -> H horizons.
  step0 : reversible instance-norm = centre each cell's window on TODAY (last slice);
          DO NOT divide out the window scale (amplitude is a regime cue); keep the
          per-cell window std + today's level as features for the head.
  step1 : per-cell value embedding (lift scalar->d) + coordinate embeddings of
          (log-fwd-moneyness, log-tau) + lag embedding.   [financial coords, surface-aware]
  step2 : N layers, each = three FACTORED multi-head self-attention sub-blocks (pre-LN):
            A) over moneyness  (central-smile skew/curvature)
            B) over maturity   (term structure)
            C) over time/lags  (state-dependent persistence / long memory)
  step3 : readout the TODAY token at every cell (it has attended over history+surface).
  step4 : shared multi-horizon head; re-inject today's level + window scale so
          level-dependent mean reversion is learnable.
  step5 : predict the per-cell CHANGE as a RESIDUAL on today; loss = uniform MSE on
          cumulative standardised changes (RW null = predict 0). No importance weights:
          per-cell standardisation already gives correct implicit variance-weighting.

All tensors carry the surface as a (B, L, M, T) object; it is never flattened to a
vector or compressed to factors. Pure PyTorch, no external deps (einops optional).
===================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------
@dataclass
class Config:
    M: int = 11                      # moneyness points
    T: int = 10                      # maturity points
    L: int = 60                      # input window length (trading days)
    horizons: Tuple[int, ...] = (1, 5, 10, 21)   # forecast horizons (days ahead)
    d: int = 48                      # embedding dim (must be divisible by n_heads)
    n_heads: int = 4
    n_layers: int = 2                # N
    d_ff_mult: int = 2               # FFN hidden = d_ff_mult * d
    d_head_hidden: int = 48          # hidden width of the output head MLP
    dropout: float = 0.1
    # Financial coordinates of the grid (length M and T). Defaults match the
    # current preprocessed dataset (`_data_prep/data/optionmetrics_processed/
    # SPX_surfaces.csv`): 11 uniform log-fwd-moneyness strikes in [-0.10, +0.10]
    # at 0.02 spacing (ATM=0 at i=5) and 10 explicit day-count maturities
    # (30, 45, 60, 90, 120, 150, 180, 240, 300, 365 days) in years.
    # build_model in train.py overrides both with the live parsed grid, so
    # these defaults only affect standalone instantiation (the smoke test).
    k_grid: Tuple[float, ...] = tuple(
        round(-0.10 + 0.02 * i, 6) for i in range(11)   # log-fwd-moneyness, ATM=0 at i=5
    )
    tau_grid_years: Tuple[float, ...] = tuple(
        round(d_ / 365.0, 6)
        for d_ in (30, 45, 60, 90, 120, 150, 180, 240, 300, 365)
    )
    centre_on: str = "last"          # "last" (centre on today) or "mean" (window mean)

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)

    @property
    def d_head(self) -> int:
        return self.d // self.n_heads


# ----------------------------------------------------------------------------------
# Coordinate / positional embeddings  (Step 1)
# ----------------------------------------------------------------------------------
class CoordinateEmbedding(nn.Module):
    """MLP mapping a scalar financial coordinate -> R^d. One embedding per grid value.

    Using the *continuous* coordinate (not a grid index) is what makes the model
    surface-aware: proximity in coordinate space => proximity in embedding space,
    and the ATM pivot (k=0) is encoded explicitly.
    """
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d), nn.GELU(), nn.Linear(d, d)
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (n,) -> (n, d)
        return self.net(coords.unsqueeze(-1))


# ----------------------------------------------------------------------------------
# Multi-head self-attention with EXPLICIT Q/K/V  (Step 2 core)
# ----------------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA over the *last-but-one* axis of a (Bx, N, d) tensor.

    Bx is a flattened batch (whatever axes we are NOT attending over); N is the
    sequence length along the attended axis (M, T or L); d is the model dim.
    """
    def __init__(self, d: int, n_heads: int, dropout: float):
        super().__init__()
        assert d % n_heads == 0, "d must be divisible by n_heads"
        self.d, self.h, self.dk = d, n_heads, d // n_heads
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        self.W_o = nn.Linear(d, d, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.collect_attn = False     # set True to stash maps for diagnostics
        self._attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Bx, N, d)
        Bx, N, _ = x.shape
        # project then split into heads -----------------------------------------
        q = self.W_q(x).view(Bx, N, self.h, self.dk).permute(0, 2, 1, 3)  # (Bx,h,N,dk)
        k = self.W_k(x).view(Bx, N, self.h, self.dk).permute(0, 2, 1, 3)  # (Bx,h,N,dk)
        v = self.W_v(x).view(Bx, N, self.h, self.dk).permute(0, 2, 1, 3)  # (Bx,h,N,dk)
        # scaled dot-product attention ------------------------------------------
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (Bx,h,N,N)
        attn = torch.softmax(scores, dim=-1)                                # (Bx,h,N,N)
        if self.collect_attn:
            self._attn = attn.detach()
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)                          # (Bx,h,N,dk)
        # merge heads -----------------------------------------------------------
        out = out.permute(0, 2, 1, 3).reshape(Bx, N, self.d)  # (Bx,N,d)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d: int, d_ff_mult: int, dropout: float):
        super().__init__()
        d_ff = d * d_ff_mult
        self.net = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d)
        )

    def forward(self, x): return self.net(x)


class SubBlock(nn.Module):
    """Pre-LN transformer sub-block operating on (Bx, N, d):
           x <- x + MHSA(LN(x));   x <- x + FFN(LN(x))
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.attn = MultiHeadSelfAttention(cfg.d, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.ff = FeedForward(cfg.d, cfg.d_ff_mult, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


# ----------------------------------------------------------------------------------
# One factored layer: A (moneyness) -> B (maturity) -> C (time)   (Step 2)
# ----------------------------------------------------------------------------------
class FactoredLayer(nn.Module):
    """Applies three SubBlocks, each attending over ONE axis of the (B,L,M,T,d) tensor.

    The reshaping below is the whole trick: to attend over an axis we move that axis
    to the sequence position (-2) and flatten everything else into the batch, run a
    standard MHSA, then restore the original layout. Attention weights are SHARED
    across the flattened batch -> one operator per axis (e.g. a single temporal
    dynamics operator applied at every cell), but its output is state-dependent
    because the inputs (and hence the softmax weights) differ per location/day.
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
        # T is already axis -2 in (B,L,M,T,d), so just flatten the leading dims.
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
        # Coordinate axes the MLP embeddings see:
        #   moneyness  k   directly (log-fwd-moneyness in [-0.10, +0.10]);
        #   maturity   log(tau in years)  — the data's natural metric (the
        #     preprocessing smoother uses a log-T kernel and the explicit
        #     day-count tau grid is roughly log-spaced over [30d, 365d]).
        #   lag        learned discrete embedding over L.
        self.value_proj = nn.Linear(1, cfg.d)                 # lift scalar -> d
        self.emb_money = CoordinateEmbedding(cfg.d)           # pK(k)
        self.emb_mat = CoordinateEmbedding(cfg.d)             # pT(log tau)
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

    # -- Step 0: instance normalisation ------------------------------------------
    def _instance_norm(self, z: torch.Tensor):
        """z: (B, L, M, T) standardised log-IV window (chronological; index -1 = today).
        Returns centred history u (B,L,M,T), today level L0 (B,M,T), window std s~ (B,M,T).
        """
        if self.cfg.centre_on == "last":
            L0 = z[:, -1, :, :]                       # today (B,M,T)
        elif self.cfg.centre_on == "mean":
            L0 = z.mean(dim=1)                         # window mean (B,M,T)
        else:
            raise ValueError(self.cfg.centre_on)
        u = z - L0.unsqueeze(1)                        # centred history (B,L,M,T)
        s_tilde = z.std(dim=1)                         # per-cell window scale (B,M,T)
        # NOTE: we keep z_today as the reconstruction reference regardless of centre_on.
        z_today = z[:, -1, :, :]
        return u, L0, s_tilde, z_today

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, M, T) standardised log-IV window.
        Returns netDelta: (B, n_horizons, M, T) = predicted cumulative standardised
        change from today to each horizon (the residual added to today's surface).
        """
        cfg = self.cfg
        B, L, M, T = z.shape
        assert (L, M, T) == (cfg.L, cfg.M, cfg.T), "window shape mismatch with config"

        # Step 0 ------------------------------------------------------------------
        u, L0, s_tilde, z_today = self._instance_norm(z)

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

    # -- convenience: turn on attention-map collection for diagnostics -----------
    def enable_attn_collection(self, on: bool = True):
        for m in self.modules():
            if isinstance(m, MultiHeadSelfAttention):
                m.collect_attn = on


# ----------------------------------------------------------------------------------
# Reconstruction + Loss  (Step 5)
# ----------------------------------------------------------------------------------
def reconstruct(z_today: torch.Tensor, netDelta: torch.Tensor) -> torch.Tensor:
    """zhat_{t+h} = z_today + netDelta_h.  z_today (B,M,T); netDelta (B,Hh,M,T)."""
    return z_today.unsqueeze(1) + netDelta                     # (B,Hh,M,T)


def surface_loss(netDelta: torch.Tensor,
                 z_today: torch.Tensor,
                 z_future: torch.Tensor,
                 gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Uniform MSE on cumulative standardised changes.

      netDelta : (B,Hh,M,T)  predicted change to each horizon
      z_today  : (B,M,T)     standardised log-IV today
      z_future : (B,Hh,M,T)  standardised log-IV at each horizon (the targets)
      gamma    : (Hh,)       horizon weights (default ones)

    RW null = netDelta == 0  =>  loss = mean over cells/horizons of (Dz)^2.
    No w(m,tau): per-cell standardisation already equalises cell scales.
    """
    B, Hh, M, T = netDelta.shape
    Dz = z_future - z_today.unsqueeze(1)                       # (B,Hh,M,T) target change
    sq = (netDelta - Dz) ** 2                                  # (B,Hh,M,T)
    per_h = sq.mean(dim=(0, 2, 3))                             # (Hh,)  mean over batch+cells
    if gamma is None:
        gamma = torch.ones(Hh, device=netDelta.device)
    gamma = gamma / gamma.sum() * Hh                           # normalise so mean weight = 1
    return (gamma * per_h).mean()


def rw_loss(z_today: torch.Tensor, z_future: torch.Tensor,
            gamma: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Random-walk baseline loss = surface_loss with netDelta == 0 (predict no change)."""
    netDelta = torch.zeros_like(z_future)
    return surface_loss(netDelta, z_today, z_future, gamma)


def destandardise(z: torch.Tensor, mu_cell: torch.Tensor,
                  sd_cell: torch.Tensor) -> torch.Tensor:
    """Map standardised log-IV back to log-IV. mu_cell, sd_cell: (M,T) train stats.
    z: (..., M, T). Returns log-IV in the same shape. (Then exp() for vol if desired.)"""
    return z * sd_cell + mu_cell


# ----------------------------------------------------------------------------------
# Window builder (honours the chronological split + H-day embargo externally)
# ----------------------------------------------------------------------------------
def build_windows(Z: torch.Tensor, horizons: Tuple[int, ...], L: int
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z: (S, M, T) chronological standardised log-IV series (stats from TRAIN only).
    Returns
      windows  : (Nsamp, L, M, T)  input windows (index -1 = base date t)
      z_today  : (Nsamp, M, T)
      z_future : (Nsamp, Hh, M, T)
      base_idx : (Nsamp,)          the integer index t of each sample (for split masks)

    Build per split by passing the slice of Z for that split, OR build over the whole
    series and then select base_idx ranges per split with an H_max-day embargo so no
    target window straddles a boundary. Standardisation must already be applied to Z
    using TRAIN-ONLY mu/sd.
    """
    S, M, T = Z.shape
    Hmax = max(horizons)
    samples_w, samples_today, samples_fut, idxs = [], [], [], []
    for t in range(L - 1, S - Hmax):
        samples_w.append(Z[t - L + 1: t + 1])                          # (L,M,T)
        samples_today.append(Z[t])                                     # (M,T)
        samples_fut.append(torch.stack([Z[t + h] for h in horizons]))  # (Hh,M,T)
        idxs.append(t)
    return (torch.stack(samples_w), torch.stack(samples_today),
            torch.stack(samples_fut), torch.tensor(idxs))


# ----------------------------------------------------------------------------------
# Smoke test: verify shapes + a single training step run end-to-end
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Config()
    model = SANTA(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    # fake standardised series: 300 days of a 15x10 surface
    S = 300
    Z = torch.randn(S, cfg.M, cfg.T).cumsum(0) * 0.02   # random-walk-ish, standardised-scale
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)             # pretend train-standardised

    windows, z_today, z_future, base_idx = build_windows(Z, cfg.horizons, cfg.L)
    print("windows :", tuple(windows.shape))      # (Nsamp, L, M, T)
    print("z_today :", tuple(z_today.shape))       # (Nsamp, M, T)
    print("z_future:", tuple(z_future.shape))      # (Nsamp, Hh, M, T)

    # one minibatch
    bs = 64
    xb, tb, fb = windows[:bs], z_today[:bs], z_future[:bs]
    netDelta = model(xb)
    print("netDelta:", tuple(netDelta.shape))      # (bs, Hh, M, T)

    loss = surface_loss(netDelta, tb, fb)
    base = rw_loss(tb, fb)
    print(f"model loss = {loss.item():.4f}   RW loss = {base.item():.4f}")

    loss.backward()                                # confirm gradients flow
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print("backward OK, all params have grads:", grad_ok)

    # diagnostics: pull a temporal-attention map (Block C, layer 0) ---------------
    model.enable_attn_collection(True)
    _ = model(xb[:2])
    cmap = model.layers[0].block_time.attn._attn   # (B*M*T, heads, L, L)
    print("temporal attn map:", tuple(cmap.shape))
